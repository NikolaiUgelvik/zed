use crate::{AgentTool, ToolCallEventStream, ToolInput, semantic_search};
use agent_client_protocol::schema as acp;
use agent_settings::AgentSettings;
use anyhow::{Context as _, Result};
use futures::FutureExt as _;
use gpui::{App, Entity, SharedString, Task};
use language_model::LanguageModelToolResultContent;
use project::Project;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use settings::Settings;
use std::{fmt::Write, ops::Range, sync::Arc};
use util::{ResultExt as _, markdown::MarkdownInlineCode};

/// Searches code semantically using a natural language query.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct SemanticSearchCodeToolInput {
    /// The natural language code search query.
    pub query: String,
    /// A glob pattern for the paths of files to include in the search.
    pub include_pattern: Option<String>,
    /// The maximum number of matches to return.
    pub max_results: Option<usize>,
    /// Whether to include expanded surrounding context for each match.
    pub expand_context: Option<bool>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum SemanticSearchCodeToolOutput {
    Disabled,
    ConfigurationError {
        message: String,
    },
    Success {
        matches: Vec<SemanticSearchCodeMatch>,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SemanticSearchCodeMatch {
    pub path: String,
    pub line_range: Range<u32>,
    pub score: f64,
    pub source: String,
    pub snippet: String,
    pub expanded_context: Vec<String>,
}

impl From<SemanticSearchCodeToolOutput> for LanguageModelToolResultContent {
    fn from(output: SemanticSearchCodeToolOutput) -> Self {
        match output {
            SemanticSearchCodeToolOutput::Disabled => concat!(
                "Semantic code search is disabled. ",
                "Enable it by setting `agent.semantic_search.enabled` to true."
            )
            .into(),
            SemanticSearchCodeToolOutput::ConfigurationError { message } => message.into(),
            SemanticSearchCodeToolOutput::Success { matches } => {
                if matches.is_empty() {
                    return "No semantic code search matches found".into();
                }

                let mut output = format!("Found {} semantic code search match", matches.len());
                if matches.len() != 1 {
                    output.push_str("es");
                }
                output.push(':');

                for mat in matches {
                    let start_line = mat.line_range.start.saturating_add(1);
                    write!(
                        &mut output,
                        "\n\n## {}:{}-{}\nscore: {:.3}\nsource: {}\n\n```\n{}\n```",
                        mat.path,
                        start_line,
                        mat.line_range.end,
                        mat.score,
                        mat.source,
                        mat.snippet,
                    )
                    .ok();

                    if !mat.expanded_context.is_empty() {
                        output.push_str("\n\nExpanded context:");
                        for context in mat.expanded_context {
                            write!(&mut output, "\n\n```\n{context}\n```").ok();
                        }
                    }
                }

                output.into()
            }
        }
    }
}

trait SemanticSearchProviderFactory: Send + Sync {
    fn providers(
        &self,
        settings: &agent_settings::SemanticSearchSettings,
        cx: &gpui::App,
    ) -> Result<crate::semantic_search::SemanticSearchProviderSet, String>;
}

trait SemanticSearchHyDEGenerator: Send + Sync {
    fn generate(
        &self,
        hyde_settings: agent_settings::SemanticSearchHyDESettings,
        query: String,
        cx: &mut gpui::AsyncApp,
    ) -> Task<anyhow::Result<String>>;
}

struct LanguageModelSemanticSearchHyDEGenerator;

impl SemanticSearchHyDEGenerator for LanguageModelSemanticSearchHyDEGenerator {
    fn generate(
        &self,
        hyde_settings: agent_settings::SemanticSearchHyDESettings,
        query: String,
        cx: &mut gpui::AsyncApp,
    ) -> Task<anyhow::Result<String>> {
        cx.spawn(async move |cx| {
            semantic_search::generate_hyde_search_text_from_settings(&hyde_settings, &query, cx)
                .await
        })
    }
}

struct HttpSemanticSearchProviderFactory {
    http_client: Arc<dyn http_client::HttpClient>,
}

impl SemanticSearchProviderFactory for HttpSemanticSearchProviderFactory {
    fn providers(
        &self,
        settings: &agent_settings::SemanticSearchSettings,
        cx: &gpui::App,
    ) -> Result<crate::semantic_search::SemanticSearchProviderSet, String> {
        let embedding = settings.embedding.as_ref().ok_or_else(|| {
            "Semantic code search requires `agent.semantic_search.embedding`.".to_string()
        })?;
        let reranker = settings.reranker.as_ref().ok_or_else(|| {
            "Semantic code search requires `agent.semantic_search.reranker`.".to_string()
        })?;
        let embedding_config = crate::semantic_search::resolve_semantic_provider_config(
            embedding,
            crate::semantic_search::SemanticSearchProviderRole::Embedding,
            cx,
        )?;
        let reranker_config = crate::semantic_search::resolve_semantic_provider_config(
            reranker,
            crate::semantic_search::SemanticSearchProviderRole::Reranker,
            cx,
        )?;

        Ok(crate::semantic_search::SemanticSearchProviderSet {
            embedding: Arc::new(
                crate::semantic_search::OpenAiCompatibleEmbeddingProvider::new(
                    self.http_client.clone(),
                    embedding_config.api_url,
                    embedding_config.api_key.unwrap_or_default(),
                    embedding_config.model,
                ),
            ),
            reranker: Arc::new(crate::semantic_search::JinaRerankerProvider::new(
                self.http_client.clone(),
                reranker_config.api_url,
                reranker_config.api_key.unwrap_or_default(),
                reranker_config.model,
            )),
        })
    }
}

pub struct SemanticSearchCodeTool {
    project: Entity<Project>,
    provider_factory: Arc<dyn SemanticSearchProviderFactory>,
    hyde_generator: Arc<dyn SemanticSearchHyDEGenerator>,
}

impl SemanticSearchCodeTool {
    pub fn new(project: Entity<Project>, http_client: Arc<dyn http_client::HttpClient>) -> Self {
        Self {
            project,
            provider_factory: Arc::new(HttpSemanticSearchProviderFactory { http_client }),
            hyde_generator: Arc::new(LanguageModelSemanticSearchHyDEGenerator),
        }
    }

    #[cfg(test)]
    fn new_for_test(
        project: Entity<Project>,
        provider_factory: impl SemanticSearchProviderFactory + 'static,
    ) -> Self {
        Self {
            project,
            provider_factory: Arc::new(provider_factory),
            hyde_generator: Arc::new(LanguageModelSemanticSearchHyDEGenerator),
        }
    }

    #[cfg(test)]
    fn new_for_test_with_hyde_generator(
        project: Entity<Project>,
        provider_factory: impl SemanticSearchProviderFactory + 'static,
        hyde_generator: impl SemanticSearchHyDEGenerator + 'static,
    ) -> Self {
        Self {
            project,
            provider_factory: Arc::new(provider_factory),
            hyde_generator: Arc::new(hyde_generator),
        }
    }
}

impl AgentTool for SemanticSearchCodeTool {
    type Input = SemanticSearchCodeToolInput;
    type Output = SemanticSearchCodeToolOutput;

    const NAME: &'static str = "semantic_search_code";

    fn kind() -> acp::ToolKind {
        acp::ToolKind::Search
    }

    fn initial_title(
        &self,
        input: Result<Self::Input, serde_json::Value>,
        _cx: &mut App,
    ) -> SharedString {
        match input {
            Ok(input) => format!(
                "Semantically search code for {}",
                MarkdownInlineCode(&input.query)
            )
            .into(),
            Err(_) => "Semantically search code".into(),
        }
    }

    fn run(
        self: Arc<Self>,
        input: ToolInput<Self::Input>,
        event_stream: ToolCallEventStream,
        cx: &mut App,
    ) -> Task<Result<Self::Output, Self::Output>> {
        cx.spawn(async move |cx| {
            let input = with_semantic_search_cancellation(input.recv(), &event_stream)
                .await?
                .map_err(|error| SemanticSearchCodeToolOutput::ConfigurationError {
                    message: format!("Failed to receive semantic code search input: {error}"),
                })?;

            if event_stream.was_cancelled_by_user() {
                return Err(semantic_search_cancelled_output());
            }

            let settings = cx.update(|cx| AgentSettings::get_global(cx).semantic_search.clone());
            if let Err(message) = semantic_search::validate_semantic_search_configuration(&settings)
            {
                if !settings.enabled {
                    return Ok(SemanticSearchCodeToolOutput::Disabled);
                }

                return Err(SemanticSearchCodeToolOutput::ConfigurationError { message });
            }

            let runtime_settings = semantic_search::SemanticSearchRuntimeSettings::from_settings(
                &settings,
                input.max_results,
                input.expand_context,
            );
            let indexing_settings = semantic_search::SemanticSearchIndexingSettings {
                include_pattern: input.include_pattern.clone(),
                max_indexed_file_bytes: settings.max_indexed_file_bytes,
                chunk_max_non_whitespace_size: settings.chunk_max_non_whitespace_size,
            };
            let providers = cx
                .update(|cx| self.provider_factory.providers(&settings, cx))
                .map_err(|message| SemanticSearchCodeToolOutput::ConfigurationError { message })?;

            event_stream.update_fields(
                acp::ToolCallUpdateFields::new()
                    .title("Semantic code search")
                    .content(vec![acp::ToolCallContent::Content(acp::Content::new(
                        "Building semantic code index...",
                    ))]),
            );
            let index = with_semantic_search_cancellation(
                semantic_search::build_project_semantic_index(
                    self.project.clone(),
                    indexing_settings,
                    cx,
                ),
                &event_stream,
            )
            .await?
            .map_err(|error| SemanticSearchCodeToolOutput::ConfigurationError {
                message: format!("Failed to build semantic search index: {error}"),
            })?;
            let index = with_semantic_search_cancellation(
                semantic_search::embed_code_search_index(index, providers.embedding.as_ref()),
                &event_stream,
            )
            .await?
            .map_err(|error| SemanticSearchCodeToolOutput::ConfigurationError {
                message: format!("Failed to embed semantic search index: {error}"),
            })?;

            event_stream.update_fields(acp::ToolCallUpdateFields::new().content(vec![
                acp::ToolCallContent::Content(acp::Content::new(
                    "Running semantic code query...",
                )),
            ]));
            let mut result = with_semantic_search_cancellation(
                semantic_search::run_semantic_search_query(
                    &index,
                    &providers,
                    &input.query,
                    &runtime_settings,
                    None,
                ),
                &event_stream,
            )
            .await?
            .map_err(|error| SemanticSearchCodeToolOutput::ConfigurationError {
                message: format!("Semantic code search failed: {error}"),
            })?;
            let should_run_hyde = settings.hyde.mode == agent_settings::HyDEMode::Fallback
                && settings.hyde.model.is_some()
                && semantic_search::should_run_hyde_fallback(
                    result.matches.first().map(|search_match| search_match.rerank_score),
                    runtime_settings.hyde_threshold,
                );
            if should_run_hyde {
                let hyde_search_text = with_semantic_search_cancellation(
                    self.hyde_generator
                        .generate(settings.hyde.clone(), input.query.clone(), cx),
                    &event_stream,
                )
                .await?
                .context("HyDE search text generation failed")
                .log_err()
                .and_then(|text| {
                        if text.is_empty() {
                            log::info!("HyDE search text generation returned empty text; skipping HyDE fallback");
                            None
                        } else {
                            Some(text)
                        }
                    });

                if let Some(hyde_search_text) = hyde_search_text {
                    event_stream.update_fields(acp::ToolCallUpdateFields::new().content(vec![
                        acp::ToolCallContent::Content(acp::Content::new(
                            "Running HyDE semantic code fallback...",
                        )),
                    ]));
                    result = with_semantic_search_cancellation(
                        semantic_search::run_semantic_search_hyde_fallback(
                            &index,
                            &providers,
                            &runtime_settings,
                            result,
                            hyde_search_text,
                        ),
                        &event_stream,
                    )
                    .await?
                    .map_err(|error| SemanticSearchCodeToolOutput::ConfigurationError {
                        message: format!("Semantic code search failed: {error}"),
                    })?;
                }
            }
            let matches = format_semantic_search_matches(
                &index,
                result,
                &runtime_settings.topology_expansion,
            );
            if event_stream.was_cancelled_by_user() {
                return Err(semantic_search_cancelled_output());
            }
            let output = SemanticSearchCodeToolOutput::Success { matches };
            if let LanguageModelToolResultContent::Text(text) =
                LanguageModelToolResultContent::from(output.clone())
            {
                event_stream.update_fields(acp::ToolCallUpdateFields::new().content(vec![
                    acp::ToolCallContent::Content(acp::Content::new(text.to_string())),
                ]));
            }
            Ok(output)
        })
    }
}

async fn with_semantic_search_cancellation<T>(
    future: impl std::future::Future<Output = T>,
    event_stream: &ToolCallEventStream,
) -> Result<T, SemanticSearchCodeToolOutput> {
    futures::pin_mut!(future);
    futures::select! {
        result = future.fuse() => {
            if event_stream.was_cancelled_by_user() {
                Err(semantic_search_cancelled_output())
            } else {
                Ok(result)
            }
        },
        _ = event_stream.cancelled_by_user().fuse() => Err(semantic_search_cancelled_output()),
    }
}

fn semantic_search_cancelled_output() -> SemanticSearchCodeToolOutput {
    SemanticSearchCodeToolOutput::ConfigurationError {
        message: "Semantic code search was cancelled by the user.".to_string(),
    }
}

fn format_semantic_search_matches(
    index: &semantic_search::CodeSearchIndex,
    result: semantic_search::SemanticSearchResult,
    topology_settings: &semantic_search::TopologyExpansionRuntimeSettings,
) -> Vec<SemanticSearchCodeMatch> {
    result
        .matches
        .into_iter()
        .filter_map(|search_match| {
            let chunk = index.chunk(search_match.chunk_id)?;
            let expanded_ids = semantic_search::expand_topology(
                index,
                &[search_match.chunk_id],
                topology_settings,
            );
            let expanded_context = expanded_ids
                .into_iter()
                .filter(|expanded_id| *expanded_id != search_match.chunk_id)
                .filter_map(|expanded_id| index.chunk(expanded_id).map(|chunk| chunk.text.clone()))
                .collect();
            Some(SemanticSearchCodeMatch {
                path: chunk.path.to_string_lossy().replace('\\', "/"),
                line_range: chunk.line_range.clone(),
                score: f64::from(search_match.rerank_score),
                source: search_match.source.to_string(),
                snippet: chunk.text.clone(),
                expanded_context,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use language_model::LanguageModelToolResultContent;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[gpui::test]
    async fn semantic_search_code_returns_ranked_project_snippets(cx: &mut gpui::TestAppContext) {
        use fs::FakeFs;
        use serde_json::json;

        crate::tests::init_test(cx);
        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(
            "/root",
            json!({
                "src": {
                    "auth.rs": "fn retry_login() { backoff(); }",
                    "ui.rs": "fn render_button() {}"
                }
            }),
        )
        .await;
        let project = Project::test(fs, ["/root".as_ref()], cx).await;
        let providers = FakeSemanticSearchProviderFactory::new(
            vec![
                ("retry login".to_string(), vec![1.0, 0.0]),
                (
                    "fn retry_login() { backoff(); }".to_string(),
                    vec![0.9, 0.1],
                ),
                ("fn render_button() {}".to_string(), vec![0.1, 0.9]),
            ],
            vec![(0, 0.93), (1, 0.1)],
        );
        let tool = Arc::new(SemanticSearchCodeTool::new_for_test(project, providers));

        cx.update(|cx| {
            let mut settings = agent_settings::AgentSettings::get_global(cx).clone();
            settings.semantic_search.enabled = true;
            settings.semantic_search.embedding = Some(test_model_selection(
                "local-embeddings",
                "embedding-model",
                agent_settings::SemanticSearchApiFormat::OpenAiEmbeddings,
            ));
            settings.semantic_search.reranker = Some(test_model_selection(
                "local-reranker",
                "reranker-model",
                agent_settings::SemanticSearchApiFormat::JinaRerank,
            ));
            agent_settings::AgentSettings::override_global(settings, cx);
        });

        let output = cx
            .update(|cx| {
                tool.run(
                    ToolInput::resolved(SemanticSearchCodeToolInput {
                        query: "retry login".to_string(),
                        include_pattern: Some("root/src/**/*.rs".to_string()),
                        max_results: Some(1),
                        expand_context: Some(false),
                    }),
                    ToolCallEventStream::test().0,
                    cx,
                )
            })
            .await
            .expect("tool should succeed");

        let SemanticSearchCodeToolOutput::Success { matches } = output else {
            panic!("expected success output");
        };
        assert_eq!(matches.len(), 1);
        assert!(matches[0].path.ends_with("src/auth.rs"));
        assert!(matches[0].snippet.contains("retry_login"));
    }

    #[gpui::test]
    async fn semantic_search_code_runs_hyde_after_low_score_first_pass(
        cx: &mut gpui::TestAppContext,
    ) {
        use fs::FakeFs;
        use serde_json::json;

        crate::tests::init_test(cx);
        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(
            "/root",
            json!({
                "src": {
                    "auth.rs": "fn retry_login() { backoff(); }",
                    "ui.rs": "fn render_button() {}"
                }
            }),
        )
        .await;
        let project = Project::test(fs, ["/root".as_ref()], cx).await;
        let providers = FakeSemanticSearchProviderFactory::new_sequence(
            vec![
                ("unclear request".to_string(), vec![0.1, 0.9]),
                ("retry backoff implementation".to_string(), vec![1.0, 0.0]),
                (
                    "fn retry_login() { backoff(); }".to_string(),
                    vec![0.9, 0.1],
                ),
                ("fn render_button() {}".to_string(), vec![0.1, 0.9]),
            ],
            vec![vec![(1, 0.2), (0, 0.1)], vec![(0, 0.91), (1, 0.1)]],
        );
        let hyde_generator = FakeHyDEGenerator::new(Ok("retry backoff implementation".to_string()));
        let calls = hyde_generator.calls.clone();
        let tool = Arc::new(SemanticSearchCodeTool::new_for_test_with_hyde_generator(
            project,
            providers,
            hyde_generator,
        ));

        set_semantic_search_settings(cx, agent_settings::HyDEMode::Fallback, 0.6, true);

        let output = run_tool(tool, "unclear request", cx).await;
        let SemanticSearchCodeToolOutput::Success { matches } = output else {
            panic!("expected success output");
        };
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(matches[0].source, "hyde_fallback");
        assert!(matches[0].path.ends_with("src/auth.rs"));
    }

    #[gpui::test]
    async fn semantic_search_cancellation_checks_cancelled_state_after_future_completes(
        _cx: &mut gpui::TestAppContext,
    ) {
        let (event_stream, _receiver, mut cancellation_tx) =
            ToolCallEventStream::test_with_cancellation();

        let output = with_semantic_search_cancellation(
            async move {
                ToolCallEventStream::signal_cancellation_with_sender(&mut cancellation_tx);
                "completed"
            },
            &event_stream,
        )
        .await
        .expect_err("helper should return cancellation when cancellation was already signaled");
        let content = LanguageModelToolResultContent::from(output);
        let content = content.to_str().unwrap();
        assert!(content.to_lowercase().contains("cancel"));
    }

    #[gpui::test]
    async fn semantic_search_code_cancels_before_provider_call(cx: &mut gpui::TestAppContext) {
        use fs::FakeFs;
        use serde_json::json;

        crate::tests::init_test(cx);
        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(
            "/root",
            json!({ "src": { "auth.rs": "fn retry_login() {}" } }),
        )
        .await;
        let project = Project::test(fs, ["/root".as_ref()], cx).await;
        let providers = CountingSemanticSearchProviderFactory::new();
        let provider_calls = providers.calls.clone();
        let tool = Arc::new(SemanticSearchCodeTool::new_for_test(project, providers));
        let (event_stream, _receiver, mut cancellation_tx) =
            ToolCallEventStream::test_with_cancellation();

        set_semantic_search_settings(cx, agent_settings::HyDEMode::Off, 0.6, false);

        let task = cx.update(|cx| {
            tool.run(
                ToolInput::resolved(SemanticSearchCodeToolInput {
                    query: "retry login".to_string(),
                    include_pattern: Some("root/src/**/*.rs".to_string()),
                    max_results: Some(1),
                    expand_context: Some(false),
                }),
                event_stream,
                cx,
            )
        });

        ToolCallEventStream::signal_cancellation_with_sender(&mut cancellation_tx);
        let output = task
            .await
            .expect_err("tool should return cancellation error");
        let content = LanguageModelToolResultContent::from(output);
        let content = content.to_str().unwrap();
        assert!(content.to_lowercase().contains("cancel"));
        assert_eq!(provider_calls.load(Ordering::SeqCst), 0);
    }

    #[gpui::test]
    async fn semantic_search_code_skips_hyde_when_first_pass_score_is_high(
        cx: &mut gpui::TestAppContext,
    ) {
        use fs::FakeFs;
        use serde_json::json;

        crate::tests::init_test(cx);
        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(
            "/root",
            json!({
                "src": {
                    "auth.rs": "fn retry_login() { backoff(); }",
                    "ui.rs": "fn render_button() {}"
                }
            }),
        )
        .await;
        let project = Project::test(fs, ["/root".as_ref()], cx).await;
        let providers = FakeSemanticSearchProviderFactory::new(
            vec![
                ("retry login".to_string(), vec![1.0, 0.0]),
                (
                    "fn retry_login() { backoff(); }".to_string(),
                    vec![0.9, 0.1],
                ),
                ("fn render_button() {}".to_string(), vec![0.1, 0.9]),
            ],
            vec![(0, 0.93), (1, 0.1)],
        );
        let hyde_generator = FakeHyDEGenerator::new(Ok("unused hyde text".to_string()));
        let calls = hyde_generator.calls.clone();
        let tool = Arc::new(SemanticSearchCodeTool::new_for_test_with_hyde_generator(
            project,
            providers,
            hyde_generator,
        ));

        set_semantic_search_settings(cx, agent_settings::HyDEMode::Fallback, 0.6, true);

        let output = run_tool(tool, "retry login", cx).await;
        let SemanticSearchCodeToolOutput::Success { matches } = output else {
            panic!("expected success output");
        };
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        assert_eq!(matches[0].source, "reranked");
        assert!(matches[0].path.ends_with("src/auth.rs"));
    }

    struct FakeSemanticSearchProviderFactory {
        providers: crate::semantic_search::SemanticSearchProviderSet,
    }

    impl FakeSemanticSearchProviderFactory {
        fn new(embeddings: Vec<(String, Vec<f32>)>, reranks: Vec<(usize, f32)>) -> Self {
            Self::new_sequence(embeddings, vec![reranks])
        }

        fn new_sequence(
            embeddings: Vec<(String, Vec<f32>)>,
            reranks: Vec<Vec<(usize, f32)>>,
        ) -> Self {
            Self {
                providers: crate::semantic_search::SemanticSearchProviderSet {
                    embedding: Arc::new(crate::semantic_search::FakeEmbeddingProvider::new(
                        embeddings,
                    )),
                    reranker: Arc::new(crate::semantic_search::FakeRerankerProvider::new_sequence(
                        reranks,
                    )),
                },
            }
        }
    }

    struct CountingSemanticSearchProviderFactory {
        calls: Arc<AtomicUsize>,
    }

    impl CountingSemanticSearchProviderFactory {
        fn new() -> Self {
            Self {
                calls: Arc::new(AtomicUsize::new(0)),
            }
        }
    }

    impl SemanticSearchProviderFactory for CountingSemanticSearchProviderFactory {
        fn providers(
            &self,
            _settings: &agent_settings::SemanticSearchSettings,
            _cx: &gpui::App,
        ) -> Result<crate::semantic_search::SemanticSearchProviderSet, String> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(FakeSemanticSearchProviderFactory::new(
                vec![
                    ("retry login".to_string(), vec![1.0]),
                    ("fn retry_login() {}".to_string(), vec![1.0]),
                ],
                vec![(0, 0.9)],
            )
            .providers)
        }
    }

    struct FakeHyDEGenerator {
        result: anyhow::Result<String>,
        calls: Arc<AtomicUsize>,
    }

    impl FakeHyDEGenerator {
        fn new(result: anyhow::Result<String>) -> Self {
            Self {
                result,
                calls: Arc::new(AtomicUsize::new(0)),
            }
        }
    }

    impl SemanticSearchHyDEGenerator for FakeHyDEGenerator {
        fn generate(
            &self,
            _hyde_settings: agent_settings::SemanticSearchHyDESettings,
            _query: String,
            _cx: &mut gpui::AsyncApp,
        ) -> Task<anyhow::Result<String>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let result = self
                .result
                .as_ref()
                .map(|text| text.clone())
                .map_err(|error| anyhow::anyhow!(error.to_string()));
            Task::ready(result)
        }
    }

    async fn run_tool(
        tool: Arc<SemanticSearchCodeTool>,
        query: &str,
        cx: &mut gpui::TestAppContext,
    ) -> SemanticSearchCodeToolOutput {
        cx.update(|cx| {
            tool.run(
                ToolInput::resolved(SemanticSearchCodeToolInput {
                    query: query.to_string(),
                    include_pattern: Some("root/src/**/*.rs".to_string()),
                    max_results: Some(1),
                    expand_context: Some(false),
                }),
                ToolCallEventStream::test().0,
                cx,
            )
        })
        .await
        .expect("tool should succeed")
    }

    fn set_semantic_search_settings(
        cx: &mut gpui::TestAppContext,
        hyde_mode: agent_settings::HyDEMode,
        hyde_threshold: f32,
        include_hyde_model: bool,
    ) {
        cx.update(|cx| {
            let mut settings = agent_settings::AgentSettings::get_global(cx).clone();
            settings.semantic_search.enabled = true;
            settings.semantic_search.embedding = Some(test_model_selection(
                "local-embeddings",
                "embedding-model",
                agent_settings::SemanticSearchApiFormat::OpenAiEmbeddings,
            ));
            settings.semantic_search.reranker = Some(test_model_selection(
                "local-reranker",
                "reranker-model",
                agent_settings::SemanticSearchApiFormat::JinaRerank,
            ));
            settings.semantic_search.hyde.mode = hyde_mode;
            settings.semantic_search.hyde.threshold = hyde_threshold;
            settings.semantic_search.hyde.model =
                include_hyde_model.then(|| settings::LanguageModelSelection {
                    provider: "local-hyde".to_string().into(),
                    model: "hyde-model".to_string(),
                    enable_thinking: false,
                    effort: None,
                    speed: None,
                });
            agent_settings::AgentSettings::override_global(settings, cx);
        });
    }

    impl SemanticSearchProviderFactory for FakeSemanticSearchProviderFactory {
        fn providers(
            &self,
            _settings: &agent_settings::SemanticSearchSettings,
            _cx: &gpui::App,
        ) -> Result<crate::semantic_search::SemanticSearchProviderSet, String> {
            Ok(self.providers.clone())
        }
    }

    fn test_model_selection(
        provider: &str,
        model: &str,
        api_format: agent_settings::SemanticSearchApiFormat,
    ) -> agent_settings::SemanticSearchModelSelection {
        agent_settings::SemanticSearchModelSelection {
            provider: provider.into(),
            model: model.to_string(),
            api_format,
        }
    }

    #[test]
    fn disabled_output_is_clear_for_llm() {
        let output = SemanticSearchCodeToolOutput::Disabled;
        let content = LanguageModelToolResultContent::from(output);
        let content = content.to_str().unwrap();
        assert!(content.contains("Semantic code search is disabled"));
        assert!(content.contains("agent.semantic_search.enabled"));
    }

    #[test]
    fn result_output_includes_path_line_score_and_snippet() {
        let output = SemanticSearchCodeToolOutput::Success {
            matches: vec![SemanticSearchCodeMatch {
                path: "root/src/player.rs".to_string(),
                line_range: 10..14,
                score: 0.87,
                source: "reranked".to_string(),
                snippet: "fn movement_speed(&self) -> f32 { self.speed }".to_string(),
                expanded_context: Vec::new(),
            }],
        };
        let content = LanguageModelToolResultContent::from(output);
        let content = content.to_str().unwrap();
        assert!(content.contains("root/src/player.rs:11-14"));
        assert!(content.contains("score: 0.870"));
        assert!(content.contains("movement_speed"));
    }

    #[test]
    fn success_output_includes_expanded_parent_context() {
        let output = SemanticSearchCodeToolOutput::Success {
            matches: vec![SemanticSearchCodeMatch {
                path: "root/src/player.rs".to_string(),
                line_range: 20..22,
                score: 0.91,
                source: "hyde_fallback".to_string(),
                snippet: "fn movement_speed(&self) -> f32 { self.speed }".to_string(),
                expanded_context: vec!["impl Player { ... }".to_string()],
            }],
        };
        let content = LanguageModelToolResultContent::from(output);
        let content = content.to_str().unwrap();
        assert!(content.contains("hyde_fallback"));
        assert!(content.contains("source: hyde_fallback"));
        assert!(content.contains("Expanded context"));
        assert!(content.contains("impl Player"));
    }
}
