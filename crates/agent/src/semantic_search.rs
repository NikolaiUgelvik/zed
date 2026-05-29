use anyhow::anyhow;
use futures::{AsyncReadExt as _, FutureExt as _, StreamExt as _, future::BoxFuture};
use gpui::{App, AsyncApp, Entity, Task};
use http_client::{AsyncBody, HttpClient, Method, Request as HttpRequest};
use language_model::{
    CompletionIntent, LanguageModel, LanguageModelId, LanguageModelProviderId,
    LanguageModelRegistry, LanguageModelRequest, LanguageModelRequestMessage,
    LanguageModelToolChoice, MessageContent, Role, SelectedModel, Speed,
};
use serde::{Deserialize, Serialize};
use settings::Settings as _;
use std::{
    collections::{HashMap, HashSet},
    fmt,
    net::{IpAddr, Ipv4Addr, Ipv6Addr},
    path::PathBuf,
    sync::Arc,
};
use util::ResultExt as _;

#[cfg(test)]
use std::collections::VecDeque;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub(crate) struct CodeChunkId(pub(crate) usize);

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct IndexedCodeChunk {
    pub(crate) id: CodeChunkId,
    pub(crate) worktree_id: Option<u64>,
    pub(crate) path: Arc<PathBuf>,
    pub(crate) byte_range: std::ops::Range<usize>,
    pub(crate) line_range: std::ops::Range<u32>,
    pub(crate) text: String,
    pub(crate) non_whitespace_size: usize,
    pub(crate) primary_node_kind: Option<String>,
    pub(crate) topology: CodeChunkTopology,
    pub(crate) embedding: Vec<f32>,
    pub(crate) lexical_terms: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct CodeChunkTopology {
    pub(crate) parent: Option<CodeChunkId>,
    pub(crate) children: Vec<CodeChunkId>,
    pub(crate) previous_sibling: Option<CodeChunkId>,
    pub(crate) next_sibling: Option<CodeChunkId>,
    pub(crate) enclosing_symbols: Vec<EnclosingSymbol>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct EnclosingSymbol {
    pub(crate) name: String,
    pub(crate) kind: Option<String>,
    pub(crate) byte_range: std::ops::Range<usize>,
    pub(crate) line_range: std::ops::Range<u32>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ScoredChunk {
    pub(crate) chunk_id: CodeChunkId,
    pub(crate) score: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct RerankResult {
    pub(crate) document_index: usize,
    pub(crate) score: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct SemanticSearchMatch {
    pub(crate) chunk_id: CodeChunkId,
    pub(crate) rerank_score: f32,
    pub(crate) source: &'static str,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct SemanticSearchResult {
    pub(crate) matches: Vec<SemanticSearchMatch>,
    pub(crate) used_hyde: bool,
}

#[derive(Clone, Debug)]
pub(crate) struct SemanticSearchIndexingSettings {
    pub(crate) include_pattern: Option<String>,
    pub(crate) max_indexed_file_bytes: usize,
    pub(crate) chunk_max_non_whitespace_size: usize,
}

pub(crate) trait CodeSearchEmbeddingProvider: Send + Sync {
    fn embed(&self, text: String) -> BoxFuture<'static, anyhow::Result<Vec<f32>>>;
    fn embed_batch(&self, texts: Vec<String>) -> BoxFuture<'static, anyhow::Result<Vec<Vec<f32>>>>;
}

pub(crate) trait CodeSearchRerankerProvider: Send + Sync {
    fn rerank(
        &self,
        query: String,
        documents: Vec<String>,
    ) -> BoxFuture<'static, anyhow::Result<Vec<RerankResult>>>;

    fn rerank_top_n(
        &self,
        query: String,
        documents: Vec<String>,
        top_n: usize,
    ) -> BoxFuture<'static, anyhow::Result<Vec<RerankResult>>>;
}

pub(crate) fn validate_semantic_search_configuration(
    settings: &agent_settings::SemanticSearchSettings,
) -> Result<(), String> {
    if !settings.enabled {
        return Err(
            "Semantic code search is disabled. Enable `agent.semantic_search.enabled`.".to_string(),
        );
    }

    let Some(embedding) = settings.embedding.as_ref() else {
        return Err("Semantic code search requires `agent.semantic_search.embedding`.".to_string());
    };

    if embedding.api_format != agent_settings::SemanticSearchApiFormat::OpenAiEmbeddings {
        return Err(
            "Semantic code search embedding must use `openai_embeddings` API format.".to_string(),
        );
    }

    let Some(reranker) = settings.reranker.as_ref() else {
        return Err("Semantic code search requires `agent.semantic_search.reranker`.".to_string());
    };

    if reranker.api_format != agent_settings::SemanticSearchApiFormat::JinaRerank {
        return Err("Semantic code search reranker must use `jina_rerank` API format.".to_string());
    }

    Ok(())
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct OpenAiEmbeddingsRequest {
    pub(crate) model: String,
    pub(crate) input: Vec<String>,
}

#[derive(Clone, Debug, Deserialize)]
pub(crate) struct OpenAiEmbeddingsResponse {
    data: Vec<OpenAiEmbeddingData>,
}

#[derive(Clone, Debug, Deserialize)]
pub(crate) struct OpenAiEmbeddingData {
    embedding: Vec<f32>,
}

impl OpenAiEmbeddingsResponse {
    pub(crate) fn into_embeddings(self) -> Vec<Vec<f32>> {
        self.data.into_iter().map(|data| data.embedding).collect()
    }
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct JinaRerankRequest {
    pub(crate) model: String,
    pub(crate) query: String,
    pub(crate) documents: Vec<String>,
    pub(crate) top_n: usize,
}

#[derive(Clone, Debug, Deserialize)]
pub(crate) struct JinaRerankResponse {
    results: Vec<JinaRerankResponseItem>,
}

#[derive(Clone, Debug, Deserialize)]
pub(crate) struct JinaRerankResponseItem {
    index: usize,
    relevance_score: f32,
}

impl JinaRerankResponse {
    pub(crate) fn into_results(self) -> Vec<RerankResult> {
        self.results
            .into_iter()
            .map(|item| RerankResult {
                document_index: item.index,
                score: item.relevance_score,
            })
            .collect()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SemanticSearchProviderRole {
    Embedding,
    Reranker,
}

#[derive(Clone, Eq, PartialEq)]
pub(crate) struct SemanticSearchHttpProviderConfig {
    pub(crate) api_url: String,
    pub(crate) api_key: Option<String>,
    pub(crate) model: String,
}

impl fmt::Debug for SemanticSearchHttpProviderConfig {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SemanticSearchHttpProviderConfig")
            .field("api_url", &self.api_url)
            .field("api_key", &self.api_key.as_ref().map(|_| "<redacted>"))
            .field("model", &self.model)
            .finish()
    }
}

pub(crate) fn openai_compatible_api_key_env_var(provider_id: &str) -> String {
    let mut name = String::new();
    let mut previous_was_separator = true;

    for character in provider_id.chars() {
        if character.is_ascii_alphanumeric() {
            if !previous_was_separator && character.is_ascii_uppercase() {
                name.push('_');
            }
            name.push(character.to_ascii_uppercase());
            previous_was_separator = false;
        } else if !previous_was_separator {
            name.push('_');
            previous_was_separator = true;
        }
    }

    if name.ends_with('_') {
        name.pop();
    }

    if name.is_empty() {
        "OPENAI_COMPATIBLE_API_KEY".to_string()
    } else {
        format!("{name}_API_KEY")
    }
}

pub(crate) fn allows_empty_semantic_api_key(api_url: &str) -> bool {
    let Ok(url) = url::Url::parse(api_url) else {
        return false;
    };

    if url.scheme() != "http" {
        return false;
    }

    match url.host_str() {
        Some("localhost") | Some("[::1]") => true,
        Some(host) => host.parse::<IpAddr>().is_ok_and(|ip_address| {
            matches!(
                ip_address,
                IpAddr::V4(ip_address) if ip_address == Ipv4Addr::LOCALHOST
            ) || matches!(
                ip_address,
                IpAddr::V6(ip_address) if ip_address == Ipv6Addr::LOCALHOST
            )
        }),
        None => false,
    }
}

#[cfg(test)]
fn resolve_semantic_provider_config_for_test(
    selection: &agent_settings::SemanticSearchModelSelection,
    role: SemanticSearchProviderRole,
    openai_api_url: Option<&str>,
    openai_compatible: Vec<(&str, &str)>,
    api_key: Option<&str>,
) -> Result<SemanticSearchHttpProviderConfig, String> {
    resolve_semantic_provider_config_from_values(
        selection,
        role,
        openai_api_url,
        &openai_compatible
            .into_iter()
            .map(|(provider, api_url)| (provider.to_string(), api_url.to_string()))
            .collect::<Vec<_>>(),
        api_key.map(ToString::to_string),
    )
}

fn resolve_semantic_provider_config_from_values(
    selection: &agent_settings::SemanticSearchModelSelection,
    role: SemanticSearchProviderRole,
    openai_api_url: Option<&str>,
    openai_compatible: &[(String, String)],
    api_key: Option<String>,
) -> Result<SemanticSearchHttpProviderConfig, String> {
    validate_semantic_search_model_role(selection, role)?;

    let provider = selection.provider.as_ref();
    let api_url = if provider == "openai" {
        if role != SemanticSearchProviderRole::Embedding {
            return Err(
                "OpenAI is only supported for semantic search embeddings in v1".to_string(),
            );
        }
        openai_api_url
            .unwrap_or(open_ai::OPEN_AI_API_URL)
            .to_string()
    } else if let Some((_, api_url)) = openai_compatible
        .iter()
        .find(|(provider_id, _)| provider_id == provider)
    {
        api_url.clone()
    } else {
        return Err(format!(
            "Unsupported semantic search provider `{provider}`. Configure it under `language_models.openai_compatible` for semantic search v1."
        ));
    };

    if api_key.as_deref().unwrap_or_default().is_empty() && !allows_empty_semantic_api_key(&api_url)
    {
        let api_key_env_var = if provider == "openai" {
            "OPENAI_API_KEY".to_string()
        } else {
            openai_compatible_api_key_env_var(provider)
        };
        return Err(format!(
            "Semantic search provider `{provider}` requires {api_key_env_var}."
        ));
    }

    Ok(SemanticSearchHttpProviderConfig {
        api_url,
        api_key,
        model: selection.model.clone(),
    })
}

fn validate_semantic_search_model_role(
    selection: &agent_settings::SemanticSearchModelSelection,
    role: SemanticSearchProviderRole,
) -> Result<(), String> {
    match (role, selection.api_format) {
        (
            SemanticSearchProviderRole::Embedding,
            agent_settings::SemanticSearchApiFormat::OpenAiEmbeddings,
        ) => Ok(()),
        (
            SemanticSearchProviderRole::Reranker,
            agent_settings::SemanticSearchApiFormat::JinaRerank,
        ) => Ok(()),
        (SemanticSearchProviderRole::Embedding, _) => {
            Err("Semantic search embedding requires api_format `openai_embeddings`.".to_string())
        }
        (SemanticSearchProviderRole::Reranker, _) => {
            Err("Semantic search reranker requires api_format `jina_rerank`.".to_string())
        }
    }
}

pub(crate) fn resolve_semantic_provider_config(
    selection: &agent_settings::SemanticSearchModelSelection,
    role: SemanticSearchProviderRole,
    cx: &App,
) -> Result<SemanticSearchHttpProviderConfig, String> {
    let all_settings = language_models::AllLanguageModelSettings::get_global(cx);
    let provider = selection.provider.as_ref();
    let api_key = if provider == "openai" {
        std::env::var("OPENAI_API_KEY").ok()
    } else {
        std::env::var(openai_compatible_api_key_env_var(provider)).ok()
    };
    let openai_api_url = if all_settings.openai.api_url.is_empty() {
        Some(open_ai::OPEN_AI_API_URL)
    } else {
        Some(all_settings.openai.api_url.as_str())
    };
    let openai_compatible = all_settings
        .openai_compatible
        .iter()
        .map(|(provider_id, settings)| (provider_id.to_string(), settings.api_url.clone()))
        .collect::<Vec<_>>();

    resolve_semantic_provider_config_from_values(
        selection,
        role,
        openai_api_url,
        &openai_compatible,
        api_key,
    )
}

#[derive(Clone)]
pub(crate) struct OpenAiCompatibleEmbeddingProvider {
    http_client: Arc<dyn HttpClient>,
    api_url: String,
    api_key: String,
    model: String,
}

impl OpenAiCompatibleEmbeddingProvider {
    pub(crate) fn new(
        http_client: Arc<dyn HttpClient>,
        api_url: String,
        api_key: String,
        model: String,
    ) -> Self {
        Self {
            http_client,
            api_url,
            api_key,
            model,
        }
    }

    pub(crate) fn embed_inputs(
        &self,
        input: Vec<String>,
    ) -> BoxFuture<'static, anyhow::Result<Vec<Vec<f32>>>> {
        let http_client = self.http_client.clone();
        let api_url = self.api_url.clone();
        let api_key = self.api_key.clone();
        let model = self.model.clone();

        async move {
            let request = OpenAiEmbeddingsRequest {
                model: model.clone(),
                input,
            };
            post_json(&http_client, &embeddings_endpoint(&api_url), &api_key, &request)
                .await
                .map(|response: OpenAiEmbeddingsResponse| response.into_embeddings())
                .map_err(|error| {
                    anyhow::anyhow!(
                        "failed to fetch OpenAI-compatible embeddings for model {model}: expected OpenAI embeddings API format: {error}"
                    )
                })
        }
        .boxed()
    }
}

impl CodeSearchEmbeddingProvider for OpenAiCompatibleEmbeddingProvider {
    fn embed(&self, text: String) -> BoxFuture<'static, anyhow::Result<Vec<f32>>> {
        let embeddings = self.embed_inputs(vec![text]);
        let model = self.model.clone();

        async move {
            embeddings.await?.into_iter().next().ok_or_else(|| {
                anyhow::anyhow!(
                    "OpenAI-compatible embeddings response for model {model} did not include an embedding"
                )
            })
        }
        .boxed()
    }

    fn embed_batch(&self, texts: Vec<String>) -> BoxFuture<'static, anyhow::Result<Vec<Vec<f32>>>> {
        self.embed_inputs(texts)
    }
}

#[derive(Clone)]
pub(crate) struct JinaRerankerProvider {
    http_client: Arc<dyn HttpClient>,
    api_url: String,
    api_key: String,
    model: String,
}

impl JinaRerankerProvider {
    pub(crate) fn new(
        http_client: Arc<dyn HttpClient>,
        api_url: String,
        api_key: String,
        model: String,
    ) -> Self {
        Self {
            http_client,
            api_url,
            api_key,
            model,
        }
    }

    pub(crate) fn rerank_top_n(
        &self,
        query: String,
        documents: Vec<String>,
        top_n: usize,
    ) -> BoxFuture<'static, anyhow::Result<Vec<RerankResult>>> {
        let http_client = self.http_client.clone();
        let api_url = self.api_url.clone();
        let api_key = self.api_key.clone();
        let model = self.model.clone();

        async move {
            let request = JinaRerankRequest {
                model: model.clone(),
                query,
                documents,
                top_n,
            };
            post_json(&http_client, &rerank_endpoint(&api_url), &api_key, &request)
                .await
                .map(|response: JinaRerankResponse| response.into_results())
                .map_err(|error| {
                    anyhow::anyhow!(
                        "failed to fetch Jina rerank results for model {model}: expected Jina rerank API format: {error}"
                    )
                })
        }
        .boxed()
    }
}

impl CodeSearchRerankerProvider for JinaRerankerProvider {
    fn rerank(
        &self,
        query: String,
        documents: Vec<String>,
    ) -> BoxFuture<'static, anyhow::Result<Vec<RerankResult>>> {
        let top_n = documents.len();
        self.rerank_top_n(query, documents, top_n)
    }

    fn rerank_top_n(
        &self,
        query: String,
        documents: Vec<String>,
        top_n: usize,
    ) -> BoxFuture<'static, anyhow::Result<Vec<RerankResult>>> {
        JinaRerankerProvider::rerank_top_n(self, query, documents, top_n)
    }
}

async fn post_json<Request, Response>(
    http_client: &Arc<dyn HttpClient>,
    uri: &str,
    api_key: &str,
    request: &Request,
) -> anyhow::Result<Response>
where
    Request: Serialize,
    Response: for<'de> Deserialize<'de>,
{
    let serialized_request = serde_json::to_string(request)
        .map_err(|error| anyhow::anyhow!("failed to serialize request body: {error}"))?;
    let http_request = HttpRequest::builder()
        .method(Method::POST)
        .uri(uri)
        .header("Authorization", format!("Bearer {}", api_key.trim()))
        .header("Content-Type", "application/json")
        .body(AsyncBody::from(serialized_request))
        .map_err(|error| anyhow::anyhow!("failed to build HTTP request: {error}"))?;

    let mut response = http_client
        .send(http_request)
        .await
        .map_err(|error| anyhow::anyhow!("HTTP request failed: {error}"))?;
    let status = response.status();
    let mut body = String::new();
    response
        .body_mut()
        .read_to_string(&mut body)
        .await
        .map_err(|error| anyhow::anyhow!("failed to read HTTP response: {error}"))?;

    if !status.is_success() {
        return Err(anyhow::anyhow!("HTTP API returned status {status}"));
    }

    serde_json::from_str(&body)
        .map_err(|error| anyhow::anyhow!("failed to deserialize response body: {error}"))
}

fn embeddings_endpoint(api_url: &str) -> String {
    format!("{}/embeddings", api_url.trim_end_matches('/'))
}

fn rerank_endpoint(api_url: &str) -> String {
    format!("{}/rerank", api_url.trim_end_matches('/'))
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct TopologyExpansionRuntimeSettings {
    pub(crate) include_parent: bool,
    pub(crate) include_siblings: bool,
    pub(crate) max_parent_bytes: usize,
    pub(crate) max_total_expanded_bytes: usize,
}

#[derive(Clone, Debug)]
pub(crate) struct SemanticSearchRuntimeSettings {
    pub(crate) candidate_limit: usize,
    pub(crate) rerank_limit: usize,
    pub(crate) max_results: usize,
    pub(crate) hyde_mode: agent_settings::HyDEMode,
    pub(crate) hyde_threshold: f32,
    pub(crate) topology_expansion: TopologyExpansionRuntimeSettings,
}

impl SemanticSearchRuntimeSettings {
    pub(crate) fn from_settings(
        settings: &agent_settings::SemanticSearchSettings,
        input_max_results: Option<usize>,
        expand_context: Option<bool>,
    ) -> Self {
        let max_results = input_max_results
            .unwrap_or(settings.max_results)
            .min(settings.max_results)
            .max(1);
        let expand_context = expand_context.unwrap_or(true);

        Self {
            candidate_limit: settings.candidate_limit.max(max_results),
            rerank_limit: settings.rerank_limit.max(max_results),
            max_results,
            hyde_mode: settings.hyde.mode,
            hyde_threshold: settings.hyde.threshold,
            topology_expansion: TopologyExpansionRuntimeSettings {
                include_parent: expand_context && settings.topology_expansion.include_parent,
                include_siblings: expand_context && settings.topology_expansion.include_siblings,
                max_parent_bytes: settings.topology_expansion.max_parent_bytes,
                max_total_expanded_bytes: settings.topology_expansion.max_total_expanded_bytes,
            },
        }
    }

    #[cfg(test)]
    fn for_test(max_results: usize) -> Self {
        Self {
            candidate_limit: 80,
            rerank_limit: 30,
            max_results,
            hyde_mode: agent_settings::HyDEMode::Off,
            hyde_threshold: 0.6,
            topology_expansion: TopologyExpansionRuntimeSettings {
                include_parent: true,
                include_siblings: false,
                max_parent_bytes: 12_000,
                max_total_expanded_bytes: 24_000,
            },
        }
    }
}

#[derive(Clone)]
pub(crate) struct SemanticSearchProviderSet {
    pub(crate) embedding: Arc<dyn CodeSearchEmbeddingProvider>,
    pub(crate) reranker: Arc<dyn CodeSearchRerankerProvider>,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct CodeSearchIndex {
    chunks: HashMap<CodeChunkId, IndexedCodeChunk>,
}

impl CodeSearchIndex {
    pub(crate) fn from_chunks(chunks: Vec<IndexedCodeChunk>) -> Self {
        Self {
            chunks: chunks.into_iter().map(|chunk| (chunk.id, chunk)).collect(),
        }
    }

    pub(crate) fn chunks(&self) -> impl Iterator<Item = &IndexedCodeChunk> {
        self.chunks.values()
    }

    #[cfg(test)]
    pub(crate) fn from_chunks_for_test(chunks: Vec<IndexedCodeChunk>) -> Self {
        Self::from_chunks(chunks)
    }

    pub(crate) fn chunk(&self, chunk_id: CodeChunkId) -> Option<&IndexedCodeChunk> {
        self.chunks.get(&chunk_id)
    }
}

pub(crate) async fn embed_code_search_index(
    mut index: CodeSearchIndex,
    embedding_provider: &dyn CodeSearchEmbeddingProvider,
) -> anyhow::Result<CodeSearchIndex> {
    let mut chunk_ids = index.chunks.keys().copied().collect::<Vec<_>>();
    chunk_ids.sort();
    let texts = chunk_ids
        .iter()
        .filter_map(|chunk_id| index.chunk(*chunk_id).map(|chunk| chunk.text.clone()))
        .collect::<Vec<_>>();
    let embeddings = embedding_provider.embed_batch(texts).await?;
    if embeddings.len() != chunk_ids.len() {
        return Err(anyhow!(
            "embedding provider returned {} embeddings for {} indexed chunks",
            embeddings.len(),
            chunk_ids.len()
        ));
    }

    for (chunk_id, embedding) in chunk_ids.into_iter().zip(embeddings) {
        if let Some(chunk) = index.chunks.get_mut(&chunk_id) {
            chunk.embedding = embedding;
        }
    }

    Ok(index)
}

pub(crate) fn tokenize_for_search(text: &str) -> Vec<String> {
    let mut terms = Vec::new();
    let mut term = String::new();

    for character in text.chars() {
        if character.is_alphanumeric() || character == '_' {
            term.extend(character.to_lowercase());
        } else if !term.is_empty() {
            terms.push(std::mem::take(&mut term));
        }
    }

    if !term.is_empty() {
        terms.push(term);
    }

    terms
}

pub(crate) fn rank_by_vector_similarity<'a>(
    query_embedding: &[f32],
    candidates: impl IntoIterator<Item = (CodeChunkId, &'a [f32])>,
) -> Vec<ScoredChunk> {
    let mut ranked = candidates
        .into_iter()
        .map(|(chunk_id, embedding)| ScoredChunk {
            chunk_id,
            score: cosine_similarity(query_embedding, embedding),
        })
        .collect::<Vec<_>>();

    ranked.sort_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| left.chunk_id.cmp(&right.chunk_id))
    });
    ranked
}

pub(crate) fn rank_by_lexical_overlap<'a>(
    query: &str,
    candidates: impl IntoIterator<Item = &'a IndexedCodeChunk>,
) -> Vec<ScoredChunk> {
    let query_terms = tokenize_for_search(query)
        .into_iter()
        .collect::<HashSet<_>>();
    let mut ranked = candidates
        .into_iter()
        .filter_map(|chunk| {
            let score = chunk
                .lexical_terms
                .iter()
                .filter(|term| query_terms.contains(*term))
                .count() as f32;
            (score > 0.0).then_some(ScoredChunk {
                chunk_id: chunk.id,
                score,
            })
        })
        .collect::<Vec<_>>();

    ranked.sort_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| left.chunk_id.cmp(&right.chunk_id))
    });
    ranked
}

pub(crate) fn reciprocal_rank_fusion(
    ranked_lists: &[Vec<ScoredChunk>],
    rank_constant: usize,
    limit: usize,
) -> Vec<ScoredChunk> {
    let rank_constant = rank_constant as f32;
    let mut fused_scores = HashMap::new();

    for ranked_list in ranked_lists {
        for (rank, scored_chunk) in ranked_list.iter().enumerate() {
            let score = 1.0 / (rank_constant + rank as f32 + 1.0);
            *fused_scores.entry(scored_chunk.chunk_id).or_insert(0.0) += score;
        }
    }

    let mut fused = fused_scores
        .into_iter()
        .map(|(chunk_id, score)| ScoredChunk { chunk_id, score })
        .collect::<Vec<_>>();
    fused.sort_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| left.chunk_id.cmp(&right.chunk_id))
    });
    fused.truncate(limit);
    fused
}

pub(crate) fn should_run_hyde_fallback(best_score: Option<f32>, threshold: f32) -> bool {
    best_score.is_none_or(|best_score| best_score < threshold)
}

pub(crate) fn build_hyde_prompt(query: &str) -> String {
    format!(
        "You are helping search a codebase. Rewrite the user's request as a concise hypothetical code description and list likely identifiers, method names, class names, and comments that may appear in relevant code. Do not answer the user's question. Return only search text.\n\nUser request:\n{query}"
    )
}

pub(crate) fn resolve_hyde_model(
    selection: &settings::LanguageModelSelection,
    cx: &mut App,
) -> anyhow::Result<Arc<dyn LanguageModel>> {
    let selected_model = SelectedModel {
        provider: LanguageModelProviderId::from(selection.provider.0.clone()),
        model: LanguageModelId::from(selection.model.clone()),
    };

    LanguageModelRegistry::global(cx)
        .update(cx, |registry, cx| {
            registry
                .select_model(&selected_model, cx)
                .map(|configured_model| configured_model.model)
        })
        .ok_or_else(|| {
            anyhow!(
                "Configured HyDE model `agent.semantic_search.hyde.model` is not available: provider `{}`, model `{}`",
                selection.provider.0,
                selection.model
            )
        })
}

pub(crate) async fn generate_hyde_search_text_from_settings(
    hyde_settings: &agent_settings::SemanticSearchHyDESettings,
    query: &str,
    cx: &mut AsyncApp,
) -> anyhow::Result<String> {
    let model = cx.update(|cx| {
        let selection = hyde_settings
            .model
            .as_ref()
            .ok_or_else(|| anyhow!("HyDE fallback requires `agent.semantic_search.hyde.model`."))?;
        resolve_hyde_model(selection, cx)
    })?;
    cx.update(|cx| generate_hyde_search_text(model, query.to_string(), cx))
        .await
}

pub(crate) fn generate_hyde_search_text(
    model: Arc<dyn LanguageModel>,
    query: String,
    cx: &mut App,
) -> Task<anyhow::Result<String>> {
    cx.spawn(async move |cx| {
        let request = LanguageModelRequest {
            intent: Some(CompletionIntent::UserPrompt),
            messages: vec![LanguageModelRequestMessage {
                role: Role::User,
                content: vec![MessageContent::Text(build_hyde_prompt(&query))],
                cache: false,
                reasoning_details: None,
            }],
            tools: Vec::new(),
            tool_choice: Some(LanguageModelToolChoice::None),
            temperature: Some(0.2),
            thinking_allowed: false,
            speed: Some(Speed::Fast),
            ..Default::default()
        };
        let mut stream = model.stream_completion_text(request, cx).await?.stream;
        let mut search_text = String::new();
        while let Some(chunk) = stream.next().await {
            search_text.push_str(&chunk?);
        }
        Ok(search_text.trim().to_string())
    })
}

pub(crate) fn build_indexed_chunks(
    path: &str,
    source: &str,
    chunks: Vec<language::CastChunk>,
) -> Vec<IndexedCodeChunk> {
    let mut indexed_chunks = chunks
        .into_iter()
        .enumerate()
        .map(|(index, chunk)| {
            let id = CodeChunkId(index + 1);
            let byte_range = chunk.byte_range;
            let line_range = byte_range_to_line_range(source, byte_range.clone());
            let text = source
                .get(byte_range.clone())
                .unwrap_or_default()
                .to_string();
            let primary_node_kind = chunk.primary_node_kind;
            let lexical_terms = tokenize_for_search(&text);

            IndexedCodeChunk {
                id,
                worktree_id: None,
                path: Arc::new(PathBuf::from(path)),
                byte_range,
                line_range,
                text,
                non_whitespace_size: chunk.non_whitespace_size,
                primary_node_kind,
                topology: CodeChunkTopology {
                    parent: None,
                    children: Vec::new(),
                    previous_sibling: index
                        .checked_sub(1)
                        .map(|previous_index| CodeChunkId(previous_index + 1)),
                    next_sibling: None,
                    enclosing_symbols: Vec::new(),
                },
                embedding: Vec::new(),
                lexical_terms,
            }
        })
        .collect::<Vec<_>>();

    let chunk_count = indexed_chunks.len();
    for (index, chunk) in indexed_chunks.iter_mut().enumerate() {
        if index + 1 < chunk_count {
            chunk.topology.next_sibling = Some(CodeChunkId(index + 2));
        }
    }

    let mut parent_child_pairs = indexed_chunks
        .iter()
        .filter_map(|child| {
            let parent = indexed_chunks
                .iter()
                .filter(|candidate| {
                    candidate.id != child.id
                        && candidate.byte_range.start <= child.byte_range.start
                        && candidate.byte_range.end >= child.byte_range.end
                })
                .min_by_key(|candidate| {
                    (
                        candidate
                            .byte_range
                            .end
                            .saturating_sub(candidate.byte_range.start),
                        candidate.id,
                    )
                })?;

            Some((parent.id, child.id))
        })
        .collect::<Vec<_>>();

    let containment_children = parent_child_pairs
        .iter()
        .map(|(_, child_id)| *child_id)
        .collect::<HashSet<_>>();
    let mut active_parent_scope = None;
    for chunk in &indexed_chunks {
        if is_parent_like_chunk(chunk) {
            active_parent_scope = ParentScope::new(source, chunk);
            continue;
        }

        if containment_children.contains(&chunk.id) {
            continue;
        }

        if let Some(parent_scope) = active_parent_scope.as_mut()
            && parent_scope.contains(chunk)
        {
            parent_child_pairs.push((parent_scope.parent_id, chunk.id));
        }
    }

    for (parent_id, child_id) in parent_child_pairs {
        let parent_symbol = indexed_chunks
            .iter()
            .find(|chunk| chunk.id == parent_id)
            .map(enclosing_symbol_for_parent);
        if let Some(child) = indexed_chunks.iter_mut().find(|chunk| chunk.id == child_id) {
            child.topology.parent = Some(parent_id);
            if let Some(parent_symbol) = parent_symbol
                && !child.topology.enclosing_symbols.contains(&parent_symbol)
            {
                child.topology.enclosing_symbols.push(parent_symbol);
            }
        }
        if let Some(parent) = indexed_chunks
            .iter_mut()
            .find(|chunk| chunk.id == parent_id)
        {
            parent.topology.children.push(child_id);
        }
    }

    indexed_chunks
}

pub(crate) fn collect_semantic_search_project_files(
    project: Entity<project::Project>,
    settings: &SemanticSearchIndexingSettings,
    cx: &mut App,
) -> anyhow::Result<Vec<(project::ProjectPath, String, u64)>> {
    use project::WorktreeSettings;
    use util::paths::PathMatcher;

    let project = project.read(cx);
    let path_style = project.path_style(cx);
    let include_matcher = settings
        .include_pattern
        .as_ref()
        .map(|include_pattern| PathMatcher::new([include_pattern], path_style))
        .transpose()
        .map_err(|error| anyhow!("invalid include_pattern: {error}"))?;
    let global_settings = WorktreeSettings::get_global(cx);
    let mut files = Vec::new();

    for worktree in project.worktrees(cx) {
        let snapshot = worktree.read(cx).snapshot();
        for entry in snapshot.entries(false, 0) {
            if !entry.is_file() || entry.size > settings.max_indexed_file_bytes as u64 {
                continue;
            }

            let full_project_path = snapshot.root_name().join(&entry.path);
            if include_matcher
                .as_ref()
                .is_some_and(|matcher| !matcher.is_match(&full_project_path))
            {
                continue;
            }

            let project_path: project::ProjectPath = (snapshot.id(), entry.path.clone()).into();
            let worktree_settings = WorktreeSettings::get(Some((&project_path).into()), cx);
            if global_settings.is_path_excluded(&project_path.path)
                || global_settings.is_path_private(&project_path.path)
                || worktree_settings.is_path_excluded(&project_path.path)
                || worktree_settings.is_path_private(&project_path.path)
            {
                continue;
            }

            files.push((
                project_path,
                full_project_path.display(path_style).into_owned(),
                entry.size,
            ));
        }
    }

    Ok(files)
}

pub(crate) async fn build_project_semantic_index(
    project: Entity<project::Project>,
    settings: SemanticSearchIndexingSettings,
    cx: &mut AsyncApp,
) -> anyhow::Result<CodeSearchIndex> {
    let files =
        cx.update(|cx| collect_semantic_search_project_files(project.clone(), &settings, cx))?;
    let mut indexed_chunks = Vec::new();
    let mut next_chunk_id = 1usize;

    for (project_path, display_path, _size) in files {
        let buffer = project
            .update(cx, |project, cx| {
                project.open_buffer(project_path.clone(), cx)
            })
            .await?;
        let mut parse_status = buffer.read_with(cx, |buffer, _cx| buffer.parse_status());
        while *parse_status.borrow() != language::ParseStatus::Idle {
            parse_status.changed().await?;
        }

        let snapshot = buffer.read_with(cx, |buffer, _cx| buffer.snapshot());
        let source = snapshot
            .text_for_range(0..snapshot.len())
            .collect::<String>();
        let chunks = language::cast_chunks_for_buffer(
            &snapshot,
            language::CastChunkingOptions::enabled(settings.chunk_max_non_whitespace_size),
        );
        let mut file_chunks = if let Some(chunks) = chunks {
            build_indexed_chunks(&display_path, &source, chunks)
        } else {
            build_fallback_indexed_text_chunks(
                &display_path,
                &source,
                settings.chunk_max_non_whitespace_size,
            )
        };

        assign_project_chunk_ids(
            &mut file_chunks,
            &mut next_chunk_id,
            Some(project_path.worktree_id.to_proto()),
        );
        indexed_chunks.extend(file_chunks);
    }

    Ok(CodeSearchIndex::from_chunks(indexed_chunks))
}

fn assign_project_chunk_ids(
    file_chunks: &mut [IndexedCodeChunk],
    next_chunk_id: &mut usize,
    worktree_id: Option<u64>,
) {
    let id_map = file_chunks
        .iter()
        .map(|chunk| {
            let old_id = chunk.id;
            let new_id = CodeChunkId(*next_chunk_id);
            *next_chunk_id += 1;
            (old_id, new_id)
        })
        .collect::<HashMap<_, _>>();

    for chunk in file_chunks {
        let old_id = chunk.id;
        if let Some(new_id) = id_map.get(&old_id).copied() {
            chunk.id = new_id;
        } else {
            log::debug!("semantic search chunk id {old_id:?} was missing from remap table");
        }
        chunk.worktree_id = worktree_id;

        chunk.topology.parent = remap_optional_chunk_id(chunk.topology.parent, &id_map, "parent");
        chunk.topology.previous_sibling =
            remap_optional_chunk_id(chunk.topology.previous_sibling, &id_map, "previous_sibling");
        chunk.topology.next_sibling =
            remap_optional_chunk_id(chunk.topology.next_sibling, &id_map, "next_sibling");
        chunk.topology.children = remap_chunk_id_list(
            std::mem::take(&mut chunk.topology.children),
            &id_map,
            "children",
        );
    }
}

fn remap_optional_chunk_id(
    chunk_id: Option<CodeChunkId>,
    id_map: &HashMap<CodeChunkId, CodeChunkId>,
    field_name: &str,
) -> Option<CodeChunkId> {
    chunk_id.and_then(|chunk_id| {
        id_map.get(&chunk_id).copied().or_else(|| {
            log::debug!(
                "semantic search topology {field_name} reference {chunk_id:?} was missing from remap table"
            );
            None
        })
    })
}

fn remap_chunk_id_list(
    chunk_ids: Vec<CodeChunkId>,
    id_map: &HashMap<CodeChunkId, CodeChunkId>,
    field_name: &str,
) -> Vec<CodeChunkId> {
    chunk_ids
        .into_iter()
        .filter_map(|chunk_id| {
            id_map.get(&chunk_id).copied().or_else(|| {
                log::debug!(
                    "semantic search topology {field_name} reference {chunk_id:?} was missing from remap table"
                );
                None
            })
        })
        .collect()
}

#[cfg(test)]
async fn build_project_semantic_index_for_test(
    project: Entity<project::Project>,
    settings: SemanticSearchIndexingSettings,
    cx: &mut gpui::TestAppContext,
) -> anyhow::Result<CodeSearchIndex> {
    cx.update(|cx| {
        cx.spawn(async move |cx| build_project_semantic_index(project, settings, cx).await)
    })
    .await
}

pub(crate) fn build_fallback_indexed_text_chunks(
    path: &str,
    source: &str,
    max_non_whitespace_size: usize,
) -> Vec<IndexedCodeChunk> {
    let max_non_whitespace_size = max_non_whitespace_size.max(1);
    let mut ranges = Vec::new();
    let mut chunk_start = 0;
    let mut chunk_size = 0;
    let mut last_line_boundary = None;

    for (byte_index, character) in source.char_indices() {
        let next_byte_index = byte_index + character.len_utf8();
        let character_size = usize::from(!character.is_whitespace());
        if character == '\n' {
            last_line_boundary = Some(next_byte_index);
        }

        if chunk_size > 0 && chunk_size + character_size > max_non_whitespace_size {
            let chunk_end = last_line_boundary
                .filter(|line_boundary| {
                    *line_boundary > chunk_start && *line_boundary <= byte_index
                })
                .unwrap_or(byte_index);
            if chunk_end > chunk_start {
                ranges.push(chunk_start..chunk_end);
                chunk_start = chunk_end;
            }
            chunk_size = source
                .get(chunk_start..next_byte_index)
                .map(|text| {
                    text.chars()
                        .filter(|character| !character.is_whitespace())
                        .count()
                })
                .unwrap_or(character_size);
            last_line_boundary = None;
        } else {
            chunk_size += character_size;
        }
    }

    if chunk_start < source.len() {
        ranges.push(chunk_start..source.len());
    }

    let chunks = ranges
        .into_iter()
        .map(|byte_range| language::CastChunk {
            non_whitespace_size: source
                .get(byte_range.clone())
                .map(|text| {
                    text.chars()
                        .filter(|character| !character.is_whitespace())
                        .count()
                })
                .unwrap_or(0),
            byte_range,
            primary_node_kind: None,
            merged_node_count: 0,
        })
        .collect();

    build_indexed_chunks(path, source, chunks)
}

#[cfg(test)]
fn build_indexed_chunks_for_test(
    path: &str,
    source: &str,
    chunks: Vec<language::CastChunk>,
) -> Vec<IndexedCodeChunk> {
    build_indexed_chunks(path, source, chunks)
}

#[cfg(test)]
fn fallback_indexed_text_chunks_for_test(
    path: &str,
    source: &str,
    max_non_whitespace_size: usize,
) -> Vec<IndexedCodeChunk> {
    build_fallback_indexed_text_chunks(path, source, max_non_whitespace_size)
}

struct ParentScope {
    parent_id: CodeChunkId,
    byte_range: std::ops::Range<usize>,
}

impl ParentScope {
    fn new(source: &str, parent: &IndexedCodeChunk) -> Option<Self> {
        let byte_range = matching_brace_scope(source, parent.byte_range.clone())?;
        Some(Self {
            parent_id: parent.id,
            byte_range,
        })
    }

    fn contains(&self, chunk: &IndexedCodeChunk) -> bool {
        self.byte_range.start <= chunk.byte_range.start
            && self.byte_range.end >= chunk.byte_range.end
    }
}

fn matching_brace_scope(
    source: &str,
    parent_range: std::ops::Range<usize>,
) -> Option<std::ops::Range<usize>> {
    let source_bytes = source.as_bytes();
    let search_start = parent_range.start.min(source_bytes.len());
    let open_brace = source_bytes
        .get(search_start..)?
        .iter()
        .position(|byte| *byte == b'{')?
        + search_start;
    if open_brace > parent_range.end
        && !source_bytes
            .get(parent_range.end..open_brace)?
            .iter()
            .all(|byte| byte.is_ascii_whitespace())
    {
        return None;
    }
    let mut brace_depth = 0usize;

    for (byte_index, byte) in source_bytes.iter().enumerate().skip(open_brace) {
        match byte {
            b'{' => brace_depth += 1,
            b'}' => {
                brace_depth = brace_depth.saturating_sub(1);
                if brace_depth == 0 {
                    let scope_end = source_bytes
                        .iter()
                        .enumerate()
                        .skip(byte_index + 1)
                        .find_map(|(byte_index, byte)| {
                            (!byte.is_ascii_whitespace()).then_some(byte_index)
                        })
                        .unwrap_or(source_bytes.len());
                    return Some(search_start..scope_end);
                }
            }
            _ => {}
        }
    }

    None
}

fn enclosing_symbol_for_parent(parent: &IndexedCodeChunk) -> EnclosingSymbol {
    EnclosingSymbol {
        name: parent_symbol_name(parent),
        kind: parent.primary_node_kind.clone(),
        byte_range: parent.byte_range.clone(),
        line_range: parent.line_range.clone(),
    }
}

fn parent_symbol_name(parent: &IndexedCodeChunk) -> String {
    parent
        .text
        .lines()
        .next()
        .map(|line| line.trim().trim_end_matches('{').trim())
        .filter(|line| !line.is_empty())
        .unwrap_or_else(|| parent.primary_node_kind.as_deref().unwrap_or("parent"))
        .to_string()
}

fn is_parent_like_chunk(chunk: &IndexedCodeChunk) -> bool {
    chunk
        .primary_node_kind
        .as_deref()
        .is_some_and(is_parent_like_node_kind)
        || is_parent_like_source_text(&chunk.text)
}

fn is_parent_like_node_kind(kind: &str) -> bool {
    matches!(
        kind,
        "struct_item"
            | "impl_item"
            | "mod_item"
            | "enum_item"
            | "trait_item"
            | "union_item"
            | "class_declaration"
            | "interface_declaration"
            | "object"
    )
}

fn is_parent_like_source_text(text: &str) -> bool {
    let text = text.trim_start();
    [
        "impl ",
        "mod ",
        "class ",
        "interface ",
        "object ",
        "namespace ",
    ]
    .iter()
    .any(|prefix| text.starts_with(prefix))
}

fn byte_range_to_line_range(
    source: &str,
    byte_range: std::ops::Range<usize>,
) -> std::ops::Range<u32> {
    let source_bytes = source.as_bytes();
    let start_byte = byte_range.start.min(source_bytes.len());
    let end_byte = byte_range.end.min(source_bytes.len()).max(start_byte);
    let start_line = source_bytes[..start_byte]
        .iter()
        .filter(|byte| **byte == b'\n')
        .count();
    let end_line = source_bytes[..end_byte]
        .iter()
        .filter(|byte| **byte == b'\n')
        .count()
        + 1;

    usize_to_u32_saturating(start_line)..usize_to_u32_saturating(end_line)
}

fn usize_to_u32_saturating(value: usize) -> u32 {
    u32::try_from(value).unwrap_or(u32::MAX)
}

pub(crate) async fn run_semantic_search_query(
    index: &CodeSearchIndex,
    providers: &SemanticSearchProviderSet,
    query: &str,
    settings: &SemanticSearchRuntimeSettings,
    hyde_search_text: Option<String>,
) -> anyhow::Result<SemanticSearchResult> {
    let first_pass =
        run_semantic_search_pass(index, providers, query, settings, "reranked").await?;

    if let Some(hyde_query) = hyde_search_text {
        run_semantic_search_hyde_fallback(index, providers, settings, first_pass, hyde_query).await
    } else {
        Ok(first_pass)
    }
}

pub(crate) async fn run_semantic_search_hyde_fallback(
    index: &CodeSearchIndex,
    providers: &SemanticSearchProviderSet,
    settings: &SemanticSearchRuntimeSettings,
    first_pass: SemanticSearchResult,
    hyde_search_text: String,
) -> anyhow::Result<SemanticSearchResult> {
    let top_score = first_pass
        .matches
        .first()
        .map(|search_match| search_match.rerank_score);

    if settings.hyde_mode == agent_settings::HyDEMode::Fallback
        && should_run_hyde_fallback(top_score, settings.hyde_threshold)
    {
        let hyde_pass = run_semantic_search_pass(
            index,
            providers,
            &hyde_search_text,
            settings,
            "hyde_fallback",
        )
        .await
        .map_err(|error| anyhow::anyhow!("HyDE fallback semantic search failed: {error}"))
        .log_err();

        if let Some(hyde_pass) = hyde_pass {
            if hyde_pass
                .matches
                .first()
                .map(|search_match| search_match.rerank_score)
                .unwrap_or(0.0)
                > top_score.unwrap_or(0.0)
            {
                return Ok(SemanticSearchResult {
                    matches: hyde_pass.matches,
                    used_hyde: true,
                });
            }
        }
    }

    Ok(first_pass)
}

async fn run_semantic_search_pass(
    index: &CodeSearchIndex,
    providers: &SemanticSearchProviderSet,
    query: &str,
    settings: &SemanticSearchRuntimeSettings,
    source: &'static str,
) -> anyhow::Result<SemanticSearchResult> {
    let query_embedding = providers.embedding.embed(query.to_string()).await?;
    let dense = rank_by_vector_similarity(
        &query_embedding,
        index
            .chunks
            .values()
            .map(|chunk| (chunk.id, chunk.embedding.as_slice())),
    );
    let lexical = rank_by_lexical_overlap(query, index.chunks.values());
    let fused = reciprocal_rank_fusion(&[dense, lexical], 60, settings.candidate_limit);
    let rerank_candidates = fused
        .iter()
        .take(settings.rerank_limit)
        .filter_map(|candidate| {
            index
                .chunk(candidate.chunk_id)
                .map(|chunk| (*candidate, chunk.text.clone()))
        })
        .collect::<Vec<_>>();
    if rerank_candidates.is_empty() {
        return Ok(SemanticSearchResult {
            matches: Vec::new(),
            used_hyde: source == "hyde_fallback",
        });
    }

    let documents = rerank_candidates
        .iter()
        .map(|(_, document)| document.clone())
        .collect::<Vec<_>>();
    let reranked = providers
        .reranker
        .rerank_top_n(query.to_string(), documents, settings.max_results)
        .await?;

    let matches = reranked
        .into_iter()
        .map(|result| {
            let Some((candidate, _)) = rerank_candidates.get(result.document_index) else {
                return Err(anyhow::anyhow!(
                    "reranker returned document index {} but only {} rerank candidates were provided",
                    result.document_index,
                    rerank_candidates.len()
                ));
            };
            Ok(SemanticSearchMatch {
                chunk_id: candidate.chunk_id,
                rerank_score: result.score,
                source,
            })
        })
        .take(settings.max_results)
        .collect::<anyhow::Result<Vec<_>>>()?;

    Ok(SemanticSearchResult {
        matches,
        used_hyde: source == "hyde_fallback",
    })
}

#[cfg(test)]
async fn run_semantic_query_for_test(
    index: &CodeSearchIndex,
    embeddings: &(impl CodeSearchEmbeddingProvider + Clone + 'static),
    reranker: &(impl CodeSearchRerankerProvider + Clone + 'static),
    query: &str,
    limit: usize,
) -> SemanticSearchResult {
    let providers = SemanticSearchProviderSet {
        embedding: Arc::new(embeddings.clone()),
        reranker: Arc::new(reranker.clone()),
    };
    let settings = SemanticSearchRuntimeSettings::for_test(limit);
    run_semantic_search_query(index, &providers, query, &settings, None)
        .await
        .expect("fake providers should run semantic query")
}

#[cfg(test)]
async fn run_semantic_search_query_for_test(
    index: &CodeSearchIndex,
    providers: &FakeSemanticSearchProviders,
    query: &str,
    settings: &SemanticSearchRuntimeSettings,
    hyde_provider: Option<&FakeSemanticSearchProviders>,
) -> anyhow::Result<SemanticSearchResult> {
    run_semantic_search_query(
        index,
        &providers.providers,
        query,
        settings,
        hyde_provider.and_then(|provider| provider.hyde_text.clone()),
    )
    .await
}

pub(crate) fn expand_topology(
    index: &CodeSearchIndex,
    chunk_ids: &[CodeChunkId],
    settings: &TopologyExpansionRuntimeSettings,
) -> Vec<CodeChunkId> {
    let mut expanded = Vec::new();
    let mut seen = HashSet::new();
    let mut total_bytes = 0;

    for chunk_id in chunk_ids {
        let Some(chunk) = index.chunk(*chunk_id) else {
            continue;
        };

        if settings.include_parent
            && let Some(parent_id) = chunk.topology.parent
        {
            add_chunk_if_within_budget(
                index,
                parent_id,
                Some(settings.max_parent_bytes),
                settings.max_total_expanded_bytes,
                &mut total_bytes,
                &mut seen,
                &mut expanded,
            );
        }

        if settings.include_siblings {
            if let Some(previous_sibling) = chunk.topology.previous_sibling {
                add_chunk_if_within_budget(
                    index,
                    previous_sibling,
                    None,
                    settings.max_total_expanded_bytes,
                    &mut total_bytes,
                    &mut seen,
                    &mut expanded,
                );
            }
        }

        add_chunk_if_within_budget(
            index,
            *chunk_id,
            None,
            settings.max_total_expanded_bytes,
            &mut total_bytes,
            &mut seen,
            &mut expanded,
        );

        if settings.include_siblings {
            if let Some(next_sibling) = chunk.topology.next_sibling {
                add_chunk_if_within_budget(
                    index,
                    next_sibling,
                    None,
                    settings.max_total_expanded_bytes,
                    &mut total_bytes,
                    &mut seen,
                    &mut expanded,
                );
            }
        }
    }

    expanded
}

fn cosine_similarity(left: &[f32], right: &[f32]) -> f32 {
    if left.len() != right.len() || left.is_empty() {
        return 0.0;
    }

    let dot_product = left
        .iter()
        .zip(right.iter())
        .map(|(left, right)| left * right)
        .sum::<f32>();
    let left_norm = left.iter().map(|value| value * value).sum::<f32>().sqrt();
    let right_norm = right.iter().map(|value| value * value).sum::<f32>().sqrt();

    if left_norm == 0.0 || right_norm == 0.0 {
        0.0
    } else {
        dot_product / (left_norm * right_norm)
    }
}

#[cfg(test)]
#[derive(Clone, Debug)]
pub(crate) struct FakeEmbeddingProvider {
    embeddings: HashMap<String, Vec<f32>>,
}

#[cfg(test)]
impl FakeEmbeddingProvider {
    pub(crate) fn new(embeddings: Vec<(String, Vec<f32>)>) -> Self {
        Self {
            embeddings: embeddings.into_iter().collect(),
        }
    }
}

#[cfg(test)]
impl CodeSearchEmbeddingProvider for FakeEmbeddingProvider {
    fn embed(&self, text: String) -> BoxFuture<'static, anyhow::Result<Vec<f32>>> {
        let embedding = self.embeddings.get(&text).cloned();
        async move { embedding.ok_or_else(|| anyhow::anyhow!("missing fake embedding for {text}")) }
            .boxed()
    }

    fn embed_batch(&self, texts: Vec<String>) -> BoxFuture<'static, anyhow::Result<Vec<Vec<f32>>>> {
        let embeddings = texts
            .into_iter()
            .map(|text| {
                self.embeddings
                    .get(&text)
                    .cloned()
                    .ok_or_else(|| anyhow::anyhow!("missing fake embedding for {text}"))
            })
            .collect::<anyhow::Result<Vec<_>>>();
        async move { embeddings }.boxed()
    }
}

#[cfg(test)]
#[derive(Clone, Debug)]
pub(crate) struct FakeRerankerProvider {
    results: Arc<parking_lot::Mutex<VecDeque<Vec<RerankResult>>>>,
}

#[cfg(test)]
impl FakeRerankerProvider {
    pub(crate) fn new(results: Vec<(usize, f32)>) -> Self {
        Self::new_sequence(vec![results])
    }

    pub(crate) fn new_sequence(results: Vec<Vec<(usize, f32)>>) -> Self {
        Self {
            results: Arc::new(parking_lot::Mutex::new(
                results
                    .into_iter()
                    .map(|results| {
                        results
                            .into_iter()
                            .map(|(document_index, score)| RerankResult {
                                document_index,
                                score,
                            })
                            .collect()
                    })
                    .collect(),
            )),
        }
    }
}

#[cfg(test)]
impl CodeSearchRerankerProvider for FakeRerankerProvider {
    fn rerank(
        &self,
        query: String,
        documents: Vec<String>,
    ) -> BoxFuture<'static, anyhow::Result<Vec<RerankResult>>> {
        self.rerank_top_n(query, documents, usize::MAX)
    }

    fn rerank_top_n(
        &self,
        _query: String,
        _documents: Vec<String>,
        top_n: usize,
    ) -> BoxFuture<'static, anyhow::Result<Vec<RerankResult>>> {
        let results = self.results.lock().pop_front();
        async move {
            let mut results =
                results.ok_or_else(|| anyhow::anyhow!("missing fake rerank results"))?;
            results.truncate(top_n);
            Ok(results)
        }
        .boxed()
    }
}

#[cfg(test)]
struct FakeSemanticSearchProviders {
    providers: SemanticSearchProviderSet,
    hyde_text: Option<String>,
}

#[cfg(test)]
impl FakeSemanticSearchProviders {
    fn new(embeddings: Vec<(String, Vec<f32>)>, rerank_results: Vec<(usize, f32)>) -> Self {
        Self::from_parts(embeddings, vec![rerank_results], None)
    }

    fn new_with_hyde(
        embeddings: Vec<(String, Vec<f32>)>,
        rerank_results: Vec<Vec<(usize, f32)>>,
        hyde_text: String,
    ) -> Self {
        Self::from_parts(embeddings, rerank_results, Some(hyde_text))
    }

    fn from_parts(
        embeddings: Vec<(String, Vec<f32>)>,
        rerank_results: Vec<Vec<(usize, f32)>>,
        hyde_text: Option<String>,
    ) -> Self {
        Self {
            providers: SemanticSearchProviderSet {
                embedding: Arc::new(FakeEmbeddingProvider::new(embeddings)),
                reranker: Arc::new(FakeRerankerProvider::new_sequence(rerank_results)),
            },
            hyde_text,
        }
    }
}

fn add_chunk_if_within_budget(
    index: &CodeSearchIndex,
    chunk_id: CodeChunkId,
    max_chunk_bytes: Option<usize>,
    max_total_bytes: usize,
    total_bytes: &mut usize,
    seen: &mut HashSet<CodeChunkId>,
    expanded: &mut Vec<CodeChunkId>,
) {
    if seen.contains(&chunk_id) {
        return;
    }

    let Some(chunk) = index.chunk(chunk_id) else {
        return;
    };

    let chunk_bytes = chunk.text.len();
    if max_chunk_bytes.is_some_and(|max_chunk_bytes| chunk_bytes > max_chunk_bytes) {
        return;
    }

    let Some(next_total_bytes) = total_bytes.checked_add(chunk_bytes) else {
        return;
    };

    if next_total_bytes > max_total_bytes {
        return;
    }

    seen.insert(chunk_id);
    *total_bytes = next_total_bytes;
    expanded.push(chunk_id);
}

#[cfg(test)]
mod tests {
    use super::*;
    use http_client::{FakeHttpClient, Response, http::header::AUTHORIZATION};
    use language_model::LanguageModelProvider as _;
    use parking_lot::Mutex;
    use std::{ops::Range, path::PathBuf, sync::Arc};

    fn configured_model_selection(
        api_format: agent_settings::SemanticSearchApiFormat,
    ) -> agent_settings::SemanticSearchModelSelection {
        agent_settings::SemanticSearchModelSelection {
            provider: "test-provider".into(),
            model: "test-model".to_string(),
            api_format,
        }
    }

    #[test]
    fn hyde_prompt_asks_for_search_text_not_answer() {
        let prompt = build_hyde_prompt("Where is movement smoothing implemented?");
        assert!(prompt.contains("Do not answer the user's question"));
        assert!(prompt.contains("movement smoothing"));
        assert!(prompt.contains("likely identifiers"));
    }

    fn chunk(id: usize, text: &str, parent: Option<usize>) -> IndexedCodeChunk {
        let byte_range: Range<usize> = 0..text.len();

        IndexedCodeChunk {
            id: CodeChunkId(id),
            worktree_id: None,
            path: Arc::from(PathBuf::from("root/src/lib.rs")),
            byte_range,
            line_range: 0..1,
            text: text.to_string(),
            non_whitespace_size: text.chars().filter(|ch| !ch.is_whitespace()).count(),
            primary_node_kind: None,
            topology: CodeChunkTopology {
                parent: parent.map(CodeChunkId),
                children: Vec::new(),
                previous_sibling: None,
                next_sibling: None,
                enclosing_symbols: Vec::new(),
            },
            embedding: match id {
                1 => vec![0.9, 0.1],
                2 => vec![0.1, 0.9],
                _ => Vec::new(),
            },
            lexical_terms: tokenize_for_search(text),
        }
    }

    #[test]
    fn project_chunk_id_assignment_remaps_topology_references() {
        let mut parent = chunk(1, "impl Player {", None);
        parent.topology.children.push(CodeChunkId(2));
        parent.topology.next_sibling = Some(CodeChunkId(2));

        let mut child = chunk(2, "fn movement_speed() {}", Some(1));
        child.topology.previous_sibling = Some(CodeChunkId(1));

        let mut file_chunks = vec![parent, child];
        let mut next_chunk_id = 4;
        assign_project_chunk_ids(&mut file_chunks, &mut next_chunk_id, Some(10));

        assert_eq!(file_chunks[0].id, CodeChunkId(4));
        assert_eq!(file_chunks[0].worktree_id, Some(10));
        assert_eq!(file_chunks[0].topology.children, vec![CodeChunkId(5)]);
        assert_eq!(file_chunks[0].topology.next_sibling, Some(CodeChunkId(5)));

        assert_eq!(file_chunks[1].id, CodeChunkId(5));
        assert_eq!(file_chunks[1].worktree_id, Some(10));
        assert_eq!(file_chunks[1].topology.parent, Some(CodeChunkId(4)));
        assert_eq!(
            file_chunks[1].topology.previous_sibling,
            Some(CodeChunkId(4))
        );
        assert_eq!(next_chunk_id, 6);
    }

    #[gpui::test]
    async fn hyde_model_missing_error_names_provider_and_model(cx: &mut gpui::TestAppContext) {
        cx.update(|cx| {
            language_model::LanguageModelRegistry::test(cx);
        });
        let selection = settings::LanguageModelSelection {
            provider: "missing-provider".to_string().into(),
            model: "missing-model".to_string(),
            enable_thinking: false,
            effort: None,
            speed: None,
        };

        let error = cx
            .update(|cx| resolve_hyde_model(&selection, cx))
            .unwrap_err()
            .to_string();

        assert!(error.contains("agent.semantic_search.hyde.model"));
        assert!(error.contains("missing-provider"));
        assert!(error.contains("missing-model"));
    }

    #[gpui::test]
    async fn project_index_respects_include_pattern_and_file_size(cx: &mut gpui::TestAppContext) {
        use fs::FakeFs;
        use project::Project;
        use serde_json::json;
        use util::path;

        crate::tests::init_test(cx);
        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(
            path!("/root"),
            json!({
                "src": {
                    "auth.rs": "fn retry_login() { backoff(); }",
                    "large.rs": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
                    "ui.ts": "function renderButton() {}"
                },
                "large.rs": "fn top_level_large() {}"
            }),
        )
        .await;
        let project = Project::test(fs, [path!("/root").as_ref()], cx).await;
        let settings = SemanticSearchIndexingSettings {
            include_pattern: Some("root/src/*.rs".to_string()),
            max_indexed_file_bytes: 48,
            chunk_max_non_whitespace_size: 1600,
        };

        let indexed = build_project_semantic_index_for_test(project, settings, cx)
            .await
            .expect("index should build");

        assert!(
            indexed
                .chunks()
                .any(|chunk| chunk.path.ends_with(path!("src/auth.rs")))
        );
        assert!(
            !indexed
                .chunks()
                .any(|chunk| chunk.path.ends_with(path!("src/ui.ts")))
        );
        assert!(
            !indexed
                .chunks()
                .any(|chunk| chunk.path.ends_with(path!("src/large.rs")))
        );
        assert!(!indexed.chunks().any(|chunk| {
            chunk.path.ends_with(path!("large.rs")) && !chunk.path.ends_with(path!("src/large.rs"))
        }));
    }

    #[gpui::test]
    async fn hyde_generation_streams_search_text_without_tools(cx: &mut gpui::TestAppContext) {
        let (_fake_provider, model) = cx.update(|cx| {
            let fake_provider = language_model::LanguageModelRegistry::test(cx);
            let model = fake_provider.provided_models(cx)[0].clone();
            (fake_provider, model)
        });
        let task = cx.update(|cx| {
            generate_hyde_search_text(
                model.clone(),
                "Where is movement smoothing implemented?".to_string(),
                cx,
            )
        });
        cx.run_until_parked();

        let fake_model = model.as_fake();
        let request = fake_model.pending_completions().pop().unwrap();
        assert!(request.tools.is_empty());
        assert_eq!(
            request.tool_choice,
            Some(language_model::LanguageModelToolChoice::None)
        );
        assert!(!request.thinking_allowed);
        assert_eq!(request.temperature, Some(0.2));
        assert!(
            request.messages[0]
                .string_contents()
                .contains("Return only search text")
        );

        fake_model.send_last_completion_stream_text_chunk("movement_smoothing");
        fake_model.send_last_completion_stream_text_chunk(" interpolate_position");
        fake_model.end_last_completion_stream();

        assert_eq!(
            task.await.unwrap(),
            "movement_smoothing interpolate_position"
        );
    }

    #[test]
    fn embeddings_endpoint_uses_versioned_api_base() {
        assert_eq!(
            embeddings_endpoint("https://api.openai.com/v1"),
            "https://api.openai.com/v1/embeddings"
        );
        assert_eq!(
            embeddings_endpoint("https://api.openai.com/v1/"),
            "https://api.openai.com/v1/embeddings"
        );
    }

    #[test]
    fn rerank_endpoint_uses_versioned_api_base() {
        assert_eq!(
            rerank_endpoint("https://api.jina.ai/v1"),
            "https://api.jina.ai/v1/rerank"
        );
        assert_eq!(
            rerank_endpoint("https://api.jina.ai/v1/"),
            "https://api.jina.ai/v1/rerank"
        );
    }

    #[test]
    fn openai_compatible_env_var_name_uses_upper_snake_provider_id() {
        assert_eq!(
            openai_compatible_api_key_env_var("local-embeddings"),
            "LOCAL_EMBEDDINGS_API_KEY"
        );
        assert_eq!(
            openai_compatible_api_key_env_var("jina.rerank"),
            "JINA_RERANK_API_KEY"
        );
    }

    #[test]
    fn local_api_urls_allow_empty_api_key() {
        assert!(allows_empty_semantic_api_key("http://localhost:1234/v1"));
        assert!(allows_empty_semantic_api_key("http://127.0.0.1:8080/v1"));
        assert!(allows_empty_semantic_api_key("http://[::1]:8080/v1"));
        assert!(!allows_empty_semantic_api_key("https://api.openai.com/v1"));
    }

    #[test]
    fn local_api_urls_reject_loopback_lookalike_hosts() {
        assert!(!allows_empty_semantic_api_key(
            "http://localhost.evil:1234/v1"
        ));
        assert!(!allows_empty_semantic_api_key(
            "http://127.0.0.1.evil:8080/v1"
        ));
        assert!(!allows_empty_semantic_api_key("http://127.0.0.2:8080/v1"));
        assert!(!allows_empty_semantic_api_key("https://localhost:1234/v1"));
    }

    #[test]
    fn semantic_provider_config_debug_redacts_api_key() {
        let config = SemanticSearchHttpProviderConfig {
            api_url: "https://api.openai.com/v1".to_string(),
            api_key: Some("secret-key".to_string()),
            model: "text-embedding-3-small".to_string(),
        };

        let debug = format!("{config:?}");

        assert!(debug.contains("<redacted>"));
        assert!(!debug.contains("secret-key"));
    }

    #[test]
    fn semantic_provider_config_uses_openai_settings() {
        let selection = agent_settings::SemanticSearchModelSelection {
            provider: "openai".into(),
            model: "text-embedding-3-small".to_string(),
            api_format: agent_settings::SemanticSearchApiFormat::OpenAiEmbeddings,
        };

        let config = resolve_semantic_provider_config_for_test(
            &selection,
            SemanticSearchProviderRole::Embedding,
            Some("https://api.openai.com/v1"),
            Vec::new(),
            Some("test-key"),
        )
        .expect("openai embedding config should resolve");

        assert_eq!(config.api_url, "https://api.openai.com/v1");
        assert_eq!(config.api_key.as_deref(), Some("test-key"));
        assert_eq!(config.model, "text-embedding-3-small");
    }

    #[test]
    fn semantic_provider_config_rejects_unsupported_provider() {
        let selection = agent_settings::SemanticSearchModelSelection {
            provider: "ollama".into(),
            model: "nomic-embed-text".to_string(),
            api_format: agent_settings::SemanticSearchApiFormat::OpenAiEmbeddings,
        };

        let error = resolve_semantic_provider_config_for_test(
            &selection,
            SemanticSearchProviderRole::Embedding,
            None,
            Vec::new(),
            None,
        )
        .unwrap_err();

        assert!(error.contains("Unsupported semantic search provider"));
        assert!(error.contains("language_models.openai_compatible"));
    }

    #[gpui::test]
    async fn openai_embedding_adapter_posts_to_embeddings_endpoint_with_bearer_token(
        _cx: &mut gpui::TestAppContext,
    ) {
        let captured_request = Arc::new(Mutex::new(None));
        let http_client = FakeHttpClient::create({
            let captured_request = captured_request.clone();
            move |mut request| {
                let captured_request = captured_request.clone();
                async move {
                    let uri = request.uri().to_string();
                    let authorization = request
                        .headers()
                        .get(AUTHORIZATION)
                        .and_then(|header| header.to_str().ok())
                        .map(str::to_string);
                    let mut body = String::new();
                    request.body_mut().read_to_string(&mut body).await.unwrap();
                    *captured_request.lock() = Some((uri, authorization, body));
                    Ok(Response::builder()
                        .status(200)
                        .body(AsyncBody::from(r#"{"data":[{"embedding":[0.1,0.2]}]}"#))
                        .unwrap())
                }
            }
        });
        let provider = OpenAiCompatibleEmbeddingProvider::new(
            http_client,
            "https://example.test/".to_string(),
            "  secret-key  ".to_string(),
            "embedding-model".to_string(),
        );

        let embeddings = provider
            .embed_inputs(vec!["alpha".to_string()])
            .await
            .unwrap();

        assert_eq!(embeddings, vec![vec![0.1, 0.2]]);
        let (uri, authorization, body) = captured_request.lock().clone().unwrap();
        assert_eq!(uri, "https://example.test/embeddings");
        assert_eq!(authorization.as_deref(), Some("Bearer secret-key"));
        let body: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(body["model"], "embedding-model");
        assert_eq!(body["input"][0], "alpha");
    }

    #[gpui::test]
    async fn jina_rerank_adapter_posts_to_rerank_endpoint_with_bearer_token(
        _cx: &mut gpui::TestAppContext,
    ) {
        let captured_request = Arc::new(Mutex::new(None));
        let http_client = FakeHttpClient::create({
            let captured_request = captured_request.clone();
            move |mut request| {
                let captured_request = captured_request.clone();
                async move {
                    let uri = request.uri().to_string();
                    let authorization = request
                        .headers()
                        .get(AUTHORIZATION)
                        .and_then(|header| header.to_str().ok())
                        .map(str::to_string);
                    let mut body = String::new();
                    request.body_mut().read_to_string(&mut body).await.unwrap();
                    *captured_request.lock() = Some((uri, authorization, body));
                    Ok(Response::builder()
                        .status(200)
                        .body(AsyncBody::from(
                            r#"{"results":[{"index":1,"relevance_score":0.8}]}"#,
                        ))
                        .unwrap())
                }
            }
        });
        let provider = JinaRerankerProvider::new(
            http_client,
            "https://rerank.example/".to_string(),
            "  rerank-key  ".to_string(),
            "rerank-model".to_string(),
        );

        let results = provider
            .rerank_top_n(
                "needle".to_string(),
                vec!["first".to_string(), "second".to_string()],
                1,
            )
            .await
            .unwrap();

        assert_eq!(
            results,
            vec![RerankResult {
                document_index: 1,
                score: 0.8
            }]
        );
        let (uri, authorization, body) = captured_request.lock().clone().unwrap();
        assert_eq!(uri, "https://rerank.example/rerank");
        assert_eq!(authorization.as_deref(), Some("Bearer rerank-key"));
        let body: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(body["model"], "rerank-model");
        assert_eq!(body["query"], "needle");
        assert_eq!(body["documents"][0], "first");
        assert_eq!(body["top_n"], 1);
    }

    #[gpui::test]
    async fn adapter_non_success_error_omits_api_key(_cx: &mut gpui::TestAppContext) {
        let http_client = FakeHttpClient::create(|_request| async move {
            Ok(Response::builder()
                .status(500)
                .body(AsyncBody::from("server echoed secret-key"))
                .unwrap())
        });
        let provider = OpenAiCompatibleEmbeddingProvider::new(
            http_client,
            "https://example.test".to_string(),
            "secret-key".to_string(),
            "embedding-model".to_string(),
        );

        let error = provider
            .embed_inputs(vec!["alpha".to_string()])
            .await
            .unwrap_err()
            .to_string();

        assert!(error.contains("embedding-model"));
        assert!(error.contains("status 500"));
        assert!(!error.contains("secret-key"));
    }

    #[gpui::test]
    async fn adapter_malformed_json_error_mentions_deserialization(_cx: &mut gpui::TestAppContext) {
        let http_client = FakeHttpClient::create(|_request| async move {
            Ok(Response::builder()
                .status(200)
                .body(AsyncBody::from("not json"))
                .unwrap())
        });
        let provider = JinaRerankerProvider::new(
            http_client,
            "https://rerank.example".to_string(),
            "rerank-key".to_string(),
            "rerank-model".to_string(),
        );

        let error = provider
            .rerank_top_n("needle".to_string(), vec!["first".to_string()], 1)
            .await
            .unwrap_err()
            .to_string();

        assert!(error.contains("rerank-model"));
        assert!(error.contains("deserialize"));
    }

    #[test]
    fn validate_configuration_rejects_disabled_semantic_search() {
        let settings = agent_settings::SemanticSearchSettings::default();

        let error = validate_semantic_search_configuration(&settings).unwrap_err();

        assert_eq!(
            error,
            "Semantic code search is disabled. Enable `agent.semantic_search.enabled`."
        );
    }

    #[test]
    fn validate_configuration_rejects_missing_embedding() {
        let mut settings = agent_settings::SemanticSearchSettings {
            enabled: true,
            ..Default::default()
        };
        settings.reranker = Some(configured_model_selection(
            agent_settings::SemanticSearchApiFormat::JinaRerank,
        ));

        let error = validate_semantic_search_configuration(&settings).unwrap_err();

        assert_eq!(
            error,
            "Semantic code search requires `agent.semantic_search.embedding`."
        );
    }

    #[test]
    fn validate_configuration_rejects_missing_reranker() {
        let mut settings = agent_settings::SemanticSearchSettings {
            enabled: true,
            ..Default::default()
        };
        settings.embedding = Some(configured_model_selection(
            agent_settings::SemanticSearchApiFormat::OpenAiEmbeddings,
        ));

        let error = validate_semantic_search_configuration(&settings).unwrap_err();

        assert_eq!(
            error,
            "Semantic code search requires `agent.semantic_search.reranker`."
        );
    }

    #[test]
    fn validate_configuration_rejects_embedding_with_rerank_api_format() {
        let mut settings = agent_settings::SemanticSearchSettings {
            enabled: true,
            ..Default::default()
        };
        settings.embedding = Some(configured_model_selection(
            agent_settings::SemanticSearchApiFormat::JinaRerank,
        ));
        settings.reranker = Some(configured_model_selection(
            agent_settings::SemanticSearchApiFormat::JinaRerank,
        ));

        let error = validate_semantic_search_configuration(&settings).unwrap_err();

        assert_eq!(
            error,
            "Semantic code search embedding must use `openai_embeddings` API format."
        );
    }

    #[test]
    fn validate_configuration_rejects_reranker_with_embedding_api_format() {
        let mut settings = agent_settings::SemanticSearchSettings {
            enabled: true,
            ..Default::default()
        };
        settings.embedding = Some(configured_model_selection(
            agent_settings::SemanticSearchApiFormat::OpenAiEmbeddings,
        ));
        settings.reranker = Some(configured_model_selection(
            agent_settings::SemanticSearchApiFormat::OpenAiEmbeddings,
        ));

        let error = validate_semantic_search_configuration(&settings).unwrap_err();

        assert_eq!(
            error,
            "Semantic code search reranker must use `jina_rerank` API format."
        );
    }

    #[test]
    fn validate_configuration_accepts_enabled_embedding_and_reranker() {
        let mut settings = agent_settings::SemanticSearchSettings {
            enabled: true,
            ..Default::default()
        };
        settings.embedding = Some(configured_model_selection(
            agent_settings::SemanticSearchApiFormat::OpenAiEmbeddings,
        ));
        settings.reranker = Some(configured_model_selection(
            agent_settings::SemanticSearchApiFormat::JinaRerank,
        ));

        validate_semantic_search_configuration(&settings).unwrap();
    }

    #[test]
    fn openai_embeddings_request_serializes_inputs() {
        let request = OpenAiEmbeddingsRequest {
            model: "Qwen3-Embedding-4B".to_string(),
            input: vec!["alpha".to_string(), "beta".to_string()],
        };
        let value = serde_json::to_value(&request).unwrap();
        assert_eq!(value["model"], "Qwen3-Embedding-4B");
        assert_eq!(value["input"][0], "alpha");
    }

    #[test]
    fn jina_rerank_response_maps_document_indexes_and_scores() {
        let response: JinaRerankResponse = serde_json::from_value(serde_json::json!({
            "results": [
                { "index": 1, "relevance_score": 0.7 },
                { "index": 0, "relevance_score": 0.9 }
            ]
        }))
        .unwrap();
        let mapped = response.into_results();
        assert_eq!(
            mapped[0],
            RerankResult {
                document_index: 1,
                score: 0.7
            }
        );
        assert_eq!(
            mapped[1],
            RerankResult {
                document_index: 0,
                score: 0.9
            }
        );
    }

    #[test]
    fn cosine_similarity_orders_closest_vector_first() {
        let query = vec![1.0, 0.0];
        let farther = vec![0.2, 0.8];
        let closer = vec![0.9, 0.1];
        let candidates = vec![
            (CodeChunkId(1), farther.as_slice()),
            (CodeChunkId(2), closer.as_slice()),
        ];
        let ranked = rank_by_vector_similarity(&query, candidates);
        assert_eq!(ranked[0].chunk_id, CodeChunkId(2));
    }

    #[test]
    fn mismatched_embedding_dimensions_score_zero() {
        let query = vec![1.0, 0.0];
        let mismatched = vec![1.0];
        let valid = vec![0.9, 0.1];
        let candidates = vec![
            (CodeChunkId(1), mismatched.as_slice()),
            (CodeChunkId(2), valid.as_slice()),
        ];
        let ranked = rank_by_vector_similarity(&query, candidates);
        assert_eq!(ranked[0].chunk_id, CodeChunkId(2));
        assert_eq!(ranked[1].score, 0.0);
    }

    #[test]
    fn lexical_overlap_excludes_chunks_without_matching_terms() {
        let first = chunk(1, "authentication retry backoff", None);
        let second = chunk(2, "render button layout", None);
        let third = chunk(3, "authentication token", None);
        let chunks = [first, second, third];

        let ranked = rank_by_lexical_overlap("authentication", chunks.iter());

        assert_eq!(
            ranked,
            vec![
                ScoredChunk {
                    chunk_id: CodeChunkId(1),
                    score: 1.0,
                },
                ScoredChunk {
                    chunk_id: CodeChunkId(3),
                    score: 1.0,
                }
            ]
        );
    }

    #[test]
    fn reciprocal_rank_fusion_combines_dense_and_lexical_ranks() {
        let dense = vec![
            ScoredChunk {
                chunk_id: CodeChunkId(1),
                score: 0.9,
            },
            ScoredChunk {
                chunk_id: CodeChunkId(2),
                score: 0.8,
            },
        ];
        let lexical = vec![
            ScoredChunk {
                chunk_id: CodeChunkId(2),
                score: 10.0,
            },
            ScoredChunk {
                chunk_id: CodeChunkId(3),
                score: 1.0,
            },
        ];
        let fused = reciprocal_rank_fusion(&[dense, lexical], 60, 10);
        assert_eq!(fused[0].chunk_id, CodeChunkId(2));
        assert_eq!(fused.len(), 3);
    }

    #[gpui::test]
    async fn query_pipeline_reranks_fused_candidates(_cx: &mut gpui::TestAppContext) {
        let index = CodeSearchIndex::from_chunks_for_test(vec![
            chunk(1, "authentication retry backoff", None),
            chunk(2, "render button layout", None),
        ]);
        let embeddings = FakeEmbeddingProvider::new(vec![
            ("auth retry".to_string(), vec![1.0, 0.0]),
            ("authentication retry backoff".to_string(), vec![0.9, 0.1]),
            ("render button layout".to_string(), vec![0.1, 0.9]),
        ]);
        let reranker = FakeRerankerProvider::new(vec![(0, 0.95), (1, 0.2)]);
        let result =
            run_semantic_query_for_test(&index, &embeddings, &reranker, "auth retry", 2).await;
        assert_eq!(result.matches[0].chunk_id, CodeChunkId(1));
        assert_eq!(result.matches[0].rerank_score, 0.95);
    }

    #[gpui::test]
    async fn semantic_query_pipeline_returns_reranked_matches(_cx: &mut gpui::TestAppContext) {
        let index = CodeSearchIndex::from_chunks_for_test(vec![
            chunk(1, "authentication retry backoff", None),
            chunk(2, "render button layout", None),
        ]);
        let providers = FakeSemanticSearchProviders::new(
            vec![
                ("auth retry".to_string(), vec![1.0, 0.0]),
                ("authentication retry backoff".to_string(), vec![0.9, 0.1]),
                ("render button layout".to_string(), vec![0.1, 0.9]),
            ],
            vec![(0, 0.95), (1, 0.2)],
        );
        let settings = SemanticSearchRuntimeSettings::for_test(2);

        let result =
            run_semantic_search_query_for_test(&index, &providers, "auth retry", &settings, None)
                .await
                .expect("query should succeed");

        assert_eq!(result.matches[0].chunk_id, CodeChunkId(1));
        assert_eq!(result.matches[0].rerank_score, 0.95);
        assert_eq!(result.matches[0].source, "reranked");
        assert!(!result.used_hyde);
    }

    #[gpui::test]
    async fn semantic_query_pipeline_uses_hyde_when_first_pass_score_is_low(
        _cx: &mut gpui::TestAppContext,
    ) {
        let index = CodeSearchIndex::from_chunks_for_test(vec![
            chunk(1, "authentication retry backoff", None),
            chunk(2, "render button layout", None),
        ]);
        let providers = FakeSemanticSearchProviders::new_with_hyde(
            vec![
                ("unclear request".to_string(), vec![0.1, 0.9]),
                ("retry backoff implementation".to_string(), vec![1.0, 0.0]),
                ("authentication retry backoff".to_string(), vec![0.9, 0.1]),
                ("render button layout".to_string(), vec![0.1, 0.9]),
            ],
            vec![vec![(1, 0.2), (0, 0.1)], vec![(0, 0.91), (1, 0.1)]],
            "retry backoff implementation".to_string(),
        );
        let mut settings = SemanticSearchRuntimeSettings::for_test(2);
        settings.hyde_mode = agent_settings::HyDEMode::Fallback;
        settings.hyde_threshold = 0.6;

        let result = run_semantic_search_query_for_test(
            &index,
            &providers,
            "unclear request",
            &settings,
            Some(&providers),
        )
        .await
        .expect("query should succeed");

        assert_eq!(result.matches[0].chunk_id, CodeChunkId(1));
        assert_eq!(result.matches[0].source, "hyde_fallback");
        assert!(result.used_hyde);
    }

    #[gpui::test]
    async fn hyde_failure_keeps_first_pass_matches(_cx: &mut gpui::TestAppContext) {
        let index = CodeSearchIndex::from_chunks_for_test(vec![
            chunk(1, "authentication retry backoff", None),
            chunk(2, "render button layout", None),
        ]);
        let providers = FakeSemanticSearchProviders::new(
            vec![
                ("unclear request".to_string(), vec![1.0, 0.0]),
                ("authentication retry backoff".to_string(), vec![0.9, 0.1]),
                ("render button layout".to_string(), vec![0.1, 0.9]),
            ],
            vec![(0, 0.2), (1, 0.1)],
        );
        let mut settings = SemanticSearchRuntimeSettings::for_test(2);
        settings.hyde_mode = agent_settings::HyDEMode::Fallback;
        settings.hyde_threshold = 0.6;

        let result = run_semantic_search_query(
            &index,
            &providers.providers,
            "unclear request",
            &settings,
            None,
        )
        .await
        .expect("first pass should be returned when hyde text is unavailable");

        assert_eq!(result.matches[0].chunk_id, CodeChunkId(1));
        assert_eq!(result.matches[0].source, "reranked");
        assert!(!result.used_hyde);
    }

    #[gpui::test]
    async fn semantic_query_pipeline_keeps_first_pass_when_hyde_fallback_errors(
        _cx: &mut gpui::TestAppContext,
    ) {
        let index = CodeSearchIndex::from_chunks_for_test(vec![
            chunk(1, "authentication retry backoff", None),
            chunk(2, "render button layout", None),
        ]);
        let providers = FakeSemanticSearchProviders::new_with_hyde(
            vec![
                ("unclear request".to_string(), vec![0.1, 0.9]),
                ("authentication retry backoff".to_string(), vec![0.9, 0.1]),
                ("render button layout".to_string(), vec![0.1, 0.9]),
            ],
            vec![vec![(1, 0.2), (0, 0.1)]],
            "missing hyde embedding".to_string(),
        );
        let mut settings = SemanticSearchRuntimeSettings::for_test(2);
        settings.hyde_mode = agent_settings::HyDEMode::Fallback;
        settings.hyde_threshold = 0.6;

        let result = run_semantic_search_query_for_test(
            &index,
            &providers,
            "unclear request",
            &settings,
            Some(&providers),
        )
        .await
        .expect("HyDE fallback error should not fail the query");

        assert_eq!(result.matches[0].chunk_id, CodeChunkId(1));
        assert_eq!(result.matches[0].rerank_score, 0.2);
        assert_eq!(result.matches[0].source, "reranked");
        assert!(!result.used_hyde);
    }

    #[gpui::test]
    async fn semantic_query_pipeline_errors_on_invalid_rerank_document_index(
        _cx: &mut gpui::TestAppContext,
    ) {
        let index = CodeSearchIndex::from_chunks_for_test(vec![
            chunk(1, "authentication retry backoff", None),
            chunk(2, "render button layout", None),
        ]);
        let providers = FakeSemanticSearchProviders::new(
            vec![
                ("auth retry".to_string(), vec![1.0, 0.0]),
                ("authentication retry backoff".to_string(), vec![0.9, 0.1]),
                ("render button layout".to_string(), vec![0.1, 0.9]),
            ],
            vec![(2, 0.95)],
        );
        let settings = SemanticSearchRuntimeSettings::for_test(2);

        let error =
            run_semantic_search_query_for_test(&index, &providers, "auth retry", &settings, None)
                .await
                .unwrap_err()
                .to_string();

        assert!(error.contains("document index 2"));
        assert!(error.contains("2 rerank candidates"));
    }

    #[gpui::test]
    async fn semantic_query_pipeline_returns_empty_without_reranking_empty_candidates(
        _cx: &mut gpui::TestAppContext,
    ) {
        let index = CodeSearchIndex::from_chunks_for_test(Vec::new());
        let providers = FakeSemanticSearchProviders::from_parts(
            vec![("missing code".to_string(), vec![1.0, 0.0])],
            Vec::new(),
            None,
        );
        let settings = SemanticSearchRuntimeSettings::for_test(2);

        let result =
            run_semantic_search_query_for_test(&index, &providers, "missing code", &settings, None)
                .await
                .expect("empty candidate query should succeed without reranking");

        assert!(result.matches.is_empty());
        assert!(!result.used_hyde);
    }

    #[gpui::test]
    async fn hyde_fallback_runs_only_below_threshold(_cx: &mut gpui::TestAppContext) {
        let decision = should_run_hyde_fallback(Some(0.59), 0.6);
        assert!(decision);
        let decision = should_run_hyde_fallback(Some(0.6), 0.6);
        assert!(!decision);
        let decision = should_run_hyde_fallback(None, 0.6);
        assert!(decision);
    }

    #[test]
    fn topology_expansion_adds_parent_before_child() {
        let mut parent = chunk(1, "class Player {\n    void Move() {}\n}", None);
        parent.topology.children.push(CodeChunkId(2));
        let child = chunk(2, "void Move() {}", Some(1));
        let index = CodeSearchIndex::from_chunks_for_test(vec![parent, child]);
        let settings = TopologyExpansionRuntimeSettings {
            include_parent: true,
            include_siblings: false,
            max_parent_bytes: 1000,
            max_total_expanded_bytes: 2000,
        };
        let expanded = expand_topology(&index, &[CodeChunkId(2)], &settings);
        assert_eq!(expanded, vec![CodeChunkId(1), CodeChunkId(2)]);
    }

    #[test]
    fn parent_like_node_kind_recognizes_declaration_containers() {
        assert!(is_parent_like_node_kind("struct_item"));
        assert!(is_parent_like_node_kind("impl_item"));
        assert!(is_parent_like_node_kind("class_declaration"));
        assert!(!is_parent_like_node_kind("identifier"));
        assert!(!is_parent_like_node_kind("primitive_type"));
    }

    #[test]
    fn fallback_text_chunks_are_utf8_safe_and_cover_source() {
        let source = "alpha βeta\ngamma delta\nemoji 😀 done\n";
        let chunks = fallback_indexed_text_chunks_for_test("root/src/text.txt", source, 8);

        assert!(!chunks.is_empty());
        assert_eq!(
            chunks
                .iter()
                .map(|chunk| chunk.text.as_str())
                .collect::<String>(),
            source
        );
        assert!(
            chunks
                .iter()
                .all(|chunk| source.get(chunk.byte_range.clone()).is_some())
        );
    }

    #[test]
    fn fallback_text_chunks_respect_non_whitespace_budget_when_possible() {
        let source = "one two three\nfour five six\n";
        let chunks = fallback_indexed_text_chunks_for_test("root/src/text.txt", source, 8);

        assert!(chunks.len() > 1);
        assert!(
            chunks
                .iter()
                .all(|chunk| chunk.non_whitespace_size <= 8 || !chunk.text.contains('\n'))
        );
    }

    #[test]
    fn flat_cast_index_links_following_chunks_to_parent_like_source_chunk() {
        let source = "impl Player {\n    fn movement_speed(&self) -> f32 { self.speed }\n}\n";
        let indexed = build_indexed_chunks_for_test(
            "root/src/player.rs",
            source,
            vec![
                language::CastChunk {
                    byte_range: 0..12,
                    non_whitespace_size: 10,
                    primary_node_kind: Some("type_identifier".to_string()),
                    merged_node_count: 1,
                },
                language::CastChunk {
                    byte_range: 12..source.len(),
                    non_whitespace_size: 42,
                    primary_node_kind: Some("identifier".to_string()),
                    merged_node_count: 1,
                },
            ],
        );

        assert_eq!(indexed[1].topology.parent, Some(CodeChunkId(1)));
        assert_eq!(indexed[0].topology.children, vec![CodeChunkId(2)]);
    }

    #[test]
    fn cast_index_does_not_seed_self_symbols() {
        let source = "struct Player {\n    speed: f32,\n}\n";
        let indexed = build_indexed_chunks_for_test(
            "root/src/player.rs",
            source,
            vec![language::CastChunk {
                byte_range: 0..source.len(),
                non_whitespace_size: 24,
                primary_node_kind: Some("struct_item".to_string()),
                merged_node_count: 1,
            }],
        );

        assert!(indexed[0].topology.enclosing_symbols.is_empty());
        assert_eq!(indexed[0].primary_node_kind.as_deref(), Some("struct_item"));
    }

    #[test]
    fn flat_cast_index_adds_parent_symbol_to_child() {
        let source = "impl Player {\n    fn movement_speed(&self) -> f32 { self.speed }\n}\n";
        let indexed = build_indexed_chunks_for_test(
            "root/src/player.rs",
            source,
            vec![
                language::CastChunk {
                    byte_range: 0..12,
                    non_whitespace_size: 10,
                    primary_node_kind: Some("type_identifier".to_string()),
                    merged_node_count: 1,
                },
                language::CastChunk {
                    byte_range: 12..source.len(),
                    non_whitespace_size: 42,
                    primary_node_kind: Some("identifier".to_string()),
                    merged_node_count: 1,
                },
            ],
        );

        let child_symbols = &indexed[1].topology.enclosing_symbols;
        assert!(child_symbols.iter().any(|symbol| {
            symbol.name == "impl Player" && symbol.kind.as_deref() == Some("type_identifier")
        }));
    }

    #[test]
    fn flat_cast_index_requires_matched_scope_for_parent_fallback() {
        let source = "impl Player\nfn movement_speed(&self) -> f32 { self.speed }\n";
        let child_start = source.find("fn movement_speed").unwrap();
        let indexed = build_indexed_chunks_for_test(
            "root/src/player.rs",
            source,
            vec![
                language::CastChunk {
                    byte_range: 0..child_start,
                    non_whitespace_size: 10,
                    primary_node_kind: Some("type_identifier".to_string()),
                    merged_node_count: 1,
                },
                language::CastChunk {
                    byte_range: child_start..source.len(),
                    non_whitespace_size: 42,
                    primary_node_kind: Some("identifier".to_string()),
                    merged_node_count: 1,
                },
            ],
        );

        assert_eq!(indexed[1].topology.parent, None);
        assert!(indexed[0].topology.children.is_empty());
    }

    #[test]
    fn flat_cast_index_does_not_attach_later_top_level_function_to_impl() {
        let source = "impl Player {\n    fn movement_speed(&self) -> f32 { self.speed }\n}\n\nfn top_level() {}\n";
        let method_start = source.find("fn movement_speed").unwrap();
        let top_level_start = source.find("fn top_level").unwrap();
        let indexed = build_indexed_chunks_for_test(
            "root/src/player.rs",
            source,
            vec![
                language::CastChunk {
                    byte_range: 0..method_start,
                    non_whitespace_size: 10,
                    primary_node_kind: Some("type_identifier".to_string()),
                    merged_node_count: 1,
                },
                language::CastChunk {
                    byte_range: method_start..top_level_start,
                    non_whitespace_size: 42,
                    primary_node_kind: Some("identifier".to_string()),
                    merged_node_count: 1,
                },
                language::CastChunk {
                    byte_range: top_level_start..source.len(),
                    non_whitespace_size: 12,
                    primary_node_kind: Some("function_item".to_string()),
                    merged_node_count: 1,
                },
            ],
        );

        assert_eq!(indexed[1].topology.parent, Some(CodeChunkId(1)));
        assert_eq!(indexed[2].topology.parent, None);
        assert_eq!(indexed[0].topology.children, vec![CodeChunkId(2)]);
    }

    #[gpui::test]
    fn cast_index_preserves_parent_topology_for_rust_items(cx: &mut gpui::App) {
        use gpui::AppContext as _;
        use language::{Buffer, CastChunkingOptions, rust_lang};

        cx.new(|cx| {
            let source = "struct Player {\n    speed: f32,\n}\n\nimpl Player {\n    fn movement_speed(&self) -> f32 { self.speed }\n}\n";
            let buffer = Buffer::local(source, cx).with_language(rust_lang(), cx);
            let snapshot = buffer.snapshot();
            let chunks = language::cast_chunks_for_buffer(&snapshot, CastChunkingOptions::enabled(24))
                .expect("Rust syntax should produce chunks");
            let indexed = build_indexed_chunks_for_test("root/src/player.rs", source, chunks);
            assert!(indexed.iter().any(|chunk| chunk.topology.parent.is_some()));
            assert!(
                indexed
                    .iter()
                    .any(|chunk| !chunk.topology.enclosing_symbols.is_empty())
            );
            buffer
        });
    }
}
