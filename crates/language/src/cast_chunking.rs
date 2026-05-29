use std::ops::Range;

use crate::BufferSnapshot;
use tree_sitter::Node;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CastChunk {
    pub byte_range: Range<usize>,
    pub non_whitespace_size: usize,
    pub primary_node_kind: Option<String>,
    pub merged_node_count: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CastChunkingOptions {
    pub enabled: bool,
    pub max_size: usize,
}

impl CastChunkingOptions {
    pub const fn disabled(max_size: usize) -> Self {
        Self {
            enabled: false,
            max_size,
        }
    }

    pub const fn enabled(max_size: usize) -> Self {
        Self {
            enabled: true,
            max_size,
        }
    }
}

fn non_whitespace_size(text: &str) -> usize {
    text.chars()
        .filter(|character| !character.is_whitespace())
        .count()
}

fn range_non_whitespace_size(source: &str, range: Range<usize>) -> usize {
    source.get(range).map_or(0, non_whitespace_size)
}

fn fallback_text_chunks(source: &str, range: Range<usize>, max_size: usize) -> Vec<CastChunk> {
    let Some(text) = source.get(range.clone()) else {
        return Vec::new();
    };

    let max_size = max_size.max(1);
    let mut chunks = Vec::new();
    let mut chunk_start = range.start;
    let mut chunk_size = 0;
    let mut last_line_boundary = None;

    for (relative_index, character) in text.char_indices() {
        let byte_index = range.start + relative_index;
        let next_byte_index = byte_index + character.len_utf8();
        let character_size = usize::from(!character.is_whitespace());

        if character == '\n' && next_byte_index > chunk_start {
            last_line_boundary = Some(next_byte_index);
        }

        if chunk_size > 0 && chunk_size + character_size > max_size {
            let chunk_end = last_line_boundary
                .filter(|line_boundary| {
                    *line_boundary > chunk_start
                        && *line_boundary <= byte_index
                        && range_non_whitespace_size(source, *line_boundary..next_byte_index)
                            <= max_size
                })
                .unwrap_or(byte_index);
            push_text_chunk(source, &mut chunks, chunk_start..chunk_end);
            chunk_start = chunk_end;
            chunk_size = range_non_whitespace_size(source, chunk_start..next_byte_index);
            last_line_boundary = None;
        } else {
            chunk_size += character_size;
        }
    }

    if chunk_start < range.end {
        push_text_chunk(source, &mut chunks, chunk_start..range.end);
    }

    chunks
}

fn push_text_chunk(source: &str, chunks: &mut Vec<CastChunk>, byte_range: Range<usize>) {
    if byte_range.is_empty() {
        return;
    }

    chunks.push(CastChunk {
        byte_range: byte_range.clone(),
        non_whitespace_size: range_non_whitespace_size(source, byte_range),
        primary_node_kind: None,
        merged_node_count: 0,
    });
}

pub fn cast_chunks_for_node(source: &str, root: Node<'_>, max_size: usize) -> Vec<CastChunk> {
    let max_size = max_size.max(1);
    let root_range = root.start_byte()..root.end_byte();
    if root_range.is_empty() {
        return Vec::new();
    }

    if range_non_whitespace_size(source, root_range.clone()) <= max_size {
        let mut chunks = Vec::new();
        push_ast_chunk(
            source,
            &mut chunks,
            root_range,
            Some(root.kind().to_string()),
            1,
        );
        return chunks;
    }

    chunk_node_children(source, root, max_size, root_range)
}

fn chunk_node_children(
    source: &str,
    node: Node<'_>,
    max_size: usize,
    containing_range: Range<usize>,
) -> Vec<CastChunk> {
    let children = named_children(node);
    if children.is_empty() {
        return fallback_text_chunks(source, containing_range, max_size);
    }

    let mut chunks = Vec::new();
    let mut current_start = None;
    let mut current_end = containing_range.start;
    let mut current_primary_kind = None;
    let mut current_node_count = 0;

    for child in children {
        let child_range = child.start_byte()..child.end_byte();
        if child_range.is_empty()
            || child_range.start < containing_range.start
            || child_range.end > containing_range.end
        {
            continue;
        }

        let child_size = range_non_whitespace_size(source, child_range.clone());
        if child_size > max_size {
            flush_current_before_gap(
                source,
                &mut chunks,
                &mut current_start,
                &mut current_primary_kind,
                current_node_count,
                current_end,
                child_range.start,
                max_size,
            );
            chunks.extend(chunk_node_children(
                source,
                child,
                max_size,
                child_range.clone(),
            ));
            current_end = child_range.end;
            current_node_count = 0;
            continue;
        }

        let candidate_start = current_start.unwrap_or(current_end);
        let candidate_range = candidate_start..child_range.end;
        if range_non_whitespace_size(source, candidate_range) > max_size {
            flush_current_before_gap(
                source,
                &mut chunks,
                &mut current_start,
                &mut current_primary_kind,
                current_node_count,
                current_end,
                child_range.start,
                max_size,
            );
            current_start = Some(child_range.start);
            current_primary_kind = Some(child.kind().to_string());
            current_node_count = 1;
        } else {
            if current_start.is_none() {
                current_start = Some(candidate_start);
                current_primary_kind = Some(child.kind().to_string());
            }
            current_node_count += 1;
        }

        current_end = child_range.end;
    }

    if let Some(start) = current_start {
        let candidate_range = start..containing_range.end;
        if range_non_whitespace_size(source, candidate_range) <= max_size {
            push_ast_chunk(
                source,
                &mut chunks,
                start..containing_range.end,
                current_primary_kind,
                current_node_count,
            );
        } else {
            push_ast_chunk(
                source,
                &mut chunks,
                start..current_end,
                current_primary_kind,
                current_node_count,
            );
            push_fallback_text_chunks(
                source,
                &mut chunks,
                current_end..containing_range.end,
                max_size,
            );
        }
    } else {
        push_fallback_text_chunks(
            source,
            &mut chunks,
            current_end..containing_range.end,
            max_size,
        );
    }

    chunks
}

fn flush_current_before_gap(
    source: &str,
    chunks: &mut Vec<CastChunk>,
    current_start: &mut Option<usize>,
    current_primary_kind: &mut Option<String>,
    current_node_count: usize,
    current_end: usize,
    gap_end: usize,
    max_size: usize,
) {
    if let Some(start) = current_start.take() {
        if range_non_whitespace_size(source, start..gap_end) <= max_size {
            push_ast_chunk(
                source,
                chunks,
                start..gap_end,
                current_primary_kind.take(),
                current_node_count,
            );
        } else {
            push_ast_chunk(
                source,
                chunks,
                start..current_end,
                current_primary_kind.take(),
                current_node_count,
            );
            push_fallback_text_chunks(source, chunks, current_end..gap_end, max_size);
        }
    } else {
        push_fallback_text_chunks(source, chunks, current_end..gap_end, max_size);
    }
}

fn push_fallback_text_chunks(
    source: &str,
    chunks: &mut Vec<CastChunk>,
    byte_range: Range<usize>,
    max_size: usize,
) {
    chunks.extend(fallback_text_chunks(source, byte_range, max_size));
}

fn named_children(node: Node<'_>) -> Vec<Node<'_>> {
    let mut cursor = node.walk();
    node.named_children(&mut cursor).collect()
}

fn push_ast_chunk(
    source: &str,
    chunks: &mut Vec<CastChunk>,
    byte_range: Range<usize>,
    primary_node_kind: Option<String>,
    merged_node_count: usize,
) {
    if byte_range.is_empty() {
        return;
    }

    chunks.push(CastChunk {
        byte_range: byte_range.clone(),
        non_whitespace_size: range_non_whitespace_size(source, byte_range),
        primary_node_kind,
        merged_node_count,
    });
}

pub fn validate_chunks(chunks: &[CastChunk], source: &str) -> bool {
    if chunks.is_empty() {
        return false;
    }

    let mut previous_end = 0;

    for chunk in chunks {
        if chunk.byte_range.start >= chunk.byte_range.end {
            return false;
        }
        if chunk.byte_range.end > source.len() {
            return false;
        }
        if !source.is_char_boundary(chunk.byte_range.start)
            || !source.is_char_boundary(chunk.byte_range.end)
        {
            return false;
        }
        if chunk.byte_range.start != previous_end {
            return false;
        }
        previous_end = chunk.byte_range.end;
    }

    previous_end == source.len()
}

pub fn cast_chunks_for_buffer(
    snapshot: &BufferSnapshot,
    options: CastChunkingOptions,
) -> Option<Vec<CastChunk>> {
    if !options.enabled {
        return None;
    }

    let source = snapshot
        .text_for_range(0..snapshot.len())
        .collect::<String>();
    let root = select_full_buffer_root(
        snapshot.syntax_layers().map(|layer| layer.node()),
        snapshot.len(),
    )?;
    let chunks = cast_chunks_for_node(&source, root, options.max_size);
    if !validate_chunks(&chunks, &source) {
        None
    } else {
        Some(chunks)
    }
}

fn select_full_buffer_root<'a>(
    nodes: impl IntoIterator<Item = Node<'a>>,
    source_len: usize,
) -> Option<Node<'a>> {
    nodes
        .into_iter()
        .filter(|node| node.start_byte() == 0 && node.end_byte() >= source_len)
        .max_by_key(|node| node.end_byte() - node.start_byte())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Buffer, rust_lang};
    use gpui::{App, AppContext as _};

    #[test]
    fn cast_chunking_options_can_be_disabled() {
        assert_eq!(
            CastChunkingOptions::disabled(128),
            CastChunkingOptions {
                enabled: false,
                max_size: 128,
            }
        );
    }

    #[gpui::test]
    fn disabled_buffer_adapter_returns_none(cx: &mut App) {
        let buffer = cx.new(|cx| Buffer::local("fn a() {}", cx).with_language(rust_lang(), cx));
        let snapshot = buffer.update(cx, |buffer, _| buffer.snapshot());

        assert_eq!(
            cast_chunks_for_buffer(&snapshot, CastChunkingOptions::disabled(16)),
            None
        );
    }

    #[gpui::test]
    fn enabled_buffer_adapter_returns_ast_chunks(cx: &mut App) {
        cx.new(|cx| {
            let source = "fn a() {\n    one();\n}\n\nfn b() {\n    two();\n}\n";
            let buffer = Buffer::local(source, cx).with_language(rust_lang(), cx);
            let snapshot = buffer.snapshot();
            let chunks = cast_chunks_for_buffer(&snapshot, CastChunkingOptions::enabled(18))
                .expect("enabled adapter should return chunks for parsed Rust buffer");

            assert_eq!(reconstruct(source, &chunks), source);
            assert!(chunks.len() >= 2);
            assert!(chunks.iter().all(|chunk| chunk.non_whitespace_size > 0));
            buffer
        });
    }

    #[gpui::test]
    fn enabled_buffer_adapter_returns_none_without_syntax(cx: &mut App) {
        cx.new(|cx| {
            let buffer = Buffer::local("plain text without a language", cx);
            let snapshot = buffer.snapshot();
            assert_eq!(
                cast_chunks_for_buffer(&snapshot, CastChunkingOptions::enabled(18)),
                None
            );
            buffer
        });
    }

    #[test]
    fn counts_non_whitespace_characters() {
        assert_eq!(non_whitespace_size(" fn a() {\n    b();\n}\n"), 11);
    }

    #[test]
    fn fallback_split_preserves_source_text() {
        let source = "alpha beta\ngamma delta\nepsilon";
        let chunks = fallback_text_chunks(source, 0..source.len(), 10);
        assert_eq!(reconstruct(source, &chunks), source);
        assert!(chunks.iter().all(|chunk| chunk.non_whitespace_size <= 10));
        assert_eq!(chunks.len(), 3);
    }

    #[test]
    fn fallback_split_handles_tiny_budget_without_invalid_utf8() {
        let source = "ééé";
        let chunks = fallback_text_chunks(source, 0..source.len(), 1);
        assert_eq!(reconstruct(source, &chunks), source);
        assert_eq!(chunks.len(), 3);
        assert!(
            chunks
                .iter()
                .all(|chunk| source.is_char_boundary(chunk.byte_range.start))
        );
        assert!(
            chunks
                .iter()
                .all(|chunk| source.is_char_boundary(chunk.byte_range.end))
        );
    }

    #[test]
    fn fallback_split_does_not_exceed_budget_after_line_boundary() {
        let source = "\nabcdefghijk";
        let chunks = fallback_text_chunks(source, 0..source.len(), 10);
        assert_eq!(reconstruct(source, &chunks), source);
        assert!(chunks.iter().all(|chunk| chunk.non_whitespace_size <= 10));
    }

    #[test]
    fn fallback_split_rejects_invalid_ranges_without_panicking() {
        assert!(fallback_text_chunks("éx", 1..3, 1).is_empty());
        let reversed_start = 2;
        let reversed_end = 1;
        assert!(fallback_text_chunks("abc", reversed_start..reversed_end, 1).is_empty());
    }

    #[test]
    fn whole_root_that_fits_returns_one_chunk() {
        let source = "fn a() {}\n";
        let chunks = parse_rust_and_chunk(source, 32);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].byte_range, 0..source.len());
        assert_eq!(reconstruct(source, &chunks), source);
    }

    #[test]
    fn root_selection_prefers_full_buffer_root_over_partial_root_at_zero() {
        let partial_tree = parse_rust("fn a() {}\n");
        let full_source = "fn a() {}\nfn b() {}\n";
        let full_tree = parse_rust(full_source);

        let selected = select_full_buffer_root(
            [partial_tree.root_node(), full_tree.root_node()],
            full_source.len(),
        )
        .expect("full-buffer root should be selected");

        assert_eq!(selected.start_byte(), 0);
        assert_eq!(selected.end_byte(), full_source.len());
    }

    #[test]
    fn root_selection_rejects_partial_roots() {
        let partial_tree = parse_rust("fn a() {}\n");
        assert!(
            select_full_buffer_root([partial_tree.root_node()], "fn a() {}\nfn b() {}\n".len())
                .is_none()
        );
    }

    #[test]
    fn preserves_comments_and_blank_lines_between_nodes() {
        let source = "// leading comment\n\nfn a() {}\n\n// between\nfn b() {}\n";
        let chunks = parse_rust_and_chunk(source, 12);
        assert_eq!(reconstruct(source, &chunks), source);
        assert!(
            chunks
                .iter()
                .any(|chunk| source[chunk.byte_range.clone()].contains("between"))
        );
    }

    #[test]
    fn validates_ordered_non_overlapping_chunks() {
        let source = "abcdef";
        let valid = vec![test_chunk(0..3), test_chunk(3..6)];
        assert!(validate_chunks(&valid, source));

        let overlapping = vec![test_chunk(0..4), test_chunk(3..6)];
        assert!(!validate_chunks(&overlapping, source));
    }

    #[test]
    fn validation_rejects_incomplete_or_invalid_coverage() {
        assert!(!validate_chunks(
            &[test_chunk(0..2), test_chunk(3..6)],
            "abcdef"
        ));
        assert!(!validate_chunks(&[test_chunk(1..6)], "abcdef"));
        assert!(!validate_chunks(&[test_chunk(0..5)], "abcdef"));
        assert!(!validate_chunks(&[test_chunk(0..1)], "é"));
        assert!(!validate_chunks(&[], "abcdef"));
    }

    #[test]
    fn preserves_top_level_function_boundaries_when_possible() {
        let source = "fn a() {\n    one();\n}\n\nfn b() {\n    two();\n}\n";
        let chunks = parse_rust_and_chunk(source, 18);
        let chunk_texts: Vec<&str> = chunks
            .iter()
            .map(|chunk| &source[chunk.byte_range.clone()])
            .collect();

        assert_eq!(
            chunk_texts,
            vec!["fn a() {\n    one();\n}\n\n", "fn b() {\n    two();\n}\n"]
        );
        assert_eq!(reconstruct(source, &chunks), source);
    }

    #[test]
    fn merges_adjacent_small_siblings_until_budget_is_reached() {
        let source = "use a::A;\nuse b::B;\nuse c::C;\nfn main() {}\n";
        let chunks = parse_rust_and_chunk(source, 24);
        let chunk_texts: Vec<&str> = chunks
            .iter()
            .map(|chunk| &source[chunk.byte_range.clone()])
            .collect();

        assert_eq!(
            chunk_texts,
            vec!["use a::A;\nuse b::B;\nuse c::C;\n", "fn main() {}\n"]
        );
        assert_eq!(chunks[0].merged_node_count, 3);
        assert_eq!(reconstruct(source, &chunks), source);
    }

    #[test]
    fn oversized_node_recursively_splits_children() {
        let source = "fn main() {\n    let a = 1;\n    let b = 2;\n    let c = 3;\n}\n";
        let chunks = parse_rust_and_chunk(source, 12);
        assert!(chunks.len() > 1);
        assert_eq!(reconstruct(source, &chunks), source);
        assert!(chunks.iter().all(|chunk| !chunk.byte_range.is_empty()));
        assert!(chunks.iter().all(|chunk| chunk.non_whitespace_size <= 12));
    }

    #[test]
    fn trailing_non_named_gap_is_split_to_budget() {
        let source = "fn a() {}\n/*abcdefghijklmnop*/";
        let max_size = 8;
        let chunks = parse_rust_and_chunk(source, max_size);

        assert_eq!(reconstruct(source, &chunks), source);
        assert!(
            chunks
                .iter()
                .all(|chunk| chunk.non_whitespace_size <= max_size.max(1))
        );
    }

    #[test]
    fn unnamed_function_syntax_is_budgeted_when_splitting_children() {
        let source = "fn main() {\n    let alpha = beta;\n}\n";
        let max_size = 6;
        let chunks = parse_rust_and_chunk(source, max_size);

        assert_eq!(reconstruct(source, &chunks), source);
        assert!(
            chunks
                .iter()
                .all(|chunk| chunk.non_whitespace_size <= max_size.max(1))
        );
    }

    fn parse_rust_and_chunk(source: &str, max_size: usize) -> Vec<CastChunk> {
        let tree = parse_rust(source);
        cast_chunks_for_node(source, tree.root_node(), max_size)
    }

    fn parse_rust(source: &str) -> tree_sitter::Tree {
        let mut parser = tree_sitter::Parser::new();
        parser
            .set_language(&tree_sitter_rust::LANGUAGE.into())
            .expect("Rust parser should load");
        parser.parse(source, None).expect("source should parse")
    }

    fn reconstruct(source: &str, chunks: &[CastChunk]) -> String {
        chunks
            .iter()
            .map(|chunk| &source[chunk.byte_range.clone()])
            .collect::<String>()
    }

    fn test_chunk(byte_range: Range<usize>) -> CastChunk {
        CastChunk {
            non_whitespace_size: byte_range.len(),
            byte_range,
            primary_node_kind: None,
            merged_node_count: 0,
        }
    }
}
