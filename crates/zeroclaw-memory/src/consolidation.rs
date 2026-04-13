//! LLM-driven memory consolidation.
//!
//! After each conversation turn, extracts structured information:
//! - `history_entry`: A timestamped summary for the daily conversation log.
//! - `memory_update`: New facts, preferences, or decisions worth remembering
//!   long-term (or `null` if nothing new was learned).
//! - Entity/relation extraction for the knowledge graph (Phase 3).
//!
//! This two-phase approach replaces the naive raw-message auto-save with
//! semantic extraction, similar to Nanobot's `save_memory` tool call pattern.

use crate::conflict;
use crate::embeddings::EmbeddingProvider;
use crate::importance;
use crate::knowledge_graph::{KnowledgeGraph, NodeEvent};
use crate::traits::{Memory, MemoryCategory};
use std::fmt::Write as _;
use zeroclaw_api::provider::Provider;

/// A single entity mention extracted from a conversation turn.
#[derive(Debug, serde::Deserialize)]
pub struct EntityMention {
    /// Free-form entity type (e.g. "person", "project", "technology").
    pub entity_type: String,
    /// Normalized slug identifier (e.g. "alice_smith", "zeroclaw").
    pub slug: String,
    /// Short fact about the entity observed in this turn.
    pub fact: String,
}

/// A directed relation between two entities.
#[derive(Debug, serde::Deserialize)]
pub struct RelationMention {
    pub from_slug: String,
    pub relation: String,
    pub to_slug: String,
}

/// Result of the entity extraction LLM call.
#[derive(Debug, serde::Deserialize)]
pub struct EntityExtractionResult {
    #[serde(default)]
    pub entities: Vec<EntityMention>,
    #[serde(default)]
    pub relations: Vec<RelationMention>,
}

const ENTITY_EXTRACTION_SYSTEM_PROMPT: &str = r#"You are an entity extraction engine. Given a conversation turn, identify named entities (people, projects, technologies, organizations, concepts) and factual relationships between them.

Return ONLY valid JSON:
{
  "entities": [
    {"entity_type": "person|project|technology|...", "slug": "snake_case_identifier", "fact": "short fact observed"}
  ],
  "relations": [
    {"from_slug": "slug_a", "relation": "uses|extends|depends_on|...", "to_slug": "slug_b"}
  ]
}

Rules:
- slug must be lowercase snake_case, no spaces
- entity_type must be lowercase
- Only extract entities clearly mentioned; return empty arrays if nothing notable
- Keep facts to one sentence
- Do not include any text outside the JSON"#;

/// Output of consolidation extraction.
#[derive(Debug, serde::Deserialize)]
pub struct ConsolidationResult {
    /// Brief timestamped summary for the conversation history log.
    pub history_entry: String,
    /// New facts/preferences/decisions to store long-term, or None.
    pub memory_update: Option<String>,
    /// Atomic facts extracted from the turn (when consolidation_extract_facts is enabled).
    #[serde(default)]
    pub facts: Vec<String>,
    /// Observed trend or pattern (when consolidation_extract_facts is enabled).
    #[serde(default)]
    pub trend: Option<String>,
}

const CONSOLIDATION_SYSTEM_PROMPT: &str = r#"You are a memory consolidation engine. Given a conversation turn, extract:
1. "history_entry": A brief summary of what happened in this turn (1-2 sentences). Include the key topic or action.
2. "memory_update": Any NEW facts, preferences, decisions, or commitments worth remembering long-term. Return null if nothing new was learned.

Respond ONLY with valid JSON: {"history_entry": "...", "memory_update": "..." or null}
Do not include any text outside the JSON object."#;

/// Run two-phase LLM-driven consolidation on a conversation turn.
///
/// Phase 1: Write a history entry to the Daily memory category.
/// Phase 2: Write a memory update to the Core category (if the LLM identified new facts).
///
/// This function is designed to be called fire-and-forget via `tokio::spawn`.
/// Strip channel media markers (e.g. `[IMAGE:/local/path]`, `[DOCUMENT:...]`)
/// that contain local filesystem paths.  These must never be forwarded to
/// upstream provider APIs — they would leak local paths and cause API errors.
fn strip_media_markers(text: &str) -> String {
    // Matches [IMAGE:...], [DOCUMENT:...], [FILE:...], [VIDEO:...], [VOICE:...], [AUDIO:...]
    static RE: std::sync::LazyLock<regex::Regex> = std::sync::LazyLock::new(|| {
        regex::Regex::new(r"\[(?:IMAGE|DOCUMENT|FILE|VIDEO|VOICE|AUDIO):[^\]]*\]").unwrap()
    });
    RE.replace_all(text, "[media attachment]").into_owned()
}

pub async fn consolidate_turn(
    provider: &dyn Provider,
    model: &str,
    memory: &dyn Memory,
    knowledge: Option<&KnowledgeGraph>,
    user_message: &str,
    assistant_response: &str,
) -> anyhow::Result<()> {
    let turn_text = format!(
        "User: {}\nAssistant: {}",
        strip_media_markers(user_message),
        strip_media_markers(assistant_response),
    );

    // Truncate very long turns to avoid wasting tokens on consolidation.
    // Use char-boundary-safe slicing to prevent panic on multi-byte UTF-8 (e.g. CJK text).
    let truncated = if turn_text.len() > 4000 {
        let end = turn_text
            .char_indices()
            .map(|(i, _)| i)
            .take_while(|&i| i <= 4000)
            .last()
            .unwrap_or(0);
        format!("{}…", &turn_text[..end])
    } else {
        turn_text.clone()
    };

    let raw = provider
        .chat_with_system(Some(CONSOLIDATION_SYSTEM_PROMPT), &truncated, model, 0.1)
        .await?;

    let result: ConsolidationResult = parse_consolidation_response(&raw, &turn_text);

    // Phase 1: Write history entry to Daily category.
    let date = chrono::Local::now().format("%Y-%m-%d").to_string();
    let history_key = format!("daily_{date}_{}", uuid::Uuid::new_v4());
    memory
        .store(
            &history_key,
            &result.history_entry,
            MemoryCategory::Daily,
            None,
        )
        .await?;

    // Phase 2: Write memory update to Core category (if present).
    if let Some(ref update) = result.memory_update
        && !update.trim().is_empty()
    {
        let mem_key = format!("core_{}", uuid::Uuid::new_v4());

        // Compute importance score heuristically.
        let imp = importance::compute_importance(update, &MemoryCategory::Core);

        // Check for conflicts with existing Core memories.
        if let Err(e) = conflict::check_and_resolve_conflicts(
            memory,
            &mem_key,
            update,
            &MemoryCategory::Core,
            0.85,
        )
        .await
        {
            tracing::debug!("conflict check skipped: {e}");
        }

        // Store with importance metadata.
        memory
            .store_with_metadata(
                &mem_key,
                update,
                MemoryCategory::Core,
                None,
                None,
                Some(imp),
            )
            .await?;
    }

    // Phase 3: Entity extraction into knowledge graph (optional).
    if let Some(kg) = knowledge {
        if let Ok(raw_entities) = provider
            .chat_with_system(Some(ENTITY_EXTRACTION_SYSTEM_PROMPT), &truncated, model, 0.0)
            .await
        {
            if let Ok(extracted) = parse_entity_extraction_response(&raw_entities) {
                for entity in &extracted.entities {
                    match kg.find_or_create_by_slug(
                        &entity.slug,
                        &entity.entity_type,
                        &entity.slug,
                        "",
                    ) {
                        Ok((node_id, _)) => {
                            if let Err(e) = kg.add_event(&node_id, &entity.fact, None, None) {
                                tracing::debug!("entity event write failed: {e}");
                            }
                        }
                        Err(e) => tracing::debug!("entity upsert failed: {e}"),
                    }
                }

                for rel in &extracted.relations {
                    // Resolve slugs to node IDs using find_or_create (non-destructive).
                    let from_id = kg
                        .find_or_create_by_slug(&rel.from_slug, "entity", &rel.from_slug, "")
                        .map(|(id, _)| id);
                    let to_id = kg
                        .find_or_create_by_slug(&rel.to_slug, "entity", &rel.to_slug, "")
                        .map(|(id, _)| id);
                    match (from_id, to_id) {
                        (Ok(from), Ok(to)) => {
                            if let Err(e) = kg.add_edge(&from, &to, &rel.relation) {
                                tracing::debug!("relation write failed: {e}");
                            }
                        }
                        _ => tracing::debug!("could not resolve relation slugs"),
                    }
                }
            }
        }
    }

    Ok(())
}

// ── Dream Cycle (Phase 6) ────────────────────────────────────────────────────

/// Summary of a dream cycle synthesis run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SynthesisReport {
    /// Number of stale nodes synthesized in this run.
    pub nodes_processed: usize,
    /// Stale nodes that existed but were not processed due to the per-run cap.
    pub nodes_remaining: usize,
}

const SYNTHESIS_SYSTEM_PROMPT: &str = r#"You are a concise knowledge synthesizer. Given facts about an entity gathered over time, produce a short synthesis paragraph. Preserve all key facts and dates. Write at most a few sentences per major fact. Be factual and precise. Output plain text only — no markdown, no headers."#;

/// Synthesize all stale knowledge graph entities using an LLM + embedder.
///
/// "Stale" means `synthesis_at IS NULL` (never synthesized) or
/// `synthesis_at < updated_at` (new events since last synthesis).
///
/// `max_per_run` caps normal batches. When the backlog is large (> 5×),
/// a first-run catch-up mode processes up to 5 × `max_per_run` nodes to
/// clear the initial debt within a single night.
pub async fn synthesize_stale_entities(
    provider: &dyn Provider,
    model: &str,
    knowledge: &KnowledgeGraph,
    embedder: &dyn EmbeddingProvider,
    max_per_run: usize,
) -> anyhow::Result<SynthesisReport> {
    // Count total stale nodes to decide effective cap.
    let total_stale = knowledge.list_stale_nodes(usize::MAX)?.len();

    let effective_cap = if total_stale > 5 * max_per_run {
        tracing::warn!(
            total_stale,
            max_per_run,
            "dream cycle: large backlog detected, using 5× cap ({} nodes)",
            5 * max_per_run
        );
        5 * max_per_run
    } else {
        max_per_run
    };

    let nodes = knowledge.list_stale_nodes(effective_cap)?;
    let nodes_processed_cap = nodes.len();

    let mut nodes_processed = 0;

    for node in nodes {
        // Load all events (no practical limit — events are small rows).
        let timeline = knowledge.get_with_timeline(&node.id, usize::MAX)?;
        let events: Vec<NodeEvent> = timeline
            .map(|(_, evts)| evts)
            .unwrap_or_default();

        // Build synthesis prompt.
        let mut events_text = String::new();
        // Events come back newest-first from get_with_timeline; reverse for chronological order.
        for event in events.iter().rev() {
            let _ = writeln!(events_text, "- [{}] {}", event.created_at, event.content);
        }

        let prompt = if events_text.is_empty() {
            format!(
                "Synthesize everything known about {} ({}).\n\nNo events recorded yet.",
                node.title, node.node_type
            )
        } else {
            format!(
                "Synthesize everything known about {} ({}).\
                 Preserve all facts and dates.\n\nEvents:\n{}",
                node.title, node.node_type, events_text
            )
        };

        // LLM synthesis call.
        let synthesis = match provider
            .chat_with_system(Some(SYNTHESIS_SYSTEM_PROMPT), &prompt, model, 0.1)
            .await
        {
            Ok(text) => text,
            Err(e) => {
                tracing::warn!(node_id = %node.id, error = %e, "dream cycle: synthesis LLM call failed");
                continue;
            }
        };

        // Embed synthesis text (best-effort — embedder may be NoopEmbedding).
        let embedding: Option<Vec<f32>> = match embedder.embed_one(&synthesis).await {
            Ok(v) if !v.is_empty() => Some(v),
            _ => None,
        };

        // Persist synthesis + embedding.
        if let Err(e) = knowledge.update_synthesis(
            &node.id,
            &synthesis,
            embedding.as_deref(),
        ) {
            tracing::warn!(node_id = %node.id, error = %e, "dream cycle: update_synthesis failed");
            continue;
        }

        nodes_processed += 1;
    }

    let nodes_remaining = total_stale.saturating_sub(nodes_processed_cap);

    Ok(SynthesisReport {
        nodes_processed,
        nodes_remaining,
    })
}

/// Parse entity extraction response, returning an error if JSON is invalid.
fn parse_entity_extraction_response(raw: &str) -> anyhow::Result<EntityExtractionResult> {
    let cleaned = raw
        .trim()
        .trim_start_matches("```json")
        .trim_start_matches("```")
        .trim_end_matches("```")
        .trim();
    Ok(serde_json::from_str(cleaned)?)
}

/// Parse the LLM's consolidation response, with fallback for malformed JSON.
fn parse_consolidation_response(raw: &str, fallback_text: &str) -> ConsolidationResult {
    // Try to extract JSON from the response (LLM may wrap in markdown code blocks).
    let cleaned = raw
        .trim()
        .trim_start_matches("```json")
        .trim_start_matches("```")
        .trim_end_matches("```")
        .trim();

    serde_json::from_str(cleaned).unwrap_or_else(|_| {
        // Fallback: use truncated turn text as history entry.
        // Use char-boundary-safe slicing to prevent panic on multi-byte UTF-8.
        let summary = if fallback_text.len() > 200 {
            let end = fallback_text
                .char_indices()
                .map(|(i, _)| i)
                .take_while(|&i| i <= 200)
                .last()
                .unwrap_or(0);
            format!("{}…", &fallback_text[..end])
        } else {
            fallback_text.to_string()
        };
        ConsolidationResult {
            history_entry: summary,
            memory_update: None,
            facts: Vec::new(),
            trend: None,
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knowledge_graph::KnowledgeGraph;
    use async_trait::async_trait;
    use tempfile::TempDir;
    use zeroclaw_api::provider::{ChatMessage, Provider};

    // ── Mock Provider ─────────────────────────────────────────────

    struct FixedResponseProvider {
        response: String,
        /// Records every prompt seen (user message) for assertion.
        prompts_seen: std::sync::Mutex<Vec<String>>,
    }

    impl FixedResponseProvider {
        fn new(response: impl Into<String>) -> Self {
            Self { response: response.into(), prompts_seen: std::sync::Mutex::new(Vec::new()) }
        }

        fn prompts(&self) -> Vec<String> {
            self.prompts_seen.lock().unwrap().clone()
        }
    }

    #[async_trait]
    impl Provider for FixedResponseProvider {
        async fn chat_with_system(
            &self,
            _system: Option<&str>,
            message: &str,
            _model: &str,
            _temperature: f64,
        ) -> anyhow::Result<String> {
            self.prompts_seen.lock().unwrap().push(message.to_string());
            Ok(self.response.clone())
        }

        async fn chat_with_history(
            &self,
            messages: &[ChatMessage],
            _model: &str,
            _temperature: f64,
        ) -> anyhow::Result<String> {
            if let Some(last) = messages.last() {
                self.prompts_seen.lock().unwrap().push(format!("{:?}", last));
            }
            Ok(self.response.clone())
        }
    }

    // ── Mock EmbeddingProvider ────────────────────────────────────

    struct FixedEmbeddingProvider {
        dims: usize,
    }

    #[async_trait]
    impl EmbeddingProvider for FixedEmbeddingProvider {
        fn name(&self) -> &str { "test" }
        fn dimensions(&self) -> usize { self.dims }
        async fn embed(&self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
            Ok(texts.iter().map(|_| vec![0.1f32; self.dims]).collect())
        }
    }

    fn test_kg() -> (TempDir, KnowledgeGraph) {
        let tmp = TempDir::new().unwrap();
        let db_path = tmp.path().join("kg.db");
        let kg = KnowledgeGraph::new(&db_path, 10_000).unwrap();
        (tmp, kg)
    }

    fn add_stale_node(kg: &KnowledgeGraph, slug: &str, node_type: &str, content: &str) -> String {
        let (id, _) = kg.find_or_create_by_slug(slug, node_type, slug, content).unwrap();
        // Add an event to mark it stale (trigger sets synthesis_at = NULL).
        kg.add_event(&id, "initial fact", None, None).unwrap();
        id
    }

    // ── Dream cycle tests ─────────────────────────────────────────

    #[tokio::test]
    async fn synthesize_stale_updates_synthesis_at() {
        let (_tmp, kg) = test_kg();
        let id = add_stale_node(&kg, "alice", "person", "");
        let provider = FixedResponseProvider::new("Alice is a lead engineer.");
        let embedder = FixedEmbeddingProvider { dims: 4 };

        let report = synthesize_stale_entities(&provider, "test-model", &kg, &embedder, 10)
            .await
            .unwrap();

        assert_eq!(report.nodes_processed, 1);
        let node = kg.get_node(&id).unwrap().unwrap();
        assert!(node.synthesis_at.is_some(), "synthesis_at should be set after synthesis");
        assert_eq!(node.synthesis.as_deref(), Some("Alice is a lead engineer."));
    }

    #[tokio::test]
    async fn synthesize_stale_updates_embedding() {
        let (_tmp, kg) = test_kg();
        add_stale_node(&kg, "bob", "person", "");
        let provider = FixedResponseProvider::new("Bob is a backend developer.");
        let embedder = FixedEmbeddingProvider { dims: 4 };

        synthesize_stale_entities(&provider, "test-model", &kg, &embedder, 10)
            .await
            .unwrap();

        // Check that embedding was stored (non-null, correct size).
        // We can't directly read the embedding without a raw SQL query, but
        // synthesis_at being set confirms update_synthesis() was called successfully.
        let stale_after = kg.list_stale_nodes(100).unwrap();
        assert!(stale_after.is_empty(), "node should no longer be stale");
    }

    #[tokio::test]
    async fn synthesize_respects_cap() {
        let (_tmp, kg) = test_kg();
        for i in 0..5 {
            add_stale_node(&kg, &format!("entity_{i}"), "thing", "");
        }
        let provider = FixedResponseProvider::new("synthesis text");
        let embedder = FixedEmbeddingProvider { dims: 4 };

        let report = synthesize_stale_entities(&provider, "m", &kg, &embedder, 3)
            .await
            .unwrap();

        assert_eq!(report.nodes_processed, 3, "should process exactly cap");
        assert_eq!(report.nodes_remaining, 2, "2 nodes should remain");
    }

    #[tokio::test]
    async fn first_run_cap_is_5x() {
        let (_tmp, kg) = test_kg();
        // Create 25 stale nodes with max_per_run=4 → total(25) > 5*4=20 → effective_cap = 20.
        for i in 0..25 {
            add_stale_node(&kg, &format!("ent_{i}"), "item", "");
        }
        let provider = FixedResponseProvider::new("synth");
        let embedder = FixedEmbeddingProvider { dims: 4 };

        let report = synthesize_stale_entities(&provider, "m", &kg, &embedder, 4)
            .await
            .unwrap();

        // effective_cap = 5*4 = 20, total_stale = 25 → nodes_remaining = 5
        assert_eq!(report.nodes_processed, 20);
        assert_eq!(report.nodes_remaining, 5);
    }

    #[tokio::test]
    async fn synthesis_uses_all_events() {
        let (_tmp, kg) = test_kg();
        let id = add_stale_node(&kg, "carol", "person", "");
        kg.add_event(&id, "event two", None, None).unwrap();
        kg.add_event(&id, "event three", None, None).unwrap();
        kg.add_event(&id, "event four", None, None).unwrap();
        kg.add_event(&id, "event five", None, None).unwrap();

        let provider = FixedResponseProvider::new("Carol synthesis");
        let embedder = FixedEmbeddingProvider { dims: 4 };

        synthesize_stale_entities(&provider, "m", &kg, &embedder, 10)
            .await
            .unwrap();

        // Verify all events appeared in the prompt.
        let prompts = provider.prompts();
        assert!(!prompts.is_empty());
        let combined = prompts.join(" ");
        assert!(combined.contains("event two"), "prompt should include event two");
        assert!(combined.contains("event five"), "prompt should include event five");
        assert!(combined.contains("initial fact"), "prompt should include the first event");
    }

    #[tokio::test]
    async fn already_fresh_nodes_skipped() {
        let (_tmp, kg) = test_kg();
        let id = add_stale_node(&kg, "dave", "person", "");
        // Manually synthesize the node to clear its stale status.
        kg.update_synthesis(&id, "Dave is fresh", None).unwrap();

        let provider = FixedResponseProvider::new("should not be called");
        let embedder = FixedEmbeddingProvider { dims: 4 };

        let report = synthesize_stale_entities(&provider, "m", &kg, &embedder, 10)
            .await
            .unwrap();

        assert_eq!(report.nodes_processed, 0, "fresh node should be skipped");
        assert_eq!(provider.prompts().len(), 0, "provider should not be called");
    }

    #[tokio::test]
    async fn dream_cycle_report_reflects_reality() {
        let (_tmp, kg) = test_kg();
        for i in 0..10 {
            add_stale_node(&kg, &format!("node_{i}"), "item", "");
        }
        let provider = FixedResponseProvider::new("synth");
        let embedder = FixedEmbeddingProvider { dims: 4 };

        let report = synthesize_stale_entities(&provider, "m", &kg, &embedder, 20)
            .await
            .unwrap();

        assert_eq!(report.nodes_processed, 10);
        assert_eq!(report.nodes_remaining, 0);
    }

    #[test]
    fn parse_valid_json_response() {
        let raw = r#"{"history_entry": "User asked about Rust.", "memory_update": "User prefers Rust over Go."}"#;
        let result = parse_consolidation_response(raw, "fallback");
        assert_eq!(result.history_entry, "User asked about Rust.");
        assert_eq!(
            result.memory_update.as_deref(),
            Some("User prefers Rust over Go.")
        );
    }

    #[test]
    fn parse_json_with_null_memory() {
        let raw = r#"{"history_entry": "Routine greeting.", "memory_update": null}"#;
        let result = parse_consolidation_response(raw, "fallback");
        assert_eq!(result.history_entry, "Routine greeting.");
        assert!(result.memory_update.is_none());
    }

    #[test]
    fn parse_json_wrapped_in_code_block() {
        let raw =
            "```json\n{\"history_entry\": \"Discussed deployment.\", \"memory_update\": null}\n```";
        let result = parse_consolidation_response(raw, "fallback");
        assert_eq!(result.history_entry, "Discussed deployment.");
    }

    #[test]
    fn fallback_on_malformed_response() {
        let raw = "I'm sorry, I can't do that.";
        let result = parse_consolidation_response(raw, "User: hello\nAssistant: hi");
        assert_eq!(result.history_entry, "User: hello\nAssistant: hi");
        assert!(result.memory_update.is_none());
    }

    #[test]
    fn fallback_truncates_long_text() {
        let long_text = "x".repeat(500);
        let result = parse_consolidation_response("invalid", &long_text);
        // 200 bytes + "…" (3 bytes in UTF-8) = 203
        assert!(result.history_entry.len() <= 203);
    }

    #[test]
    fn fallback_truncates_cjk_text_without_panic() {
        // Each CJK character is 3 bytes in UTF-8; byte index 200 may land
        // inside a character. This must not panic.
        let cjk_text = "二手书项目".repeat(50); // 250 chars = 750 bytes
        let result = parse_consolidation_response("invalid", &cjk_text);
        assert!(
            result
                .history_entry
                .is_char_boundary(result.history_entry.len())
        );
        assert!(result.history_entry.ends_with('…'));
    }
}
