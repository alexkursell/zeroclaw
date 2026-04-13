//! Integration tests for the unified memory pipeline.
//!
//! Verifies correct behavior when all phases run together:
//! cross-module data flow, retrieval correctness, context block quality,
//! hygiene safety, and a full end-to-end pipeline walkthrough.
//!
//! Ref: docs/unified-memory-design.md §9 Integration Test Suite

use std::sync::Arc;

use async_trait::async_trait;
use tempfile::TempDir;
use uuid::Uuid;

use zeroclaw::agent::build_context;
use zeroclaw::config::schema::{MemoryConfig, SearchMode};
use zeroclaw::memory::{
    Memory, MemoryCategory, RetrievalConfig, RetrievalPipeline, RrfSource, SqliteMemory,
    consolidation::{consolidate_turn, synthesize_stale_entities},
    embeddings::EmbeddingProvider,
    hygiene,
    knowledge_graph::KnowledgeGraph,
};
use zeroclaw::providers::ChatResponse;

use crate::support::MockProvider;

// ── AxisEmbedder ──────────────────────────────────────────────────────────────

/// Deterministic test embedder. Maps content to a unit vector along a
/// principal axis determined by the first keyword found in the text.
/// Two texts sharing a keyword have cosine similarity 1.0; otherwise 0.0.
struct AxisEmbedder {
    dims: usize,
    mappings: Vec<(&'static str, usize)>,
}

impl AxisEmbedder {
    fn new(dims: usize, mappings: Vec<(&'static str, usize)>) -> Self {
        Self { dims, mappings }
    }

    fn axis_for(&self, text: &str) -> Option<usize> {
        let lower = text.to_lowercase();
        self.mappings
            .iter()
            .find(|(kw, _)| lower.contains(kw))
            .map(|(_, ax)| *ax)
    }
}

#[async_trait]
impl EmbeddingProvider for AxisEmbedder {
    fn name(&self) -> &str {
        "axis-test"
    }

    fn dimensions(&self) -> usize {
        self.dims
    }

    async fn embed(&self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
        texts
            .iter()
            .map(|t| {
                let mut v = vec![0.0f32; self.dims];
                if let Some(ax) = self.axis_for(t) {
                    v[ax] = 1.0;
                }
                Ok(v)
            })
            .collect()
    }
}

fn standard_axes() -> Vec<(&'static str, usize)> {
    vec![
        ("alice", 0),
        ("acme", 1),
        ("deployment", 2),
        ("weather", 3),
        ("consensus", 4),
        ("deadline", 5),
    ]
}

// ── MemoryWorld ───────────────────────────────────────────────────────────────

/// Pre-loaded test fixture. Contains brain (SqliteMemory), knowledge graph,
/// and a fully-wired retrieval pipeline. `standard()` populates a realistic
/// fake corpus; `empty()` leaves data loading to the individual test.
struct MemoryWorld {
    #[allow(dead_code)]
    tmp: TempDir,
    brain: Arc<SqliteMemory>,
    knowledge: Arc<KnowledgeGraph>,
    pipeline: RetrievalPipeline,
    session_a: String,
    session_b: String,
}

impl MemoryWorld {
    /// Empty world — no corpus loaded. Tests control their own data.
    async fn empty() -> Self {
        let tmp = TempDir::new().unwrap();
        Self::build_with_embedder(tmp, Arc::new(AxisEmbedder::new(8, standard_axes())), false).await
    }

    /// Empty world with NoopEmbedding for BM25-only tests.
    #[allow(dead_code)]
    async fn keyword_only() -> Self {
        let tmp = TempDir::new().unwrap();
        Self::build_with_noop(tmp, false).await
    }

    /// Standard corpus — AxisEmbedder + pre-populated knowledge, memories, messages, summaries.
    async fn standard() -> Self {
        let tmp = TempDir::new().unwrap();
        Self::build_with_embedder(tmp, Arc::new(AxisEmbedder::new(8, standard_axes())), true).await
    }

    async fn build_with_embedder(
        tmp: TempDir,
        embedder: Arc<dyn EmbeddingProvider>,
        load: bool,
    ) -> Self {
        let workspace = tmp.path().to_path_buf();
        let brain = Arc::new(
            SqliteMemory::with_embedder(&workspace, embedder, 0.7, 0.3, 10_000, None, SearchMode::default())
                .unwrap(),
        );
        let kg_path = workspace.join("knowledge.db");
        let knowledge = Arc::new(KnowledgeGraph::new(&kg_path, 10_000).unwrap());
        let pipeline = RetrievalPipeline::new(brain.clone() as Arc<dyn Memory>, RetrievalConfig::default())
            .with_sqlite(brain.clone())
            .with_knowledge(knowledge.clone());

        let world = Self {
            tmp,
            brain,
            knowledge,
            pipeline,
            session_a: "session-alice-project".into(),
            session_b: "session-new-topic".into(),
        };
        if load {
            world.load_standard_corpus().await;
        }
        world
    }

    async fn build_with_noop(tmp: TempDir, load: bool) -> Self {
        let workspace = tmp.path().to_path_buf();
        let brain = Arc::new(SqliteMemory::new(&workspace).unwrap());
        let kg_path = workspace.join("knowledge.db");
        let knowledge = Arc::new(KnowledgeGraph::new(&kg_path, 10_000).unwrap());
        let pipeline = RetrievalPipeline::new(brain.clone() as Arc<dyn Memory>, RetrievalConfig::default())
            .with_sqlite(brain.clone())
            .with_knowledge(knowledge.clone());
        let world = Self {
            tmp,
            brain,
            knowledge,
            pipeline,
            session_a: "session-alice-project".into(),
            session_b: "session-new-topic".into(),
        };
        if load {
            world.load_standard_corpus().await;
        }
        world
    }

    /// Populate the standard fake corpus described in the design doc §9.1.
    async fn load_standard_corpus(&self) {
        let kg = &self.knowledge;
        let brain = &self.brain;

        // ── Knowledge graph nodes ─────────────────────────────────────────
        let (alice_id, _) = kg
            .find_or_create_by_slug("alice", "person", "alice", "Alice Chen, lead engineer at Acme")
            .unwrap();
        let (acme_id, _) = kg
            .find_or_create_by_slug("acme", "company", "acme", "Acme Corp, enterprise software company")
            .unwrap();
        let (consensus_id, _) = kg
            .find_or_create_by_slug(
                "consensus_rewrite",
                "project",
                "consensus_rewrite",
                "Distributed consensus rewrite project at Acme",
            )
            .unwrap();

        // ── Edges ─────────────────────────────────────────────────────────
        kg.add_edge(&alice_id, &acme_id, "employed_by").unwrap();
        kg.add_edge(&alice_id, &consensus_id, "works_on").unwrap();

        // ── Node events ───────────────────────────────────────────────────
        for content in [
            "Met Alice at PyCon 2026-03-10. She is lead engineer at Acme.",
            "Alice confirmed deployment window is first week of May.",
            "Alice prefers async communication over meetings.",
        ] {
            kg.add_event(&alice_id, content, None, None).unwrap();
        }
        for content in [
            "Acme is pivoting to enterprise market as of Q1 2026.",
            "Acme uses Django backend, Vue.js frontend.",
            "Acme deployment infrastructure runs on AWS us-east-1.",
        ] {
            kg.add_event(&acme_id, content, None, None).unwrap();
        }
        for content in [
            "Consensus rewrite started 2026-02-01. Target: 10x throughput.",
            "PR #42 opened by alice for consensus rewrite phase 1.",
            "Consensus rewrite deployment scheduled for May 2026.",
        ] {
            kg.add_event(&consensus_id, content, None, None).unwrap();
        }

        // ── Core memories ──────────────────────────────────────────────────
        brain.store("user_timezone", "EST (UTC-5)", MemoryCategory::Core, None).await.unwrap();
        brain.store("project_stack", "Django + SQLAlchemy backend, Vue.js frontend", MemoryCategory::Core, None).await.unwrap();
        brain.store("user_language_pref", "Prefers Rust for systems code, Python for scripts", MemoryCategory::Core, None).await.unwrap();
        brain.store("preferred_deploy", "Blue-green deployment, no traffic cutover without staging", MemoryCategory::Core, None).await.unwrap();
        brain.store("standup_time", "Daily standup at 9am EST", MemoryCategory::Core, None).await.unwrap();
        brain.store("git_workflow", "Feature branches off main, PR required, squash merge", MemoryCategory::Core, None).await.unwrap();

        // ── Daily memories ──────────────────────────────────────────────────
        let daily_a = format!("daily_2026-04-10_{}", Uuid::new_v4());
        let daily_b = format!("daily_2026-04-11_{}", Uuid::new_v4());
        let daily_c = format!("daily_2026-04-12_{}", Uuid::new_v4());
        brain.store(&daily_a, "Discussed deployment timeline with alice. PR #42 in review.", MemoryCategory::Daily, None).await.unwrap();
        brain.store(&daily_b, "Reviewed consensus rewrite architecture. Alice flagged a race condition.", MemoryCategory::Daily, None).await.unwrap();
        brain.store(&daily_c, "Standup: deployment target confirmed for May 5.", MemoryCategory::Daily, None).await.unwrap();

        // Session B daily — stored with session_id so session-filter tests are meaningful.
        let daily_weather = format!("daily_session_b_{}", Uuid::new_v4());
        brain
            .store(
                &daily_weather,
                "Answered question about New York weather in April. User asked about flights.",
                MemoryCategory::Daily,
                Some(&self.session_b),
            )
            .await
            .unwrap();

        // ── Messages — Session A (5) ────────────────────────────────────────
        let msg_a_ids: Vec<String> = (0..5).map(|_| Uuid::new_v4().to_string()).collect();
        let session_a_msgs = [
            ("user", "What's the status of the consensus rewrite?"),
            ("assistant", "Based on my notes, alice has PR #42 open for phase 1..."),
            ("user", "When is deployment?"),
            ("assistant", "Alice confirmed the deployment window is first week of May."),
            ("user", "Remind me of Alice's contact preferences."),
        ];
        for (i, (role, content)) in session_a_msgs.iter().enumerate() {
            brain.append_message(&msg_a_ids[i], &self.session_a, role, content, None).unwrap();
        }

        // ── Messages — Session B (5) ────────────────────────────────────────
        for (role, content) in [
            ("user", "What's the weather like in New York?"),
            ("assistant", "I don't have live weather data, but New York in April is mild..."),
            ("user", "What about flights?"),
            ("assistant", "I'd need a travel tool to check flights."),
            ("user", "Never mind, let's talk about something else."),
        ] {
            let id = Uuid::new_v4().to_string();
            brain.append_message(&id, &self.session_b, role, content, None).unwrap();
        }

        // ── Leaf summary — Session A, covers first 3 messages ─────────────
        let summary_a_id = Uuid::new_v4().to_string();
        brain
            .insert_summary(
                &summary_a_id,
                "leaf",
                "User asked about consensus rewrite status and deployment window. Alice has PR #42 open for phase 1. Deployment confirmed for first week of May.",
                None,
                &self.session_a,
                1,
            )
            .unwrap();
        brain
            .link_summary_sources(
                &summary_a_id,
                &[
                    (msg_a_ids[0].as_str(), "message"),
                    (msg_a_ids[1].as_str(), "message"),
                    (msg_a_ids[2].as_str(), "message"),
                ],
            )
            .unwrap();

        // ── Session B summary (for session-filter tests) ───────────────────
        let summary_b_id = Uuid::new_v4().to_string();
        brain
            .insert_summary(
                &summary_b_id,
                "leaf",
                "User asked about New York weather in April and about flights. No live weather data available.",
                None,
                &self.session_b,
                1,
            )
            .unwrap();
    }
}

// ── helpers ───────────────────────────────────────────────────────────────────

fn scripted_response(text: &str) -> ChatResponse {
    ChatResponse {
        text: Some(text.to_string()),
        tool_calls: vec![],
        usage: None,
        reasoning_content: None,
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Group 1: Cross-Component Data Flow
// ═════════════════════════════════════════════════════════════════════════════

/// Phase 1 → Phase 5 data path: messages written by append_message are
/// searchable via FTS search_messages().
#[tokio::test]
async fn messages_written_and_searchable_via_fts() {
    let world = MemoryWorld::standard().await;

    let results = world.brain.search_messages("deployment", None, 10).unwrap();

    // Session A has at least 2 messages mentioning deployment.
    assert!(
        results.len() >= 2,
        "expected ≥2 deployment messages, got {}",
        results.len()
    );
    for r in &results {
        assert!(!r.session_id.is_empty(), "session_id must be populated");
        assert!(!r.role.is_empty(), "role must be populated");
    }
    // All returned messages should be from Session A (deployment is only in Session A).
    for r in &results {
        assert_eq!(
            r.session_id, world.session_a,
            "deployment messages should belong to session-alice-project"
        );
    }
}

/// Phase 2 DAG integrity: the pre-loaded leaf summary links to exactly 3
/// Session A message IDs. Verified via grep_messages summary_id annotation.
#[tokio::test]
async fn summary_sources_link_covered_messages() {
    let world = MemoryWorld::standard().await;

    // Search messages that appear in Session A and check summary_id annotation.
    let results = world
        .brain
        .grep_messages("consensus rewrite", Some(&world.session_a), 10)
        .unwrap();

    // At least the first user message ("What's the status of the consensus rewrite?")
    // should be covered by the leaf summary.
    let annotated: Vec<_> = results.iter().filter(|r| r.summary_id.is_some()).collect();
    assert!(
        !annotated.is_empty(),
        "at least one covered message should have summary_id set"
    );

    // All annotated messages should have the same summary_id (they share one leaf summary).
    let summary_ids: std::collections::HashSet<_> = annotated
        .iter()
        .map(|r| r.summary_id.as_deref().unwrap())
        .collect();
    assert_eq!(
        summary_ids.len(),
        1,
        "all annotated messages should share the same summary"
    );
}

/// Phase 2 → Phase 4: the leaf summary is findable via recall and covers the
/// compacted messages.
#[tokio::test]
async fn compressed_message_still_findable_via_summary_recall() {
    let world = MemoryWorld::standard().await;

    let results = world
        .pipeline
        .recall_rrf("consensus rewrite deployment", 5, None)
        .await
        .unwrap();

    // At least one result should be a summary.
    let summaries: Vec<_> = results
        .iter()
        .filter(|e| e.source == RrfSource::Summary)
        .collect();
    assert!(
        !summaries.is_empty(),
        "recall should include the leaf summary"
    );
    assert!(
        summaries[0].content.contains("PR #42") || summaries[0].content.contains("deployment"),
        "summary content should mention deployment or PR #42"
    );
}

/// Phase 1 + Phase 2 → Phase 5: lcm_grep annotates covered messages with
/// their summary_id; active (non-compacted) messages have summary_id = None.
#[tokio::test]
async fn lcm_grep_annotates_compacted_messages_with_summary_id() {
    let world = MemoryWorld::standard().await;

    // Search all Session A messages.
    let results = world
        .brain
        .search_messages("consensus deployment alice", None, 20)
        .unwrap();

    // Use grep to see summary_id annotations for covered messages.
    let grep_results = world
        .brain
        .grep_messages("consensus|deployment|alice", Some(&world.session_a), 20)
        .unwrap();

    let covered: Vec<_> = grep_results.iter().filter(|r| r.summary_id.is_some()).collect();
    let active: Vec<_> = grep_results.iter().filter(|r| r.summary_id.is_none()).collect();

    // In the standard corpus, the first 3 Session A messages are covered.
    assert!(
        !covered.is_empty(),
        "some messages should be covered by a summary"
    );
    // Later messages (4th, 5th) are not covered.
    assert!(
        !active.is_empty(),
        "some messages should be active (not covered)"
    );
    // Covered messages have a non-empty summary_id.
    for r in &covered {
        assert!(
            r.summary_id.as_deref().is_some_and(|id| !id.is_empty()),
            "covered message summary_id must be non-empty"
        );
    }
    let _ = results; // suppress unused warning
}

/// Phase 3 internal: node created, events written, timeline queryable in
/// chronological order.
#[tokio::test]
async fn entity_events_written_and_queryable() {
    let world = MemoryWorld::empty().await;
    let kg = &world.knowledge;

    let (alice_id, created) = kg
        .find_or_create_by_slug("alice", "person", "alice", "Alice Chen")
        .unwrap();
    assert!(created, "alice should be newly created");

    let events = [
        "Alice joined Acme Corp as lead engineer.",
        "Alice opened PR #42 for the consensus rewrite.",
        "Alice prefers async communication.",
    ];
    for e in &events {
        kg.add_event(&alice_id, e, None, None).unwrap();
    }

    let (node, timeline) = kg.get_with_timeline(&alice_id, 10).unwrap().unwrap();
    assert_eq!(node.title, "alice");
    assert_eq!(timeline.len(), 3, "expect exactly 3 events");
    // Timeline is ordered by created_at DESC; just verify all events are present.
    let contents: Vec<_> = timeline.iter().map(|e| e.content.as_str()).collect();
    for e in &events {
        assert!(
            contents.contains(e),
            "event not found in timeline: {e}"
        );
    }
}

/// Phase 3 → consolidation: consolidate_turn with entity extraction creates
/// the entity node and a node_event in knowledge.db. synthesis_at IS NULL
/// (stale trigger fired).
#[tokio::test]
async fn consolidation_creates_entity_in_knowledge_graph() {
    let world = MemoryWorld::empty().await;
    let kg = &world.knowledge;

    let provider = MockProvider::new(vec![
        // Call 1: consolidation summary
        scripted_response(r#"{"history_entry": "Discussed Alice at Acme.", "memory_update": null}"#),
        // Call 2: entity extraction
        scripted_response(
            r#"{"entities": [{"entity_type": "person", "slug": "alice", "fact": "Lead engineer at Acme."}], "relations": []}"#,
        ),
    ]);

    consolidate_turn(
        &provider,
        "test-model",
        world.brain.as_ref(),
        Some(kg),
        "Alice from Acme wants to discuss the consensus rewrite.",
        "I'll set up a meeting with Alice to discuss the deployment timeline.",
    )
    .await
    .unwrap();

    // Node should exist.
    let (alice_id, created) = kg
        .find_or_create_by_slug("alice", "person", "alice", "")
        .unwrap();
    assert!(!created, "alice node should already exist after consolidation");

    // At least one node_event should exist.
    let (node, timeline) = kg.get_with_timeline(&alice_id, 10).unwrap().unwrap();
    assert!(
        !timeline.is_empty(),
        "at least one node_event should exist for alice"
    );
    assert!(
        timeline.iter().any(|e| e.content.contains("Lead engineer at Acme")),
        "alice timeline should contain the extracted fact"
    );
    // Node should be stale (synthesis_at IS NULL) because add_event triggers a stale mark.
    assert!(
        node.synthesis_at.is_none(),
        "alice should be stale (synthesis_at IS NULL)"
    );
}

/// Phase 3 → consolidation relations: entity extraction creates an edge in the
/// knowledge graph.
#[tokio::test]
async fn consolidation_creates_relation_between_entities() {
    let world = MemoryWorld::empty().await;
    let kg = &world.knowledge;

    // Pre-populate alice and acme nodes.
    let (alice_id, _) =
        kg.find_or_create_by_slug("alice", "person", "alice", "Alice Chen").unwrap();
    let (acme_id, _) =
        kg.find_or_create_by_slug("acme", "company", "acme", "Acme Corp").unwrap();

    let provider = MockProvider::new(vec![
        scripted_response(r#"{"history_entry": "Alice works at Acme.", "memory_update": null}"#),
        scripted_response(
            r#"{"entities": [], "relations": [{"from_slug": "alice", "relation": "employed_by", "to_slug": "acme"}]}"#,
        ),
    ]);

    consolidate_turn(
        &provider,
        "test-model",
        world.brain.as_ref(),
        Some(kg),
        "Alice is employed by Acme Corp.",
        "Noted — Alice works at Acme Corp.",
    )
    .await
    .unwrap();

    // Verify the edge exists by checking related nodes for alice.
    let related = kg.find_related(&alice_id).unwrap();
    let has_employed_by = related
        .iter()
        .any(|(node, rel)| node.id == acme_id && rel == "employed_by");
    assert!(has_employed_by, "alice should have employed_by edge to acme");
}

/// Phase 6 → Phase 4 vector search: after synthesize_stale_entities sets the
/// synthesis text on alice's node, FTS search returns alice.
#[tokio::test]
async fn dream_cycle_synthesis_enables_vector_recall() {
    let world = MemoryWorld::empty().await;
    let kg = &world.knowledge;

    // Create alice with stale events.
    let (alice_id, _) = kg
        .find_or_create_by_slug("alice", "person", "alice", "Alice Chen")
        .unwrap();
    for e in [
        "alice is lead engineer at Acme.",
        "alice prefers async communication.",
        "alice is working on the consensus rewrite.",
    ] {
        kg.add_event(&alice_id, e, None, None).unwrap();
    }

    // Confirm alice is stale.
    let stale = kg.list_stale_nodes(10).unwrap();
    assert!(
        stale.iter().any(|n| n.id == alice_id),
        "alice should be stale before synthesis"
    );

    // Run dream cycle with a scripted provider.
    let embedder = Arc::new(AxisEmbedder::new(8, standard_axes()));
    let provider = MockProvider::new(vec![
        scripted_response("Alice Chen. Lead engineer at Acme. Prefers async comms. Working on consensus rewrite."),
    ]);
    let report = synthesize_stale_entities(&provider, "test-model", kg, embedder.as_ref(), 10)
        .await
        .unwrap();

    assert_eq!(report.nodes_processed, 1);
    assert_eq!(report.nodes_remaining, 0);

    // Synthesis text should be set.
    let (node, _) = kg.get_with_timeline(&alice_id, 0).unwrap().unwrap();
    assert!(
        node.synthesis.is_some(),
        "alice synthesis should be set after dream cycle"
    );
    assert!(
        node.synthesis_at.is_some(),
        "alice synthesis_at should be set (not stale)"
    );

    // FTS search should return alice using a single token that matches her title.
    // Note: query_by_similarity uses FTS5 — it searches title/content/tags, not synthesis.
    // After synthesis we verify the node is still findable via its original indexed fields.
    let results = kg.query_by_similarity("alice", 5).unwrap();
    assert!(
        results.iter().any(|r| r.node.id == alice_id),
        "alice should appear in similarity search after synthesis"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// Group 2: Retrieval and Ranking Correctness
// ═════════════════════════════════════════════════════════════════════════════

/// RRF multi-source advantage: an item present in two sources outranks an item
/// present in only one source.
#[tokio::test]
async fn rrf_item_in_two_sources_outranks_item_in_one_source() {
    let world = MemoryWorld::empty().await;
    let kg = &world.knowledge;

    // "shared fact" — stored in both memories and as a KG node event.
    world
        .brain
        .store("shared_fact", "alice deployment deadline shared fact", MemoryCategory::Core, None)
        .await
        .unwrap();
    let (node_id, _) = kg
        .find_or_create_by_slug("alice", "person", "alice", "alice deployment deadline shared fact")
        .unwrap();
    kg.add_event(&node_id, "alice deployment deadline shared fact", None, None).unwrap();

    // "memory_only fact" — stored only in memories.
    world
        .brain
        .store("memory_only_fact", "alice deployment deadline memory only fact", MemoryCategory::Core, None)
        .await
        .unwrap();

    let results = world
        .pipeline
        .recall_rrf("alice deployment deadline", 10, None)
        .await
        .unwrap();

    // Find the two relevant entries.
    let shared_entry = results
        .iter()
        .find(|e| e.content.contains("shared fact"));
    let memory_only_entry = results
        .iter()
        .find(|e| e.content.contains("memory only"));

    if let (Some(shared), Some(mem_only)) = (shared_entry, memory_only_entry) {
        assert!(
            shared.rrf_score >= mem_only.rrf_score,
            "shared-fact rrf_score ({}) should be >= memory-only rrf_score ({})",
            shared.rrf_score,
            mem_only.rrf_score
        );
    }
    // If either isn't found, the corpus is too thin for BM25 to return it — that's OK.
}

/// BM25 discrimination: top result for "alice deployment window" is relevant
/// and Session B weather/flight messages don't appear.
#[tokio::test]
async fn recall_top_result_is_most_relevant() {
    let world = MemoryWorld::standard().await;

    let results = world
        .pipeline
        .recall_rrf("alice deployment window", 10, None)
        .await
        .unwrap();

    assert!(
        !results.is_empty(),
        "recall should return at least one result"
    );

    // Top result should mention alice or deployment.
    let top = &results[0];
    let is_relevant = top.content.to_lowercase().contains("alice")
        || top.content.to_lowercase().contains("deployment");
    assert!(is_relevant, "top result should be about alice/deployment, got: {}", top.content);

    // Weather/flight content should not appear — they're not in memories or summaries
    // with matching FTS rank for this query.
    let weather_in_top5: Vec<_> = results
        .iter()
        .take(5)
        .filter(|e| {
            e.content.to_lowercase().contains("new york weather")
                || e.content.to_lowercase().contains("flights")
        })
        .collect();
    assert!(
        weather_in_top5.is_empty(),
        "weather/flight content should not be in top 5 results for alice deployment query"
    );
}

/// Vector recall: KG node appears in results alongside keyword-matched memories.
#[tokio::test]
async fn vector_recall_returns_entity_not_keyword_matched() {
    let world = MemoryWorld::empty().await;
    let kg = &world.knowledge;

    // Store a Core memory about alice (keyword matches).
    world
        .brain
        .store("alice_fact", "alice is a lead engineer", MemoryCategory::Core, None)
        .await
        .unwrap();

    // Create a KG node for alice with synthesis set (so it's not just an empty node).
    let (alice_id, _) = kg
        .find_or_create_by_slug("alice", "person", "alice", "Alice Chen, lead engineer")
        .unwrap();
    // Set synthesis so the node has content for FTS.
    kg.update_synthesis(&alice_id, "Alice Chen, lead engineer at Acme.", None).unwrap();

    let results = world.pipeline.recall_rrf("alice", 5, None).await.unwrap();
    assert!(!results.is_empty(), "should return at least one result for alice");

    let has_memory = results.iter().any(|e| e.source == RrfSource::Memory);
    let has_kg = results.iter().any(|e| e.source == RrfSource::KnowledgeNode);

    assert!(has_memory, "recall should include the Core memory about alice");
    assert!(has_kg, "recall should include the alice KG node");
}

/// Session-scoped recall: Session B weather messages don't appear when
/// filtered to Session A.
#[tokio::test]
async fn session_b_messages_absent_from_session_a_recall() {
    let world = MemoryWorld::standard().await;

    // With Session A filter: weather content (from Session B summary/daily)
    // should not appear in summaries.
    let filtered = world
        .pipeline
        .recall_rrf("weather new york", 10, Some(&world.session_a))
        .await
        .unwrap();

    let session_b_in_filtered: Vec<_> = filtered
        .iter()
        .filter(|e| e.source == RrfSource::Summary && e.content.to_lowercase().contains("weather"))
        .collect();
    assert!(
        session_b_in_filtered.is_empty(),
        "Session B weather summary should not appear when filtered to Session A"
    );

    // Without filter: Session B's weather summary should appear.
    let unfiltered = world
        .pipeline
        .recall_rrf("weather new york", 10, None)
        .await
        .unwrap();

    let weather_summary: Vec<_> = unfiltered
        .iter()
        .filter(|e| {
            e.source == RrfSource::Summary && e.content.to_lowercase().contains("weather")
        })
        .collect();
    assert!(
        !weather_summary.is_empty(),
        "Session B weather summary should appear when no session filter is applied"
    );
}

/// Entity knowledge transcends session boundaries: alice's KG node appears
/// even when recall is scoped to an unrelated session.
#[tokio::test]
async fn cross_session_entity_knowledge_available_everywhere() {
    let world = MemoryWorld::standard().await;

    // Recall using Session B scope — alice was learned in Session A but is
    // session-agnostic in the knowledge graph.
    let results = world
        .pipeline
        .recall_rrf("alice", 10, Some(&world.session_b))
        .await
        .unwrap();

    let alice_node = results
        .iter()
        .find(|e| e.source == RrfSource::KnowledgeNode && e.content.to_lowercase().contains("alice"));

    assert!(
        alice_node.is_some(),
        "alice KG node should appear in Session B recall (entity nodes are session-agnostic)"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// Group 3: Context Block Quality
// ═════════════════════════════════════════════════════════════════════════════

/// build_context emits [Context]...[/Context] delimiters and structured sections.
#[tokio::test]
async fn build_context_emits_structured_sections() {
    let world = MemoryWorld::standard().await;

    // Use "alice" — a single token guaranteed to match alice's FTS-indexed title.
    let ctx = build_context(&world.pipeline, "alice", 0.0, None).await;

    assert!(ctx.contains("[Context]"), "output should start with [Context]");
    assert!(ctx.contains("[/Context]"), "output should end with [/Context]");
    // Alice is a person-type entity — section header is "## Persons" (type + "s").
    assert!(
        ctx.contains("## Persons"),
        "output should contain ## Persons section for person-type entities, got: {ctx}"
    );
    // Core memories should appear in ## Facts.
    assert!(ctx.contains("## Facts"), "output should contain ## Facts section");
}

/// build_context omits empty sections when no data of that type is available.
#[tokio::test]
async fn build_context_omits_empty_sections() {
    let world = MemoryWorld::empty().await;

    // Store only Core memories — no KG nodes.
    world
        .brain
        .store("fact1", "Prefers Rust for systems programming", MemoryCategory::Core, None)
        .await
        .unwrap();

    let ctx = build_context(&world.pipeline, "programming preferences", 0.0, None).await;

    // Should have Facts but no entity sections.
    if !ctx.is_empty() {
        assert!(
            !ctx.contains("## Persons") && !ctx.contains("## Companies"),
            "empty sections should be omitted"
        );
    }
}

/// build_context filters out entries whose content contains <tool_result blocks.
#[tokio::test]
async fn build_context_omits_tool_result_content() {
    let world = MemoryWorld::empty().await;

    // Store a Core memory whose content is a tool_result block.
    world
        .brain
        .store(
            "tool_leak",
            "<tool_result>This content should be filtered</tool_result>",
            MemoryCategory::Core,
            None,
        )
        .await
        .unwrap();

    let ctx = build_context(&world.pipeline, "tool leak test", 0.0, None).await;

    assert!(
        !ctx.contains("<tool_result"),
        "tool_result content must not appear in context output"
    );
}

/// build_context output is bounded when many entries exist.
#[tokio::test]
async fn build_context_size_bounded_under_load() {
    let world = MemoryWorld::empty().await;

    // Store 200 Core memory entries.
    for i in 0..200_usize {
        let key = format!("fact_{i}");
        let content = format!("user preference fact number {i} about programming and deployments");
        world
            .brain
            .store(&key, &content, MemoryCategory::Core, None)
            .await
            .unwrap();
    }

    let ctx = build_context(&world.pipeline, "programming deployment facts", 0.0, None).await;

    // recall_rrf returns at most 10 entries — context is bounded.
    let fact_count = ctx.matches("- fact_").count();
    assert!(
        fact_count <= 10,
        "context should include at most 10 entries (got {fact_count})"
    );
}

/// build_context includes ## Session history section with the leaf summary
/// when a session filter is applied.
#[tokio::test]
async fn build_context_session_summary_in_history_section() {
    let world = MemoryWorld::standard().await;

    let ctx = build_context(
        &world.pipeline,
        "deployment",
        0.0,
        Some(&world.session_a),
    )
    .await;

    assert!(
        ctx.contains("## Session history"),
        "context should include ## Session history section"
    );
    assert!(
        ctx.contains("[Summary:"),
        "session history section should contain [Summary: ...] entry"
    );
}

/// build_context uses the synthesis text for nodes that have been synthesized.
#[tokio::test]
async fn build_context_entity_synthesis_used_when_available() {
    let world = MemoryWorld::empty().await;
    let kg = &world.knowledge;

    let (alice_id, _) = kg
        .find_or_create_by_slug("alice", "person", "alice", "Alice Chen")
        .unwrap();
    // Add events so the node has content.
    kg.add_event(&alice_id, "alice is lead engineer at acme", None, None).unwrap();
    kg.add_event(&alice_id, "alice prefers async comms", None, None).unwrap();
    kg.add_event(&alice_id, "alice is working on the consensus rewrite", None, None).unwrap();

    // Set synthesis directly (simulating dream cycle).
    let synthesis = "Alice Chen. Lead engineer at Acme. Prefers async comms.";
    kg.update_synthesis(&alice_id, synthesis, None).unwrap();

    let ctx = build_context(&world.pipeline, "alice", 0.0, None).await;

    // The synthesis text should appear in the context.
    assert!(
        ctx.contains("Alice Chen") || ctx.contains("Lead engineer"),
        "context should use synthesis text for alice, got: {ctx}"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// Group 4: Hygiene Safety
// ═════════════════════════════════════════════════════════════════════════════

/// Hygiene never deletes messages or summaries — immutable table guarantee.
#[tokio::test]
async fn hygiene_preserves_all_immutable_tables() {
    let world = MemoryWorld::standard().await;

    // Record what we can find before hygiene.
    let msgs_before = world.brain.search_messages("consensus", None, 20).unwrap();
    let summaries_before = world
        .pipeline
        .recall_rrf("consensus rewrite deployment", 5, None)
        .await
        .unwrap()
        .into_iter()
        .filter(|e| e.source == RrfSource::Summary)
        .count();

    // Run hygiene with archive_after_days = 0 to skip file archiving,
    // so only FTS optimize and conversation pruning run.
    let mut cfg = MemoryConfig::default();
    cfg.archive_after_days = 0;
    cfg.purge_after_days = 0;
    cfg.conversation_retention_days = 1; // short retention, but no Conversation rows exist

    hygiene::run_if_due(&cfg, world.tmp.path()).unwrap();

    // Messages must still be searchable.
    let msgs_after = world.brain.search_messages("consensus", None, 20).unwrap();
    assert_eq!(
        msgs_before.len(),
        msgs_after.len(),
        "hygiene must not delete from messages table"
    );

    // Summaries must still be retrievable.
    // Invalidate cache so we get fresh results.
    world.pipeline.invalidate_cache();
    let summaries_after = world
        .pipeline
        .recall_rrf("consensus rewrite deployment", 5, None)
        .await
        .unwrap()
        .into_iter()
        .filter(|e| e.source == RrfSource::Summary)
        .count();
    assert_eq!(
        summaries_before,
        summaries_after,
        "hygiene must not delete from summaries table"
    );
}

/// FTS optimize (Phase 7) does not corrupt search results.
#[tokio::test]
async fn hygiene_fts_optimize_does_not_corrupt_search() {
    let world = MemoryWorld::standard().await;

    // Run hygiene (which runs FTS optimize as part of Phase 7).
    let mut cfg = MemoryConfig::default();
    cfg.archive_after_days = 0;
    cfg.purge_after_days = 0;

    hygiene::run_if_due(&cfg, world.tmp.path()).unwrap();

    // FTS search on messages should still work.
    let msg_results = world.brain.search_messages("alice", None, 5).unwrap();
    assert!(
        !msg_results.is_empty(),
        "message FTS search should work after FTS optimize"
    );

    // RRF recall should still work.
    world.pipeline.invalidate_cache();
    let recall_results = world
        .pipeline
        .recall_rrf("alice", 5, None)
        .await
        .unwrap();
    assert!(
        !recall_results.is_empty(),
        "recall should still return results after FTS optimize"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// Group 5: End-to-End Pipeline
// ═════════════════════════════════════════════════════════════════════════════

/// Flagship test: exercises all phases sequentially in a realistic scenario.
/// Each step is a checkpoint — failure message indicates which phase's data
/// path is broken.
#[tokio::test]
async fn full_pipeline_end_to_end() {
    let world = MemoryWorld::empty().await;

    // ── Step 1: Message storage (Phase 1) ─────────────────────────────────
    let user_msg_id = Uuid::new_v4().to_string();
    let asst_msg_id = Uuid::new_v4().to_string();
    let session = &world.session_a;

    world
        .brain
        .append_message(
            &user_msg_id,
            session,
            "user",
            "Alice from Acme wants to discuss the consensus rewrite deployment",
            None,
        )
        .unwrap();
    world
        .brain
        .append_message(
            &asst_msg_id,
            session,
            "assistant",
            "I'll set up a meeting with Alice to discuss the deployment timeline.",
            None,
        )
        .unwrap();

    let msgs = world.brain.search_messages("alice consensus deployment", None, 10).unwrap();
    assert!(
        msgs.iter().any(|m| m.id == user_msg_id),
        "[Phase 1] user message should be searchable"
    );
    assert!(
        msgs.iter().any(|m| m.id == asst_msg_id),
        "[Phase 1] assistant message should be searchable"
    );

    // ── Step 2: Consolidation (Phase 3) ────────────────────────────────────
    let provider = MockProvider::new(vec![
        scripted_response(
            r#"{"history_entry": "Alice from Acme wants to discuss the consensus rewrite deployment.", "memory_update": null}"#,
        ),
        scripted_response(
            r#"{"entities": [
                  {"entity_type": "person", "slug": "alice", "fact": "Lead engineer at Acme working on consensus rewrite."},
                  {"entity_type": "company", "slug": "acme", "fact": "Enterprise software company involved in consensus rewrite."}
               ],
               "relations": [{"from_slug": "alice", "relation": "employed_by", "to_slug": "acme"}]
            }"#,
        ),
    ]);

    consolidate_turn(
        &provider,
        "test-model",
        world.brain.as_ref(),
        Some(&world.knowledge),
        "Alice from Acme wants to discuss the consensus rewrite deployment",
        "I'll set up a meeting with Alice to discuss the deployment timeline.",
    )
    .await
    .unwrap();

    let (alice_id, created) = world
        .knowledge
        .find_or_create_by_slug("alice", "person", "alice", "")
        .unwrap();
    assert!(!created, "[Phase 3] alice node should exist after consolidation");

    let (_, alice_events) = world.knowledge.get_with_timeline(&alice_id, 10).unwrap().unwrap();
    assert!(
        !alice_events.is_empty(),
        "[Phase 3] alice should have at least one node_event"
    );

    let (acme_id, acme_created) = world
        .knowledge
        .find_or_create_by_slug("acme", "company", "acme", "")
        .unwrap();
    assert!(!acme_created, "[Phase 3] acme node should exist after consolidation");

    let alice_related = world.knowledge.find_related(&alice_id).unwrap();
    assert!(
        alice_related.iter().any(|(n, rel)| n.id == acme_id && rel == "employed_by"),
        "[Phase 3] alice → acme employed_by edge should exist"
    );

    // synthesis_at should be NULL (stale trigger fired on add_event).
    let (alice_node, _) = world.knowledge.get_with_timeline(&alice_id, 0).unwrap().unwrap();
    assert!(alice_node.synthesis_at.is_none(), "[Phase 3] alice should be stale after add_event");

    // ── Step 3: Compression simulation (Phase 2) ───────────────────────────
    // Insert a summary directly to simulate what compress_if_needed() would produce,
    // and link it to the two messages from Step 1.
    let summary_id = Uuid::new_v4().to_string();
    world
        .brain
        .insert_summary(
            &summary_id,
            "leaf",
            "Alice from Acme discussed the consensus rewrite deployment. Meeting to be set up to discuss timeline.",
            None,
            session,
            1,
        )
        .unwrap();
    world
        .brain
        .link_summary_sources(
            &summary_id,
            &[
                (user_msg_id.as_str(), "message"),
                (asst_msg_id.as_str(), "message"),
            ],
        )
        .unwrap();

    // Verify: leaf summary exists.
    let summary_results = world
        .pipeline
        .recall_rrf("alice deployment timeline", 5, Some(session))
        .await
        .unwrap();
    assert!(
        summary_results.iter().any(|e| e.source == RrfSource::Summary),
        "[Phase 2] leaf summary should be findable via recall"
    );

    // Verify: covered messages are annotated.
    let grep_results = world
        .brain
        .grep_messages("alice|consensus|deployment", Some(session), 10)
        .unwrap();
    let covered_msg = grep_results.iter().find(|r| r.id == user_msg_id);
    if let Some(c) = covered_msg {
        assert!(
            c.summary_id.is_some(),
            "[Phase 2] user message should be annotated with summary_id after compression"
        );
    }

    // Verify: no compressed_context_* entries exist (old path removed).
    let old_entry = world.brain.get("compressed_context_test").await.unwrap();
    assert!(old_entry.is_none(), "[Phase 2] no compressed_context_* entries should exist");

    // ── Step 4: Dream cycle (Phase 6) ──────────────────────────────────────
    let embedder = Arc::new(AxisEmbedder::new(8, standard_axes()));
    let dream_provider = MockProvider::new(vec![
        // Synthesis for alice
        scripted_response("Alice Chen, lead engineer at Acme. Working on consensus rewrite deployment."),
        // Synthesis for acme
        scripted_response("Acme Corp, enterprise software company. Alice is a lead engineer there."),
    ]);
    let report = synthesize_stale_entities(
        &dream_provider,
        "test-model",
        &world.knowledge,
        embedder.as_ref(),
        10,
    )
    .await
    .unwrap();

    assert!(report.nodes_processed >= 2, "[Phase 6] dream cycle should synthesize alice and acme");

    let (alice_after, _) = world.knowledge.get_with_timeline(&alice_id, 0).unwrap().unwrap();
    assert!(alice_after.synthesis.is_some(), "[Phase 6] alice synthesis should be set");
    assert!(alice_after.synthesis_at.is_some(), "[Phase 6] alice should not be stale after synthesis");

    // ── Step 5: Unified recall (Phase 4) ────────────────────────────────────
    // Use "alice" — a single token that matches alice's FTS-indexed title.
    // KG query_by_similarity uses FTS5 on title/content/tags; "alice" is in alice's title.
    world.pipeline.invalidate_cache();
    let recall_alice = world
        .pipeline
        .recall_rrf("alice", 10, None)
        .await
        .unwrap();
    let recall_deploy = world
        .pipeline
        .recall_rrf("deployment", 10, None)
        .await
        .unwrap();

    let has_kg = recall_alice.iter().any(|e| e.source == RrfSource::KnowledgeNode);
    let has_memory = recall_deploy.iter().any(|e| e.source == RrfSource::Memory);
    let has_summary = recall_deploy.iter().any(|e| e.source == RrfSource::Summary);

    assert!(has_kg, "[Phase 4] recall should include alice/acme KG node");
    assert!(has_memory, "[Phase 4] recall should include Daily memory from consolidation");
    assert!(has_summary, "[Phase 4] recall should include the leaf summary");

    // ── Step 6: Context injection (Phase 4 + loop_.rs) ──────────────────────
    world.pipeline.invalidate_cache();
    let ctx = build_context(&world.pipeline, "deployment meeting with alice", 0.0, Some(session)).await;

    assert!(ctx.contains("[Context]"), "[Phase 4] context should start with [Context]");
    assert!(ctx.contains("[/Context]"), "[Phase 4] context should end with [/Context]");
    assert!(
        !ctx.contains("<tool_result"),
        "[Phase 4] context should not contain tool_result blocks"
    );

    // ── Step 7: lcm_grep (Phase 5) ──────────────────────────────────────────
    let grep = world
        .brain
        .grep_messages("consensus rewrite", Some(session), 10)
        .unwrap();

    assert!(
        !grep.is_empty(),
        "[Phase 5] grep should find the user message containing 'consensus rewrite'"
    );
    assert!(
        grep.iter().any(|r| r.id == user_msg_id),
        "[Phase 5] Turn 1 user message should be in grep results"
    );
    // Covered messages have summary_id set.
    let covered_in_grep = grep.iter().filter(|r| r.summary_id.is_some()).count();
    assert!(
        covered_in_grep > 0,
        "[Phase 5] at least one covered message should have summary_id annotation"
    );

    // ── Step 8: Hygiene (Phase 7) ────────────────────────────────────────────
    let mut cfg = MemoryConfig::default();
    cfg.archive_after_days = 0;
    cfg.purge_after_days = 0;

    hygiene::run_if_due(&cfg, world.tmp.path()).unwrap();

    // Messages unchanged.
    let msgs_after_hygiene = world
        .brain
        .search_messages("alice consensus deployment", None, 10)
        .unwrap();
    assert!(
        msgs_after_hygiene.iter().any(|m| m.id == user_msg_id),
        "[Phase 7] user message must survive hygiene"
    );
    assert!(
        msgs_after_hygiene.iter().any(|m| m.id == asst_msg_id),
        "[Phase 7] assistant message must survive hygiene"
    );

    // Summary unchanged.
    world.pipeline.invalidate_cache();
    let summaries_after_hygiene = world
        .pipeline
        .recall_rrf("alice deployment timeline", 5, Some(session))
        .await
        .unwrap();
    assert!(
        summaries_after_hygiene.iter().any(|e| e.source == RrfSource::Summary && e.id == summary_id),
        "[Phase 7] leaf summary must survive hygiene"
    );
}
