//! Multi-stage retrieval pipeline with Reciprocal Rank Fusion.
//!
//! Wraps a `Memory` trait object with staged retrieval:
//! - **Stage 1 (Hot cache):** In-memory LRU of recent recall results.
//! - **Stage 2 (FTS):** FTS5 keyword search with optional early-return.
//! - **Stage 3 (Vector):** Vector similarity search + hybrid merge.
//!
//! Phase 4 extends this with multi-source RRF:
//! - Memory store (brain.db)
//! - Summary DAG (summaries table in brain.db)
//! - Knowledge graph (knowledge.db)
//!
//! Configurable via `[memory]` settings: `retrieval_stages`, `fts_early_return_score`.

use super::traits::{Memory, MemoryEntry};
use crate::knowledge_graph::KnowledgeGraph;
use crate::sqlite::SqliteMemory;
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

// ── RRF types ────────────────────────────────────────────────────

/// Which storage layer an `RrfEntry` came from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RrfSource {
    Memory,
    Summary,
    KnowledgeNode,
}

/// A single result entry in the unified RRF result set.
#[derive(Debug, Clone)]
pub struct RrfEntry {
    /// Stable identifier within its source (memory id, summary id, or node id).
    pub id: String,
    pub content: String,
    pub source: RrfSource,
    /// Score from the originating search (BM25, cosine, etc.). `None` if unavailable.
    pub original_score: Option<f64>,
    /// RRF combined score, filled by `rrf_merge()`.
    pub rrf_score: f64,
    /// Node type string when `source == KnowledgeNode` (e.g. "person", "project").
    pub node_type: Option<String>,
    /// Memory key when `source == Memory`.
    pub key: Option<String>,
    pub created_at: String,
}

/// Reciprocal Rank Fusion across multiple ranked result lists.
///
/// `ranked_lists`: each inner `Vec<RrfEntry>` is already sorted best-first.
/// Items are identified by their `id` field; duplicates (same id across sources) accumulate score.
/// `k = 60` is the standard constant from the original paper (Cormack et al. 2009).
pub fn rrf_merge(ranked_lists: Vec<Vec<RrfEntry>>, limit: usize, k: usize) -> Vec<RrfEntry> {
    if ranked_lists.is_empty() {
        return Vec::new();
    }

    // Accumulate RRF scores: score(d) = Σ_i 1/(k + rank_i(d))  (0-indexed rank)
    let mut score_map: HashMap<String, f64> = HashMap::new();
    // Keep one representative entry per id (from first occurrence).
    let mut entry_map: HashMap<String, RrfEntry> = HashMap::new();

    for list in ranked_lists {
        for (rank, mut entry) in list.into_iter().enumerate() {
            let contribution = 1.0 / (k as f64 + rank as f64 + 1.0);
            *score_map.entry(entry.id.clone()).or_insert(0.0) += contribution;
            entry_map.entry(entry.id.clone()).or_insert_with(|| {
                entry.rrf_score = 0.0; // will be filled below
                entry
            });
        }
    }

    // Apply accumulated scores.
    let mut results: Vec<RrfEntry> = entry_map
        .into_values()
        .map(|mut e| {
            e.rrf_score = score_map[&e.id];
            e
        })
        .collect();

    // Sort descending by RRF score.
    results.sort_by(|a, b| b.rrf_score.partial_cmp(&a.rrf_score).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(limit);
    results
}

/// Compute a source presence bitmask for cache key differentiation.
///
/// bit 0 = memory (always 1)
/// bit 1 = sqlite summaries
/// bit 2 = knowledge graph
pub fn source_fingerprint(sqlite: bool, knowledge: bool) -> u8 {
    1 | ((sqlite as u8) << 1) | ((knowledge as u8) << 2)
}

// ── Pipeline ─────────────────────────────────────────────────────

/// A cached recall result.
struct CachedResult {
    entries: Vec<MemoryEntry>,
    created_at: Instant,
}

/// A cached RRF recall result.
struct CachedRrfResult {
    entries: Vec<RrfEntry>,
    created_at: Instant,
}

/// Multi-stage retrieval pipeline configuration.
#[derive(Debug, Clone)]
pub struct RetrievalConfig {
    /// Ordered list of stages: "cache", "fts", "vector".
    pub stages: Vec<String>,
    /// FTS score above which to early-return without vector stage.
    pub fts_early_return_score: f64,
    /// Max entries in the hot cache.
    pub cache_max_entries: usize,
    /// TTL for cached results.
    pub cache_ttl: Duration,
}

impl Default for RetrievalConfig {
    fn default() -> Self {
        Self {
            stages: vec!["cache".into(), "fts".into(), "vector".into()],
            fts_early_return_score: 0.85,
            cache_max_entries: 256,
            cache_ttl: Duration::from_secs(300),
        }
    }
}

/// Multi-stage retrieval pipeline wrapping a `Memory` backend.
///
/// Optionally extended with `SqliteMemory` (for summary search) and
/// `KnowledgeGraph` (for entity/pattern search), enabling three-source
/// RRF recall via `recall_rrf()`.
pub struct RetrievalPipeline {
    memory: Arc<dyn Memory>,
    sqlite: Option<Arc<SqliteMemory>>,
    knowledge: Option<Arc<KnowledgeGraph>>,
    /// Source presence bitmask — embedded in cache keys to prevent stale cross-source hits.
    sources_fp: u8,
    config: RetrievalConfig,
    hot_cache: Mutex<HashMap<String, CachedResult>>,
    rrf_cache: Mutex<HashMap<String, CachedRrfResult>>,
}

impl RetrievalPipeline {
    pub fn new(memory: Arc<dyn Memory>, config: RetrievalConfig) -> Self {
        Self {
            memory,
            sqlite: None,
            knowledge: None,
            sources_fp: source_fingerprint(false, false),
            config,
            hot_cache: Mutex::new(HashMap::new()),
            rrf_cache: Mutex::new(HashMap::new()),
        }
    }

    /// Wire up the SQLite memory backend for summary search.
    pub fn with_sqlite(mut self, sqlite: Arc<SqliteMemory>) -> Self {
        self.sqlite = Some(sqlite);
        self.sources_fp = source_fingerprint(true, self.knowledge.is_some());
        self
    }

    /// Wire up the knowledge graph for entity/node search.
    pub fn with_knowledge(mut self, knowledge: Arc<KnowledgeGraph>) -> Self {
        self.knowledge = Some(knowledge);
        self.sources_fp = source_fingerprint(self.sqlite.is_some(), true);
        self
    }

    /// Build a cache key from query parameters, including source fingerprint.
    fn cache_key(query: &str, limit: usize, session_id: Option<&str>, namespace: Option<&str>, fp: u8) -> String {
        format!(
            "{}:{}:{}:{}:{}",
            query,
            limit,
            session_id.unwrap_or(""),
            namespace.unwrap_or(""),
            fp,
        )
    }

    /// Check the hot cache for a previous result.
    fn check_cache(&self, key: &str) -> Option<Vec<MemoryEntry>> {
        let cache = self.hot_cache.lock();
        if let Some(cached) = cache.get(key)
            && cached.created_at.elapsed() < self.config.cache_ttl
        {
            return Some(cached.entries.clone());
        }
        None
    }

    /// Store a result in the hot cache with LRU eviction.
    fn store_in_cache(&self, key: String, entries: Vec<MemoryEntry>) {
        let mut cache = self.hot_cache.lock();

        // LRU eviction: remove oldest entries if at capacity
        if cache.len() >= self.config.cache_max_entries {
            let oldest_key = cache
                .iter()
                .min_by_key(|(_, v)| v.created_at)
                .map(|(k, _)| k.clone());
            if let Some(k) = oldest_key {
                cache.remove(&k);
            }
        }

        cache.insert(
            key,
            CachedResult {
                entries,
                created_at: Instant::now(),
            },
        );
    }

    /// Check the RRF cache.
    fn check_rrf_cache(&self, key: &str) -> Option<Vec<RrfEntry>> {
        let cache = self.rrf_cache.lock();
        if let Some(cached) = cache.get(key)
            && cached.created_at.elapsed() < self.config.cache_ttl
        {
            return Some(cached.entries.clone());
        }
        None
    }

    /// Store RRF results in cache.
    fn store_rrf_cache(&self, key: String, entries: Vec<RrfEntry>) {
        let mut cache = self.rrf_cache.lock();
        if cache.len() >= self.config.cache_max_entries {
            let oldest_key = cache
                .iter()
                .min_by_key(|(_, v)| v.created_at)
                .map(|(k, _)| k.clone());
            if let Some(k) = oldest_key {
                cache.remove(&k);
            }
        }
        cache.insert(key, CachedRrfResult { entries, created_at: Instant::now() });
    }

    /// Execute the multi-stage retrieval pipeline.
    pub async fn recall(
        &self,
        query: &str,
        limit: usize,
        session_id: Option<&str>,
        namespace: Option<&str>,
        since: Option<&str>,
        until: Option<&str>,
    ) -> anyhow::Result<Vec<MemoryEntry>> {
        let ck = Self::cache_key(query, limit, session_id, namespace, self.sources_fp);

        for stage in &self.config.stages {
            match stage.as_str() {
                "cache" => {
                    if let Some(cached) = self.check_cache(&ck) {
                        tracing::debug!("retrieval pipeline: cache hit for '{query}'");
                        return Ok(cached);
                    }
                }
                "fts" | "vector" => {
                    // Both FTS and vector are handled by the backend's recall method
                    // which already does hybrid merge. We delegate to it.
                    let results = if let Some(ns) = namespace {
                        self.memory
                            .recall_namespaced(ns, query, limit, session_id, since, until)
                            .await?
                    } else {
                        self.memory
                            .recall(query, limit, session_id, since, until)
                            .await?
                    };

                    if !results.is_empty() {
                        // Check for FTS early-return: if top score exceeds threshold
                        // and we're in the FTS stage, we can skip further stages
                        if stage == "fts"
                            && let Some(top_score) = results.first().and_then(|e| e.score)
                            && top_score >= self.config.fts_early_return_score
                        {
                            tracing::debug!(
                                "retrieval pipeline: FTS early return (score={top_score:.3})"
                            );
                            self.store_in_cache(ck, results.clone());
                            return Ok(results);
                        }

                        self.store_in_cache(ck, results.clone());
                        return Ok(results);
                    }
                }
                other => {
                    tracing::warn!("retrieval pipeline: unknown stage '{other}', skipping");
                }
            }
        }

        // No results from any stage
        Ok(Vec::new())
    }

    /// Unified multi-source recall with Reciprocal Rank Fusion.
    ///
    /// Queries all configured sources (memory, summaries, knowledge graph) and
    /// merges via RRF. Falls back to single-source when only `memory` is configured.
    ///
    /// Results are cached with the source fingerprint included in the key.
    pub async fn recall_rrf(
        &self,
        query: &str,
        limit: usize,
        session_id: Option<&str>,
    ) -> anyhow::Result<Vec<RrfEntry>> {
        let ck = Self::cache_key(query, limit, session_id, None, self.sources_fp);
        if let Some(cached) = self.check_rrf_cache(&ck) {
            return Ok(cached);
        }

        // Source 1: memory store
        let mem_entries: Vec<RrfEntry> = self
            .memory
            .recall(query, limit, session_id, None, None)
            .await
            .unwrap_or_default()
            .into_iter()
            .map(|e| RrfEntry {
                id: e.id.clone(),
                content: e.content.clone(),
                source: RrfSource::Memory,
                original_score: e.score,
                rrf_score: 0.0,
                node_type: None,
                key: Some(e.key.clone()),
                created_at: e.timestamp.clone(),
            })
            .collect();

        // Source 2: summaries (optional)
        let summary_entries: Vec<RrfEntry> = if let Some(ref sq) = self.sqlite {
            sq.search_summaries(query, limit, session_id)
                .unwrap_or_default()
                .into_iter()
                .map(|s| RrfEntry {
                    id: s.id.clone(),
                    content: s.content.clone(),
                    source: RrfSource::Summary,
                    original_score: s.score,
                    rrf_score: 0.0,
                    node_type: None,
                    key: None,
                    created_at: s.created_at.clone(),
                })
                .collect()
        } else {
            Vec::new()
        };

        // Source 3: knowledge graph (optional, sync call wrapped)
        let kg_entries: Vec<RrfEntry> = if let Some(ref kg) = self.knowledge {
            kg.query_by_similarity(query, limit)
                .unwrap_or_default()
                .into_iter()
                .map(|r| RrfEntry {
                    id: r.node.id.clone(),
                    content: r.node.synthesis.clone().unwrap_or_else(|| r.node.content.clone()),
                    source: RrfSource::KnowledgeNode,
                    original_score: Some(r.score),
                    rrf_score: 0.0,
                    node_type: Some(r.node.node_type.clone()),
                    key: None,
                    created_at: r.node.updated_at.to_rfc3339(),
                })
                .collect()
        } else {
            Vec::new()
        };

        // Collect non-empty source lists for RRF
        let mut ranked_lists = Vec::new();
        if !mem_entries.is_empty() {
            ranked_lists.push(mem_entries);
        }
        if !summary_entries.is_empty() {
            ranked_lists.push(summary_entries);
        }
        if !kg_entries.is_empty() {
            ranked_lists.push(kg_entries);
        }

        let merged = if ranked_lists.len() <= 1 {
            // Single source: skip RRF overhead, return entries with identity score.
            ranked_lists.into_iter().flatten().take(limit).map(|mut e| {
                e.rrf_score = e.original_score.unwrap_or(0.0);
                e
            }).collect()
        } else {
            rrf_merge(ranked_lists, limit, 60)
        };

        self.store_rrf_cache(ck, merged.clone());
        Ok(merged)
    }

    /// Invalidate the hot cache (e.g. after a store operation).
    pub fn invalidate_cache(&self) {
        self.hot_cache.lock().clear();
        self.rrf_cache.lock().clear();
    }

    /// Get the number of entries in the hot cache.
    pub fn cache_size(&self) -> usize {
        self.hot_cache.lock().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::none::NoneMemory;

    #[tokio::test]
    async fn pipeline_returns_empty_from_none_backend() {
        let memory = Arc::new(NoneMemory::new());
        let pipeline = RetrievalPipeline::new(memory, RetrievalConfig::default());

        let results = pipeline
            .recall("test", 10, None, None, None, None)
            .await
            .unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn pipeline_cache_invalidation() {
        let memory = Arc::new(NoneMemory::new());
        let pipeline = RetrievalPipeline::new(memory, RetrievalConfig::default());

        // Force a cache entry
        let ck = RetrievalPipeline::cache_key("test", 10, None, None, pipeline.sources_fp);
        pipeline.store_in_cache(ck, vec![]);

        assert_eq!(pipeline.cache_size(), 1);
        pipeline.invalidate_cache();
        assert_eq!(pipeline.cache_size(), 0);
    }

    #[test]
    fn cache_key_includes_all_params() {
        let fp = source_fingerprint(false, false);
        let k1 = RetrievalPipeline::cache_key("hello", 10, Some("sess-a"), Some("ns1"), fp);
        let k2 = RetrievalPipeline::cache_key("hello", 10, Some("sess-b"), Some("ns1"), fp);
        let k3 = RetrievalPipeline::cache_key("hello", 10, Some("sess-a"), Some("ns2"), fp);

        assert_ne!(k1, k2);
        assert_ne!(k1, k3);
    }

    #[test]
    fn cache_key_differs_by_source_fingerprint() {
        let k1 = RetrievalPipeline::cache_key("q", 5, None, None, source_fingerprint(false, false));
        let k2 = RetrievalPipeline::cache_key("q", 5, None, None, source_fingerprint(true, false));
        let k3 = RetrievalPipeline::cache_key("q", 5, None, None, source_fingerprint(true, true));
        assert_ne!(k1, k2);
        assert_ne!(k2, k3);
        assert_ne!(k1, k3);
    }

    #[tokio::test]
    async fn pipeline_caches_results() {
        let memory = Arc::new(NoneMemory::new());
        let config = RetrievalConfig {
            stages: vec!["cache".into()],
            ..Default::default()
        };
        let pipeline = RetrievalPipeline::new(memory, config);

        // First call: cache miss, no results
        let results = pipeline
            .recall("test", 10, None, None, None, None)
            .await
            .unwrap();
        assert!(results.is_empty());

        // Manually insert a cache entry
        let ck = RetrievalPipeline::cache_key("cached_query", 5, None, None, pipeline.sources_fp);
        let fake_entry = MemoryEntry {
            id: "1".into(),
            key: "k".into(),
            content: "cached content".into(),
            category: crate::traits::MemoryCategory::Core,
            timestamp: "now".into(),
            session_id: None,
            score: Some(0.9),
            namespace: "default".into(),
            importance: None,
            superseded_by: None,
        };
        pipeline.store_in_cache(ck, vec![fake_entry]);

        // Cache hit
        let results = pipeline
            .recall("cached_query", 5, None, None, None, None)
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].content, "cached content");
    }

    // ── RRF unit tests ────────────────────────────────────────────

    fn make_entries(ids: &[&str], source: RrfSource) -> Vec<RrfEntry> {
        ids.iter().map(|id| RrfEntry {
            id: id.to_string(),
            content: format!("content of {id}"),
            source: source.clone(),
            original_score: None,
            rrf_score: 0.0,
            node_type: None,
            key: None,
            created_at: "2026-01-01T00:00:00Z".into(),
        }).collect()
    }

    #[test]
    fn rrf_merge_empty_sources() {
        let result = rrf_merge(vec![], 10, 60);
        assert!(result.is_empty());
    }

    #[test]
    fn rrf_merge_single_source_preserves_order() {
        let list = make_entries(&["a", "b", "c", "d", "e"], RrfSource::Memory);
        let result = rrf_merge(vec![list], 3, 60);
        assert_eq!(result.len(), 3);
        // First-ranked item should have highest score
        assert!(result[0].rrf_score >= result[1].rrf_score);
        assert!(result[1].rrf_score >= result[2].rrf_score);
        assert_eq!(result[0].id, "a");
    }

    #[test]
    fn rrf_merge_cross_source_fusion() {
        // "shared" appears in both Memory and KnowledgeNode → higher score than "mem_only"
        let mem_list = make_entries(&["shared", "mem_only"], RrfSource::Memory);
        let kg_list = make_entries(&["shared", "kg_only"], RrfSource::KnowledgeNode);
        let result = rrf_merge(vec![mem_list, kg_list], 10, 60);

        let shared_score = result.iter().find(|e| e.id == "shared").map(|e| e.rrf_score).unwrap();
        let mem_only_score = result.iter().find(|e| e.id == "mem_only").map(|e| e.rrf_score).unwrap();
        assert!(shared_score > mem_only_score, "cross-source item should score higher");
    }

    #[test]
    fn rrf_merge_respects_limit() {
        let list1 = make_entries(&["a","b","c","d","e"], RrfSource::Memory);
        let list2 = make_entries(&["f","g","h","i","j"], RrfSource::Summary);
        let result = rrf_merge(vec![list1, list2], 3, 60);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn rrf_score_formula() {
        // Rank 0 in a single source with k=60: score = 1/(60+0+1) = 1/61
        let list = make_entries(&["x"], RrfSource::Memory);
        let result = rrf_merge(vec![list], 1, 60);
        let expected = 1.0 / 61.0;
        assert!((result[0].rrf_score - expected).abs() < 1e-9);
    }

    #[tokio::test]
    async fn recall_rrf_single_source_no_rrf_overhead() {
        let memory = Arc::new(NoneMemory::new());
        let pipeline = RetrievalPipeline::new(memory, RetrievalConfig::default());
        // NoneMemory returns empty — result should be empty, no panic
        let result = pipeline.recall_rrf("test query", 5, None).await.unwrap();
        assert!(result.is_empty());
    }

    #[tokio::test]
    async fn recall_rrf_cache_hit_on_second_call() {
        let memory = Arc::new(NoneMemory::new());
        let pipeline = RetrievalPipeline::new(memory, RetrievalConfig::default());

        // Pre-populate RRF cache manually
        let ck = RetrievalPipeline::cache_key("q", 5, None, None, pipeline.sources_fp);
        let fake = RrfEntry {
            id: "z".into(),
            content: "cached".into(),
            source: RrfSource::Memory,
            original_score: None,
            rrf_score: 0.5,
            node_type: None,
            key: Some("k".into()),
            created_at: "now".into(),
        };
        pipeline.store_rrf_cache(ck, vec![fake]);

        let result = pipeline.recall_rrf("q", 5, None).await.unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].content, "cached");
    }

    #[test]
    fn source_fingerprint_values() {
        assert_eq!(source_fingerprint(false, false), 0b001);
        assert_eq!(source_fingerprint(true,  false), 0b011);
        assert_eq!(source_fingerprint(false, true),  0b101);
        assert_eq!(source_fingerprint(true,  true),  0b111);
    }

    #[test]
    fn rrf_merge_all_same_id_accumulates() {
        // Same id at rank 0 in three sources — should accumulate 3x the single-source score.
        let l1 = make_entries(&["x"], RrfSource::Memory);
        let l2 = make_entries(&["x"], RrfSource::Summary);
        let l3 = make_entries(&["x"], RrfSource::KnowledgeNode);
        let result = rrf_merge(vec![l1, l2, l3], 5, 60);
        assert_eq!(result.len(), 1);
        let expected = 3.0 / 61.0;
        assert!((result[0].rrf_score - expected).abs() < 1e-9);
    }
}
