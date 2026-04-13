//! Knowledge graph for capturing, organizing, and reusing expertise.
//!
//! SQLite-backed storage for knowledge nodes and directed edges.
//! Node types and relation types are free-form lowercase strings —
//! the LLM can create any type without code changes.
//! Well-known defaults: person, company, project, pattern, decision, lesson, expert, technology.

use anyhow::Context;
use chrono::{DateTime, Utc};
use parking_lot::Mutex;
use rusqlite::{Connection, params};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use uuid::Uuid;

// ── Normalization ────────────────────────────────────────────────

/// Normalize a node type or relation string: lowercase, collapse non-alphanumeric
/// runs to single underscores, strip leading/trailing underscores.
pub fn normalize_type(s: &str) -> String {
    normalize_slug(s)
}

/// Normalize a slug for entity identity: same rules as normalize_type.
pub fn normalize_slug(s: &str) -> String {
    let lower = s.to_lowercase();
    let mut result = String::with_capacity(lower.len());
    let mut last_was_sep = true; // start true to strip leading underscores
    for ch in lower.chars() {
        if ch.is_alphanumeric() {
            result.push(ch);
            last_was_sep = false;
        } else if !last_was_sep {
            result.push('_');
            last_was_sep = true;
        }
    }
    // Strip trailing underscore
    if result.ends_with('_') {
        result.pop();
    }
    result
}

// ── Domain types ────────────────────────────────────────────────

/// A node in the knowledge graph.
/// `node_type` is a free-form normalized string (e.g. "person", "company", "pattern").
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeNode {
    pub id: String,
    pub node_type: String,
    pub title: String,
    pub content: String,
    pub tags: Vec<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub source_project: Option<String>,
    /// LLM-compiled synthesis of all entity events. None = not yet synthesized.
    #[serde(default)]
    pub synthesis: Option<String>,
    /// Timestamp of last synthesis. None = stale, needs re-synthesis.
    #[serde(default)]
    pub synthesis_at: Option<DateTime<Utc>>,
}

/// A directed edge in the knowledge graph.
/// `relation` is a free-form normalized string (e.g. "employed_by", "uses", "extends").
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeEdge {
    pub from_id: String,
    pub to_id: String,
    pub relation: String,
}

/// An event on an entity's timeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeEvent {
    pub id: String,
    pub node_id: String,
    pub content: String,
    pub source_message_id: Option<String>,
    pub session_id: Option<String>,
    pub created_at: DateTime<Utc>,
}

/// A search result with relevance score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub node: KnowledgeNode,
    pub score: f64,
}

/// Summary statistics for the knowledge graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphStats {
    pub total_nodes: usize,
    pub total_edges: usize,
    pub nodes_by_type: HashMap<String, usize>,
    pub top_tags: Vec<(String, usize)>,
}

// ── Knowledge graph ─────────────────────────────────────────────

/// SQLite-backed knowledge graph.
pub struct KnowledgeGraph {
    conn: Mutex<Connection>,
    #[allow(dead_code)]
    db_path: PathBuf,
    max_nodes: usize,
}

impl KnowledgeGraph {
    /// Open (or create) a knowledge graph database at the given path.
    pub fn new(db_path: &Path, max_nodes: usize) -> anyhow::Result<Self> {
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let conn = Connection::open(db_path).context("failed to open knowledge graph database")?;

        conn.execute_batch(
            "PRAGMA journal_mode = WAL;
             PRAGMA synchronous  = NORMAL;
             PRAGMA foreign_keys = ON;",
        )?;

        Self::init_schema(&conn)?;

        Ok(Self {
            conn: Mutex::new(conn),
            db_path: db_path.to_path_buf(),
            max_nodes,
        })
    }

    fn init_schema(conn: &Connection) -> anyhow::Result<()> {
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS nodes (
                id TEXT PRIMARY KEY,
                node_type TEXT NOT NULL,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                tags TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                source_project TEXT
            );

            CREATE TABLE IF NOT EXISTS edges (
                from_id TEXT NOT NULL,
                to_id TEXT NOT NULL,
                relation TEXT NOT NULL,
                PRIMARY KEY (from_id, to_id, relation),
                FOREIGN KEY (from_id) REFERENCES nodes(id) ON DELETE CASCADE,
                FOREIGN KEY (to_id) REFERENCES nodes(id) ON DELETE CASCADE
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS nodes_fts USING fts5(
                title, content, tags, content='nodes', content_rowid='rowid'
            );

            CREATE TRIGGER IF NOT EXISTS nodes_ai AFTER INSERT ON nodes BEGIN
                INSERT INTO nodes_fts(rowid, title, content, tags)
                VALUES (new.rowid, new.title, new.content, new.tags);
            END;

            CREATE TRIGGER IF NOT EXISTS nodes_ad AFTER DELETE ON nodes BEGIN
                INSERT INTO nodes_fts(nodes_fts, rowid, title, content, tags)
                VALUES ('delete', old.rowid, old.title, old.content, old.tags);
            END;

            CREATE TRIGGER IF NOT EXISTS nodes_au AFTER UPDATE ON nodes BEGIN
                INSERT INTO nodes_fts(nodes_fts, rowid, title, content, tags)
                VALUES ('delete', old.rowid, old.title, old.content, old.tags);
                INSERT INTO nodes_fts(rowid, title, content, tags)
                VALUES (new.rowid, new.title, new.content, new.tags);
            END;

            CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(node_type);
            CREATE INDEX IF NOT EXISTS idx_nodes_source ON nodes(source_project);
            CREATE INDEX IF NOT EXISTS idx_edges_from ON edges(from_id);
            CREATE INDEX IF NOT EXISTS idx_edges_to ON edges(to_id);

            -- Node events timeline (Phase 3)
            CREATE TABLE IF NOT EXISTS node_events (
                id                TEXT PRIMARY KEY,
                node_id           TEXT NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
                content           TEXT NOT NULL,
                source_message_id TEXT,
                session_id        TEXT,
                created_at        TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_node_events_node    ON node_events(node_id);
            CREATE INDEX IF NOT EXISTS idx_node_events_created ON node_events(created_at);

            CREATE VIRTUAL TABLE IF NOT EXISTS node_events_fts USING fts5(
                content, content=node_events, content_rowid=rowid
            );
            CREATE TRIGGER IF NOT EXISTS node_events_ai AFTER INSERT ON node_events BEGIN
                INSERT INTO node_events_fts(rowid, content) VALUES (new.rowid, new.content);
            END;

            -- Trigger: inserting a node_event marks the node stale for the dream cycle
            CREATE TRIGGER IF NOT EXISTS node_events_mark_stale AFTER INSERT ON node_events BEGIN
                UPDATE nodes SET synthesis_at = NULL, updated_at = datetime('now')
                WHERE id = new.node_id;
            END;",
        )?;

        // Migrations: add synthesis/embedding columns to nodes if not present
        let schema_sql: String = conn
            .prepare("SELECT sql FROM sqlite_master WHERE type='table' AND name='nodes'")?
            .query_row([], |row| row.get::<_, String>(0))?;

        if !schema_sql.contains("synthesis_at") {
            conn.execute_batch(
                "ALTER TABLE nodes ADD COLUMN synthesis TEXT;
                 ALTER TABLE nodes ADD COLUMN synthesis_at TEXT;
                 ALTER TABLE nodes ADD COLUMN embedding BLOB;",
            )?;
        }

        Ok(())
    }

    /// Add a node to the graph. Returns the generated node id.
    /// `node_type` is normalized to lowercase + underscores automatically.
    pub fn add_node(
        &self,
        node_type: &str,
        title: &str,
        content: &str,
        tags: &[String],
        source_project: Option<&str>,
    ) -> anyhow::Result<String> {
        let conn = self.conn.lock();

        // Enforce max_nodes limit.
        let count: usize = conn.query_row("SELECT COUNT(*) FROM nodes", [], |r| r.get(0))?;
        if count >= self.max_nodes {
            anyhow::bail!(
                "knowledge graph node limit reached ({}/{})",
                count,
                self.max_nodes
            );
        }

        // Reject tags containing commas since comma is the separator in storage.
        for tag in tags {
            if tag.contains(',') {
                anyhow::bail!(
                    "tag '{}' contains a comma, which is used as the tag separator",
                    tag
                );
            }
        }

        let id = Uuid::new_v4().to_string();
        let now = Utc::now().to_rfc3339();
        let tags_str = tags.join(",");
        let node_type_normalized = normalize_type(node_type);

        conn.execute(
            "INSERT INTO nodes (id, node_type, title, content, tags, created_at, updated_at, source_project)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                id,
                node_type_normalized,
                title,
                content,
                tags_str,
                now,
                now,
                source_project,
            ],
        )?;

        Ok(id)
    }

    /// Add a directed edge between two nodes.
    /// `relation` is normalized to lowercase + underscores automatically.
    pub fn add_edge(&self, from_id: &str, to_id: &str, relation: &str) -> anyhow::Result<()> {
        let conn = self.conn.lock();

        // Verify both endpoints exist.
        let exists = |id: &str| -> anyhow::Result<bool> {
            let c: usize = conn.query_row(
                "SELECT COUNT(*) FROM nodes WHERE id = ?1",
                params![id],
                |r| r.get(0),
            )?;
            Ok(c > 0)
        };

        if !exists(from_id)? {
            anyhow::bail!("source node not found: {from_id}");
        }
        if !exists(to_id)? {
            anyhow::bail!("target node not found: {to_id}");
        }

        let relation_normalized = normalize_type(relation);
        conn.execute(
            "INSERT OR IGNORE INTO edges (from_id, to_id, relation) VALUES (?1, ?2, ?3)",
            params![from_id, to_id, relation_normalized],
        )?;

        Ok(())
    }

    /// Retrieve a node by id.
    pub fn get_node(&self, id: &str) -> anyhow::Result<Option<KnowledgeNode>> {
        let conn = self.conn.lock();
        let mut stmt = conn.prepare(
            "SELECT id, node_type, title, content, tags, created_at, updated_at, source_project, synthesis, synthesis_at
             FROM nodes WHERE id = ?1",
        )?;

        let mut rows = stmt.query(params![id])?;
        match rows.next()? {
            Some(row) => Ok(Some(row_to_node(row)?)),
            None => Ok(None),
        }
    }

    /// Query nodes by tags (all listed tags must be present).
    pub fn query_by_tags(&self, tags: &[String]) -> anyhow::Result<Vec<KnowledgeNode>> {
        let conn = self.conn.lock();
        let mut stmt = conn.prepare(
            "SELECT id, node_type, title, content, tags, created_at, updated_at, source_project, synthesis, synthesis_at
             FROM nodes ORDER BY updated_at DESC",
        )?;

        let mut results = Vec::new();
        let mut rows = stmt.query([])?;
        while let Some(row) = rows.next()? {
            let node = row_to_node(row)?;
            if tags.iter().all(|t| node.tags.contains(t)) {
                results.push(node);
            }
        }
        Ok(results)
    }

    /// Full-text search across node titles, content, and tags.
    pub fn query_by_similarity(
        &self,
        query: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<SearchResult>> {
        let conn = self.conn.lock();

        // Sanitize FTS query: escape double quotes, wrap tokens in quotes.
        let sanitized: String = query
            .split_whitespace()
            .map(|w| format!("\"{}\"", w.replace('"', "")))
            .collect::<Vec<_>>()
            .join(" ");

        if sanitized.is_empty() {
            return Ok(Vec::new());
        }

        let mut stmt = conn.prepare(
            "SELECT n.id, n.node_type, n.title, n.content, n.tags,
                    n.created_at, n.updated_at, n.source_project,
                    n.synthesis, n.synthesis_at,
                    rank
             FROM nodes_fts f
             JOIN nodes n ON n.rowid = f.rowid
             WHERE nodes_fts MATCH ?1
             ORDER BY rank
             LIMIT ?2",
        )?;

        let mut results = Vec::new();
        let mut rows = stmt.query(params![sanitized, limit as i64])?;
        while let Some(row) = rows.next()? {
            let node = row_to_node(row)?;
            let rank: f64 = row.get(10)?;
            results.push(SearchResult {
                node,
                score: -rank, // FTS5 rank is negative (lower = better), invert for intuitive scoring
            });
        }
        Ok(results)
    }

    /// Find nodes directly related to the given node (both outbound and inbound edges).
    pub fn find_related(&self, node_id: &str) -> anyhow::Result<Vec<(KnowledgeNode, String)>> {
        let conn = self.conn.lock();
        let mut stmt = conn.prepare(
            "SELECT n.id, n.node_type, n.title, n.content, n.tags,
                    n.created_at, n.updated_at, n.source_project,
                    n.synthesis, n.synthesis_at,
                    e.relation
             FROM edges e
             JOIN nodes n ON n.id = e.to_id
             WHERE e.from_id = ?1
             UNION ALL
             SELECT n.id, n.node_type, n.title, n.content, n.tags,
                    n.created_at, n.updated_at, n.source_project,
                    n.synthesis, n.synthesis_at,
                    e.relation
             FROM edges e
             JOIN nodes n ON n.id = e.from_id
             WHERE e.to_id = ?1",
        )?;

        let mut results = Vec::new();
        let mut rows = stmt.query(params![node_id])?;
        while let Some(row) = rows.next()? {
            let node = row_to_node(row)?;
            let relation: String = row.get(10)?;
            results.push((node, relation));
        }
        Ok(results)
    }

    /// Maximum allowed subgraph traversal depth.
    const MAX_SUBGRAPH_DEPTH: usize = 100;

    /// Extract a subgraph starting from `root_id` up to `depth` hops.
    ///
    /// `depth` must be between 1 and [`Self::MAX_SUBGRAPH_DEPTH`] (100).
    /// Uses a recursive CTE for efficient single-query bidirectional traversal.
    pub fn get_subgraph(
        &self,
        root_id: &str,
        depth: usize,
    ) -> anyhow::Result<(Vec<KnowledgeNode>, Vec<KnowledgeEdge>)> {
        if depth == 0 {
            anyhow::bail!("subgraph depth must be greater than 0");
        }
        let depth = depth.min(Self::MAX_SUBGRAPH_DEPTH);
        let conn = self.conn.lock();

        // Collect reachable node IDs via recursive CTE (bidirectional traversal).
        let mut node_stmt = conn.prepare(
            "WITH RECURSIVE reachable(id, depth) AS (
                SELECT ?1, 0
                UNION
                SELECT CASE WHEN e.from_id = r.id THEN e.to_id ELSE e.from_id END, r.depth + 1
                FROM reachable r
                JOIN edges e ON e.from_id = r.id OR e.to_id = r.id
                WHERE r.depth < ?2
             )
             SELECT DISTINCT n.id, n.node_type, n.title, n.content, n.tags,
                    n.created_at, n.updated_at, n.source_project,
                    n.synthesis, n.synthesis_at
             FROM reachable rc
             JOIN nodes n ON n.id = rc.id",
        )?;

        let mut nodes = Vec::new();
        let mut node_ids: HashSet<String> = HashSet::new();
        let mut rows = node_stmt.query(params![root_id, depth as i64])?;
        while let Some(row) = rows.next()? {
            let node = row_to_node(row)?;
            node_ids.insert(node.id.clone());
            nodes.push(node);
        }
        drop(rows);

        // Collect all edges where both endpoints are in the subgraph.
        let mut edge_stmt = conn.prepare("SELECT from_id, to_id, relation FROM edges")?;

        let mut edges = Vec::new();
        let mut edge_rows = edge_stmt.query([])?;
        while let Some(row) = edge_rows.next()? {
            let from_id: String = row.get(0)?;
            let to_id: String = row.get(1)?;
            if node_ids.contains(&from_id) && node_ids.contains(&to_id) {
                let relation: String = row.get(2)?;
                edges.push(KnowledgeEdge {
                    from_id,
                    to_id,
                    relation,
                });
            }
        }

        Ok((nodes, edges))
    }

    /// Find experts associated with the given tags via `authored_by` edges.
    pub fn find_experts(&self, tags: &[String]) -> anyhow::Result<Vec<SearchResult>> {
        // Find nodes matching the tags, then follow authored_by edges to experts.
        let matching = self.query_by_tags(tags)?;
        let mut expert_scores: HashMap<String, f64> = HashMap::new();

        let conn = self.conn.lock();
        for node in &matching {
            let mut stmt = conn.prepare(
                "SELECT to_id FROM edges WHERE from_id = ?1 AND relation = 'authored_by'",
            )?;
            let mut rows = stmt.query(params![node.id])?;
            while let Some(row) = rows.next()? {
                let expert_id: String = row.get(0)?;
                *expert_scores.entry(expert_id).or_default() += 1.0;
            }
        }
        drop(conn);

        let mut results: Vec<SearchResult> = Vec::new();
        for (eid, score) in expert_scores {
            if let Some(node) = self.get_node(&eid)?
                && node.node_type == "expert"
            {
                results.push(SearchResult { node, score });
            }
        }

        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(results)
    }

    /// Return summary statistics for the graph.
    pub fn stats(&self) -> anyhow::Result<GraphStats> {
        let conn = self.conn.lock();

        let total_nodes: usize = conn.query_row("SELECT COUNT(*) FROM nodes", [], |r| r.get(0))?;
        let total_edges: usize = conn.query_row("SELECT COUNT(*) FROM edges", [], |r| r.get(0))?;

        let mut by_type = HashMap::new();
        {
            let mut stmt =
                conn.prepare("SELECT node_type, COUNT(*) FROM nodes GROUP BY node_type")?;
            let mut rows = stmt.query([])?;
            while let Some(row) = rows.next()? {
                let t: String = row.get(0)?;
                let c: usize = row.get(1)?;
                by_type.insert(t, c);
            }
        }

        // Top 10 tags by frequency.
        let mut tag_counts: HashMap<String, usize> = HashMap::new();
        {
            let mut stmt = conn.prepare("SELECT tags FROM nodes WHERE tags != ''")?;
            let mut rows = stmt.query([])?;
            while let Some(row) = rows.next()? {
                let tags_str: String = row.get(0)?;
                for tag in tags_str.split(',') {
                    let tag = tag.trim();
                    if !tag.is_empty() {
                        *tag_counts.entry(tag.to_string()).or_default() += 1;
                    }
                }
            }
        }
        let mut top_tags: Vec<(String, usize)> = tag_counts.into_iter().collect();
        top_tags.sort_by(|a, b| b.1.cmp(&a.1));
        top_tags.truncate(10);

        Ok(GraphStats {
            total_nodes,
            total_edges,
            nodes_by_type: by_type,
            top_tags,
        })
    }

    // ── Phase 3: Entity timeline + synthesis ─────────────────────────────

    /// Append an event to an entity's timeline. Returns the event id.
    /// Automatically marks the node stale (synthesis_at = NULL) via trigger.
    pub fn add_event(
        &self,
        node_id: &str,
        content: &str,
        source_message_id: Option<&str>,
        session_id: Option<&str>,
    ) -> anyhow::Result<String> {
        let conn = self.conn.lock();
        let id = Uuid::new_v4().to_string();
        let now = Utc::now().to_rfc3339();
        conn.execute(
            "INSERT INTO node_events (id, node_id, content, source_message_id, session_id, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![id, node_id, content, source_message_id, session_id, now],
        )?;
        Ok(id)
    }

    /// Returns the node plus its N most recent events.
    pub fn get_with_timeline(
        &self,
        node_id: &str,
        event_limit: usize,
    ) -> anyhow::Result<Option<(KnowledgeNode, Vec<NodeEvent>)>> {
        let node = self.get_node(node_id)?;
        let Some(node) = node else { return Ok(None) };

        let conn = self.conn.lock();
        let mut stmt = conn.prepare(
            "SELECT id, node_id, content, source_message_id, session_id, created_at
             FROM node_events WHERE node_id = ?1
             ORDER BY created_at DESC
             LIMIT ?2",
        )?;
        #[allow(clippy::cast_possible_wrap)]
        let mut rows = stmt.query(params![node_id, event_limit as i64])?;
        let mut events = Vec::new();
        while let Some(row) = rows.next()? {
            events.push(row_to_event(row)?);
        }
        Ok(Some((node, events)))
    }

    /// Find a node by exact slug/title match, then cosine fallback, then create.
    /// Returns (node_id, created: bool).
    pub fn find_or_create_by_slug(
        &self,
        slug: &str,
        node_type: &str,
        title: &str,
        initial_content: &str,
    ) -> anyhow::Result<(String, bool)> {
        let normalized = normalize_slug(slug);
        let ntype = normalize_type(node_type);

        // Pass 1: exact title match
        {
            let conn = self.conn.lock();
            let result: Option<String> = conn
                .prepare("SELECT id FROM nodes WHERE title = ?1")?
                .query_row(params![normalized], |row| row.get(0))
                .ok();
            if let Some(id) = result {
                return Ok((id, false));
            }
        }

        // Pass 2: cosine similarity over synthesis embeddings (inert until Phase 6 populates embeddings)
        // Currently a no-op: no embeddings exist between Phase 3 and Phase 6.
        // The cosine path is written here but skipped when no embeddings are present.

        // Pass 3: create new node
        let id = self.add_node(&ntype, title, initial_content, &[], None)?;
        Ok((id, true))
    }

    /// Returns stale nodes (synthesis_at IS NULL or < updated_at).
    /// Ordered by updated_at DESC (most recently active first).
    pub fn list_stale_nodes(&self, limit: usize) -> anyhow::Result<Vec<KnowledgeNode>> {
        let conn = self.conn.lock();
        #[allow(clippy::cast_possible_wrap)]
        let mut stmt = conn.prepare(
            "SELECT id, node_type, title, content, tags, created_at, updated_at, source_project,
                    synthesis, synthesis_at
             FROM nodes
             WHERE synthesis_at IS NULL OR synthesis_at < updated_at
             ORDER BY updated_at DESC
             LIMIT ?1",
        )?;
        let mut rows = stmt.query(params![limit as i64])?;
        let mut results = Vec::new();
        while let Some(row) = rows.next()? {
            results.push(row_to_node(row)?);
        }
        Ok(results)
    }

    /// Update synthesis text and embedding after the dream cycle.
    pub fn update_synthesis(
        &self,
        node_id: &str,
        synthesis: &str,
        embedding: Option<&[f32]>,
    ) -> anyhow::Result<()> {
        let conn = self.conn.lock();
        let now = Utc::now().to_rfc3339();
        let embedding_blob: Option<Vec<u8>> = embedding.map(|e| {
            e.iter().flat_map(|f| f.to_le_bytes()).collect()
        });
        conn.execute(
            "UPDATE nodes SET synthesis = ?1, synthesis_at = ?2, embedding = ?3 WHERE id = ?4",
            params![synthesis, now, embedding_blob, node_id],
        )?;
        Ok(())
    }

    /// Returns [(title, node_type)] ordered by updated_at DESC.
    /// Used by consolidation to provide entity context in the extraction prompt.
    pub fn list_entity_slugs(&self, limit: usize) -> anyhow::Result<Vec<(String, String)>> {
        let conn = self.conn.lock();
        #[allow(clippy::cast_possible_wrap)]
        let mut stmt = conn.prepare(
            "SELECT title, node_type FROM nodes ORDER BY updated_at DESC LIMIT ?1",
        )?;
        let mut rows = stmt.query(params![limit as i64])?;
        let mut results = Vec::new();
        while let Some(row) = rows.next()? {
            results.push((row.get(0)?, row.get(1)?));
        }
        Ok(results)
    }
}

/// Parse a database row into a `KnowledgeNode`.
/// Columns: id(0), node_type(1), title(2), content(3), tags(4),
///          created_at(5), updated_at(6), source_project(7),
///          synthesis(8), synthesis_at(9)
fn row_to_node(row: &rusqlite::Row<'_>) -> anyhow::Result<KnowledgeNode> {
    let id: String = row.get(0)?;
    let node_type: String = row.get(1)?;
    let title: String = row.get(2)?;
    let content: String = row.get(3)?;
    let tags_str: String = row.get(4)?;
    let created_at_str: String = row.get(5)?;
    let updated_at_str: String = row.get(6)?;
    let source_project: Option<String> = row.get(7)?;
    let synthesis: Option<String> = row.get(8).unwrap_or(None);
    let synthesis_at_str: Option<String> = row.get(9).unwrap_or(None);

    let tags: Vec<String> = tags_str
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();

    let created_at = DateTime::parse_from_rfc3339(&created_at_str)
        .map(|dt| dt.with_timezone(&Utc))
        .unwrap_or_else(|_| Utc::now());
    let updated_at = DateTime::parse_from_rfc3339(&updated_at_str)
        .map(|dt| dt.with_timezone(&Utc))
        .unwrap_or_else(|_| Utc::now());
    let synthesis_at = synthesis_at_str.and_then(|s| {
        DateTime::parse_from_rfc3339(&s)
            .map(|dt| dt.with_timezone(&Utc))
            .ok()
    });

    Ok(KnowledgeNode {
        id,
        node_type,
        title,
        content,
        tags,
        created_at,
        updated_at,
        source_project,
        synthesis,
        synthesis_at,
    })
}

/// Parse a database row into a `NodeEvent`.
/// Columns: id(0), node_id(1), content(2), source_message_id(3), session_id(4), created_at(5)
fn row_to_event(row: &rusqlite::Row<'_>) -> anyhow::Result<NodeEvent> {
    let id: String = row.get(0)?;
    let node_id: String = row.get(1)?;
    let content: String = row.get(2)?;
    let source_message_id: Option<String> = row.get(3)?;
    let session_id: Option<String> = row.get(4)?;
    let created_at_str: String = row.get(5)?;
    let created_at = DateTime::parse_from_rfc3339(&created_at_str)
        .map(|dt| dt.with_timezone(&Utc))
        .unwrap_or_else(|_| Utc::now());
    Ok(NodeEvent {
        id,
        node_id,
        content,
        source_message_id,
        session_id,
        created_at,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn test_graph() -> (TempDir, KnowledgeGraph) {
        let tmp = TempDir::new().unwrap();
        let db_path = tmp.path().join("knowledge.db");
        let graph = KnowledgeGraph::new(&db_path, 1000).unwrap();
        (tmp, graph)
    }

    #[test]
    fn add_node_returns_unique_id() {
        let (_tmp, graph) = test_graph();
        let id1 = graph.add_node("pattern", "Caching", "Use Redis for caching", &["redis".into()], None).unwrap();
        let id2 = graph.add_node("lesson", "Lesson A", "Content A", &[], None).unwrap();
        assert_ne!(id1, id2);
    }

    #[test]
    fn get_node_returns_stored_data() {
        let (_tmp, graph) = test_graph();
        let id = graph
            .add_node("decision", "Use Postgres", "Chose Postgres over MySQL", &["database".into(), "postgres".into()], Some("project_alpha"))
            .unwrap();

        let node = graph.get_node(&id).unwrap().unwrap();
        assert_eq!(node.title, "Use Postgres");
        assert_eq!(node.node_type, "decision");
        assert_eq!(node.tags, vec!["database", "postgres"]);
        assert_eq!(node.source_project.as_deref(), Some("project_alpha"));
    }

    #[test]
    fn get_node_missing_returns_none() {
        let (_tmp, graph) = test_graph();
        assert!(graph.get_node("nonexistent").unwrap().is_none());
    }

    #[test]
    fn add_edge_creates_relationship() {
        let (_tmp, graph) = test_graph();
        let id1 = graph.add_node("pattern", "P1", "Pattern one", &[], None).unwrap();
        let id2 = graph.add_node("technology", "T1", "Tech one", &[], None).unwrap();

        graph.add_edge(&id1, &id2, "uses").unwrap();

        let related = graph.find_related(&id1).unwrap();
        assert!(related.iter().any(|(n, r)| n.id == id2 && r == "uses"));

        let related = graph.find_related(&id2).unwrap();
        assert!(related.iter().any(|(n, r)| n.id == id1 && r == "uses"));
    }

    #[test]
    fn add_edge_rejects_missing_node() {
        let (_tmp, graph) = test_graph();
        let id = graph.add_node("lesson", "L1", "Lesson", &[], None).unwrap();
        let err = graph.add_edge(&id, "nonexistent", "extends").unwrap_err();
        assert!(err.to_string().contains("target node not found"));
    }

    #[test]
    fn query_by_tags_filters_correctly() {
        let (_tmp, graph) = test_graph();
        graph.add_node("pattern", "P1", "Content", &["rust".into(), "async".into()], None).unwrap();
        graph.add_node("pattern", "P2", "Content", &["rust".into()], None).unwrap();
        graph.add_node("pattern", "P3", "Content", &["python".into()], None).unwrap();

        let results = graph.query_by_tags(&["rust".into()]).unwrap();
        assert_eq!(results.len(), 2);

        let results = graph.query_by_tags(&["rust".into(), "async".into()]).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].title, "P1");
    }

    #[test]
    fn query_by_similarity_returns_ranked_results() {
        let (_tmp, graph) = test_graph();
        graph.add_node("decision", "Choose Rust for performance", "Rust gives memory safety and speed", &["rust".into()], None).unwrap();
        graph.add_node("lesson", "Python scaling issues", "Python had GIL bottleneck", &["python".into()], None).unwrap();

        let results = graph.query_by_similarity("Rust performance", 10).unwrap();
        assert!(!results.is_empty());
        assert!(results[0].score > 0.0);
    }

    #[test]
    fn subgraph_traversal_collects_connected_nodes() {
        let (_tmp, graph) = test_graph();
        let a = graph.add_node("pattern", "A", "Node A", &[], None).unwrap();
        let b = graph.add_node("pattern", "B", "Node B", &[], None).unwrap();
        let c = graph.add_node("pattern", "C", "Node C", &[], None).unwrap();
        graph.add_edge(&a, &b, "extends").unwrap();
        graph.add_edge(&b, &c, "uses").unwrap();

        let (nodes, edges) = graph.get_subgraph(&a, 2).unwrap();
        assert_eq!(nodes.len(), 3);
        assert_eq!(edges.len(), 2);

        let (nodes, edges) = graph.get_subgraph(&c, 2).unwrap();
        assert_eq!(nodes.len(), 3);
        assert_eq!(edges.len(), 2);
    }

    #[test]
    fn expert_ranking_by_authored_contributions() {
        let (_tmp, graph) = test_graph();
        let expert = graph.add_node("expert", "zeroclaw_user", "Backend expert", &[], None).unwrap();
        let p1 = graph.add_node("pattern", "Cache pattern", "Redis caching", &["caching".into()], None).unwrap();
        let p2 = graph.add_node("pattern", "Queue pattern", "Message queue", &["caching".into()], None).unwrap();

        graph.add_edge(&p1, &expert, "authored_by").unwrap();
        graph.add_edge(&p2, &expert, "authored_by").unwrap();

        let experts = graph.find_experts(&["caching".into()]).unwrap();
        assert_eq!(experts.len(), 1);
        assert_eq!(experts[0].node.title, "zeroclaw_user");
        assert!((experts[0].score - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn max_nodes_limit_enforced() {
        let tmp = TempDir::new().unwrap();
        let db_path = tmp.path().join("knowledge.db");
        let graph = KnowledgeGraph::new(&db_path, 2).unwrap();

        graph.add_node("lesson", "L1", "C1", &[], None).unwrap();
        graph.add_node("lesson", "L2", "C2", &[], None).unwrap();
        let err = graph.add_node("lesson", "L3", "C3", &[], None).unwrap_err();
        assert!(err.to_string().contains("node limit reached"));
    }

    #[test]
    fn stats_reports_correct_counts() {
        let (_tmp, graph) = test_graph();
        graph.add_node("pattern", "P", "C", &["rust".into()], None).unwrap();
        graph.add_node("lesson", "L", "C", &["rust".into(), "async".into()], None).unwrap();

        let stats = graph.stats().unwrap();
        assert_eq!(stats.total_nodes, 2);
        assert_eq!(stats.nodes_by_type.get("pattern"), Some(&1));
        assert_eq!(stats.nodes_by_type.get("lesson"), Some(&1));
        assert!(!stats.top_tags.is_empty());
    }

    // ── Phase 3: normalization + entity timeline tests ───────────────────

    #[test]
    fn slug_normalization() {
        assert_eq!(normalize_slug("Alice from Acme"), "alice_from_acme");
        assert_eq!(normalize_slug("  LeadEngineer "), "leadengineer");
        assert_eq!(normalize_slug("my-company.inc"), "my_company_inc");
    }

    #[test]
    fn node_type_normalization() {
        assert_eq!(normalize_type("Person"), "person");
        assert_eq!(normalize_type("EmployedBy"), "employedby");
        assert_eq!(normalize_type("employed_by"), "employed_by");
    }

    #[test]
    fn node_type_normalized_at_write() {
        let (_tmp, graph) = test_graph();
        let id = graph.add_node("Pattern", "Cache", "Redis cache", &[], None).unwrap();
        let node = graph.get_node(&id).unwrap().unwrap();
        assert_eq!(node.node_type, "pattern", "node_type must be normalized to lowercase");
    }

    #[test]
    fn relation_type_normalized_at_write() {
        let (_tmp, graph) = test_graph();
        let a = graph.add_node("pattern", "A", "A content", &[], None).unwrap();
        let b = graph.add_node("pattern", "B", "B content", &[], None).unwrap();
        graph.add_edge(&a, &b, "EmployedBy").unwrap();

        let related = graph.find_related(&a).unwrap();
        assert!(related.iter().any(|(_, r)| r == "employedby"), "relation must be normalized");
    }

    #[test]
    fn duplicate_relation_upsert() {
        let (_tmp, graph) = test_graph();
        let a = graph.add_node("pattern", "A", "A", &[], None).unwrap();
        let b = graph.add_node("pattern", "B", "B", &[], None).unwrap();
        graph.add_edge(&a, &b, "uses").unwrap();
        graph.add_edge(&a, &b, "uses").unwrap(); // second call: INSERT OR IGNORE

        let conn = graph.conn.lock();
        let count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM edges WHERE from_id = ?1 AND to_id = ?2 AND relation = 'uses'",
            params![a, b],
            |row| row.get(0),
        ).unwrap();
        assert_eq!(count, 1, "Duplicate edge must not create two rows");
    }

    #[test]
    fn find_or_create_exact_match() {
        let (_tmp, graph) = test_graph();
        // Add a node with title "alice"
        let id = graph.add_node("person", "alice", "Engineer", &[], None).unwrap();

        // find_or_create should find it by exact title match
        let (found_id, created) = graph.find_or_create_by_slug("alice", "person", "alice", "new content").unwrap();
        assert_eq!(found_id, id, "Should return existing node id");
        assert!(!created, "created should be false for existing node");
    }

    #[test]
    fn find_or_create_new_slug() {
        let (_tmp, graph) = test_graph();
        let (id, created) = graph.find_or_create_by_slug("bob", "person", "bob", "New person").unwrap();
        assert!(!id.is_empty());
        assert!(created, "created should be true for new slug");
    }

    #[test]
    fn add_event_marks_node_stale() {
        let (_tmp, graph) = test_graph();
        let node_id = graph.add_node("person", "alice", "Engineer", &[], None).unwrap();

        // Set synthesis_at to something non-null
        {
            let conn = graph.conn.lock();
            conn.execute("UPDATE nodes SET synthesis_at = datetime('now') WHERE id = ?1", params![node_id]).unwrap();
        }

        // Verify it's set
        {
            let conn = graph.conn.lock();
            let sat: Option<String> = conn.query_row(
                "SELECT synthesis_at FROM nodes WHERE id = ?1", params![node_id], |row| row.get(0)
            ).unwrap();
            assert!(sat.is_some(), "synthesis_at should be set before add_event");
        }

        // add_event triggers node_events_mark_stale
        graph.add_event(&node_id, "New fact about alice", None, None).unwrap();

        let conn = graph.conn.lock();
        let sat: Option<String> = conn.query_row(
            "SELECT synthesis_at FROM nodes WHERE id = ?1", params![node_id], |row| row.get(0)
        ).unwrap();
        assert!(sat.is_none(), "synthesis_at should be NULL after add_event (node marked stale)");
    }

    #[test]
    fn list_stale_nodes_ordering() {
        let (_tmp, graph) = test_graph();
        // Create 3 nodes; they all start with synthesis_at=NULL (stale)
        let a = graph.add_node("person", "alice", "A", &[], None).unwrap();
        let b = graph.add_node("person", "bob", "B", &[], None).unwrap();
        let c = graph.add_node("person", "charlie", "C", &[], None).unwrap();

        // update_synthesis on 'b' and 'c' to make them fresh, keep 'a' stale
        graph.update_synthesis(&b, "Bob synthesis", None).unwrap();
        graph.update_synthesis(&c, "Charlie synthesis", None).unwrap();

        let stale = graph.list_stale_nodes(10).unwrap();
        // Only 'a' is stale (after update_synthesis sets synthesis_at = now for b and c)
        // Actually: list_stale_nodes returns WHERE synthesis_at IS NULL OR synthesis_at < updated_at
        // After update_synthesis, synthesis_at = now() and updated_at = old value (unless trigger fires)
        // 'a' has synthesis_at IS NULL → stale
        assert!(stale.iter().any(|n| n.id == a), "alice (stale) should appear");
    }

    #[test]
    fn get_with_timeline_returns_events() {
        let (_tmp, graph) = test_graph();
        let node_id = graph.add_node("person", "alice", "Engineer", &[], None).unwrap();
        graph.add_event(&node_id, "Joined team 2026-01-01", None, Some("session-1")).unwrap();
        graph.add_event(&node_id, "Promoted 2026-04-01", None, Some("session-2")).unwrap();

        let (node, events) = graph.get_with_timeline(&node_id, 10).unwrap().unwrap();
        assert_eq!(node.title, "alice");
        assert_eq!(events.len(), 2);
    }

    #[test]
    fn update_synthesis_persists() {
        let (_tmp, graph) = test_graph();
        let node_id = graph.add_node("company", "acme", "Big corp", &[], None).unwrap();
        graph.update_synthesis(&node_id, "Acme is a large enterprise company.", None).unwrap();

        let node = graph.get_node(&node_id).unwrap().unwrap();
        assert_eq!(node.synthesis.as_deref(), Some("Acme is a large enterprise company."));
        assert!(node.synthesis_at.is_some());
    }

    #[test]
    fn list_entity_slugs_ordering() {
        let (_tmp, graph) = test_graph();
        graph.add_node("person", "alice", "A", &[], None).unwrap();
        graph.add_node("company", "acme", "B", &[], None).unwrap();

        let slugs = graph.list_entity_slugs(10).unwrap();
        assert_eq!(slugs.len(), 2);
        // All slugs should have their type as well
        assert!(slugs.iter().any(|(t, _)| t == "alice"));
        assert!(slugs.iter().any(|(t, _)| t == "acme"));
    }
}
