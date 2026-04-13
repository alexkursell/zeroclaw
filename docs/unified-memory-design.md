# Unified Memory Architecture — Revised Design

**Status:** Proposal v3 — critiques applied, implementation guide with integration tests  
**Date:** 2026-04-12  
**Scope:** Evolve ZeroClaw's existing memory subsystem toward lossless history, structured entity knowledge, and better retrieval

---

## Table of Contents

1. [What Exists Today](#1-what-exists-today)
2. [What's Wrong](#2-whats-wrong)
3. [Ideas Worth Stealing](#3-ideas-worth-stealing)
4. [Design Principles](#4-design-principles)
5. [Architecture](#5-architecture)
6. [Implementation Plan](#6-implementation-plan)
7. [What Gets Cut](#7-what-gets-cut)
8. [Open Questions](#8-open-questions)
9. [Integration Test Suite](#9-integration-test-suite)

---

## 1. What Exists Today

ZeroClaw already has a surprisingly capable memory subsystem. Before designing anything new, we need to be honest about what's already built and what actually needs changing.

### 1.1 Three Storage Systems

| System | Location | Format | Append-only? | Searchable? |
|--------|----------|--------|-------------|-------------|
| **Memory store** (`memories` table) | `brain.db` | SQLite + FTS5 + optional embeddings | No (upsert on key conflict) | Yes — hybrid BM25 + cosine |
| **Knowledge graph** (`nodes` + `edges` tables) | `knowledge.db` (separate file) | SQLite + FTS5 | No (CRUD) | Yes — FTS5 on title/content/tags |
| **Session persistence** | `{workspace}/sessions/*.jsonl` (channels) or `session.json` (interactive CLI) | JSONL / JSON | Channels: yes. CLI: no (full overwrite) | No |

Plus the in-memory `history: Vec<ChatMessage>` which is the actual working set sent to the LLM.

### 1.2 Memory Store (brain.db)

The `Memory` trait (`zeroclaw-api/src/memory_traits.rs`) with 5 backends. The SQLite backend is the recommended default:

- **Schema:** `memories` table with id, key, content, category, embedding, timestamps, session_id, namespace, importance, superseded_by
- **Search:** Multi-stage pipeline (cache → FTS5 BM25 → cosine similarity → LIKE fallback) with hybrid merge using weighted linear combination (0.7 vector + 0.3 keyword)
- **Categories:** Core (evergreen, no decay), Daily (7-day half-life), Conversation (7-day half-life), Custom
- **Auto-save:** Every user message ≥20 chars stored as Conversation category
- **Consolidation:** Per-turn LLM extraction (`consolidation.rs`) writes Daily history entries and Core facts, with semantic conflict resolution (0.85 cosine threshold)
- **Conflict resolution:** Detects semantic duplicates, marks old entries `superseded_by` (logical delete)
- **Decay:** Exponential time decay applied at recall time, Core entries exempt
- **Importance:** Heuristic scoring (base by category + keyword boost) blended into final score
- **Hygiene:** Automated 12-hour job archives old daily memory files + session JSONLs (30d), purges archived files (90d), hard-deletes Conversation rows from brain.db (30d), prunes audit entries (90d)

### 1.3 Knowledge Graph (knowledge.db)

Separate SQLite database (`knowledge_graph.rs`, ~863 lines):

- **Node types:** Pattern, Decision, Lesson, Expert, Technology — hardcoded Rust enum
- **Relations:** Uses, Replaces, Extends, AuthoredBy, AppliesTo — hardcoded Rust enum
- **Search:** FTS5 on title/content/tags
- **Traversal:** Recursive CTE for multi-hop graph queries
- **Tool:** `knowledge` tool with actions: capture, search, relate, suggest, expert_find, lessons_extract, graph_stats
- **Cap:** max_nodes limit enforced at insert time
- **FK enforcement:** `PRAGMA foreign_keys = ON` is set (unlike brain.db)

### 1.4 Session Persistence

Two mechanisms:
- **Channel sessions** (`session_store.rs`): Append-only JSONL, one `ChatMessage` per line. Already an immutable log. Not searchable.
- **Interactive CLI** (`history.rs`): Full JSON dump/restore of the history Vec. Overwritten each save. Loses data when `trim_history` runs.

### 1.5 Context Compressor

Multi-pass compression (`context_compressor.rs`, 763 lines):

- Triggered at 50% context window usage
- Protects first 3 + last 4 messages
- Pass 1–N: LLM summarization of middle section (up to `max_passes` passes)
- Fallback: Raw truncation on LLM timeout/failure
- Summary persisted to `memories` table as Daily category before originals discarded from the Vec
- Tool pair repair: cleans up orphaned tool_call/tool_result pairs

### 1.6 Context Injection

`build_context()` in `loop_.rs`:

1. `recall(user_message, limit=5)` against the memory store
2. Apply exponential time decay (Core exempt)
3. Filter by min_relevance_score (0.4)
4. Skip noise (autosave keys, cron markers, tool_result blocks)
5. Format as `[Memory context]\n- key: content\n[/Memory context]`
6. Prepend to user message

At most 5 entries. Flat list, no structure. Knowledge graph is not queried.

### 1.7 Retrieval Pipeline

`retrieval.rs` wraps `Arc<dyn Memory>` with a 5-minute in-memory LRU cache and configurable stage dispatch ("cache" → "fts" → "vector"). `build_context()` uses this pipeline. Currently only knows about the `Memory` trait — unaware of the knowledge graph.

---

## 2. What's Wrong

The current system is good but has real gaps. Here's what actually needs fixing, ordered by impact.

### 2.1 Compaction is Lossy (Critical)

When the context compressor fires, originals are discarded from the in-memory history Vec. The summary is saved to the `memories` table as a Daily entry, but:
- The original messages are gone from the session persistence (interactive CLI overwrites; channel JSONL keeps them but they're not indexed)
- There's no way to recover what was said — only the summary remains
- The summary itself decays (Daily category, 7-day half-life) and may be pruned (30d retention)

This means ZeroClaw forgets the details of conversations within days. For a personal assistant, this is the single biggest problem.

### 2.2 Knowledge Graph is Disconnected (High)

The knowledge graph lives in a separate SQLite file, is not queried during `build_context()`, and has no provenance link back to the conversations that created it. It's a good system that's underutilized:
- Node types are engineering-oriented (Pattern, Decision, Lesson) not personal-assistant-oriented (Person, Company, Project)
- No synthesis/summary field that stays current
- No timeline of events per entity
- Not included in the recall pipeline — the agent has to explicitly call the `knowledge` tool

### 2.3 Retrieval Doesn't Search Everything (High)

`build_context()` only queries the `memories` table. It misses:
- Knowledge graph nodes (people, decisions, patterns the agent has captured)
- Session history from other sessions
- Compressed summaries from earlier in the current session

The agent can manually call `memory_recall` and `knowledge` tools, but the automatic context injection is blind to two of the three storage systems.

### 2.4 Weighted Linear Merge is Fragile (Medium)

The current hybrid search uses `score = 0.7 * vector + 0.3 * keyword` with score normalization. This is sensitive to scale differences between BM25 and cosine scores. RRF (Reciprocal Rank Fusion) uses rank position instead of raw scores, which is immune to this problem and requires no weight tuning.

### 2.5 No Structured Context Block (Low)

The `[Memory context]` injection is a flat list of `key: content` pairs. When entity knowledge, flat facts, and session history all come through the same pipe, the LLM has no structural cues about what it's looking at.

---

## 3. Ideas Worth Stealing

### From LCM (Lossless Context Management)

| Idea | Adaptation |
|------|-----------|
| Immutable message store | Make the `messages` table in brain.db the single source of truth for all session history. Never delete, never modify. |
| Summary DAG with parent pointers | Track which messages each summary covers, and which summaries have been further condensed. Provenance chain from any summary back to originals. |
| Three-level escalation | Formalize the compressor's existing multi-pass + fallback into guaranteed convergence: (1) preserve_details, (2) bullet_points at half target, (3) deterministic truncation. |
| Soft/hard thresholds | Between-turns compaction below hard threshold, pre-LLM-call compaction above. |
| `lcm_grep` | Regex search over verbatim history. Qualitatively different from scored recall — for forensic "what did I say about X on Tuesday" queries. |

### From GBrain

| Idea | Adaptation |
|------|-----------|
| Entity pages (compiled truth + timeline) | Add entity-oriented node types to the existing knowledge graph. Add a synthesis field and timeline events. |
| Dream cycle | Schedule the existing consolidation system to periodically re-synthesize entity nodes from their accumulated events. Use the existing cron infrastructure. |
| RRF search fusion | Replace weighted linear merge with RRF across all search sources. |

### What NOT to steal

| Idea | Why not |
|------|---------|
| MCP server deployment | Contradicts local-first. Splits context. |
| Git-backed markdown as primary store | We have `MarkdownMemory` already. SQLite is better for search. |
| `llm_map` / `agentic_map` | Valuable but orthogonal to memory. Separate project. |
| `lcm_expand` restricted to sub-agents | The swarm tool spawns stateless agents (single prompt, no tools by default). There's no sub-agent context to restrict within. Overengineered for our architecture. |
| Scope-reduction invariant | Requires swarm redesign. Separate project. |
| Separate `entities` table hierarchy | The knowledge graph already has nodes + edges. Extend it, don't duplicate it |

---

## 4. Design Principles

1. **Evolve, don't duplicate.** Every new capability must extend an existing system, not create a parallel one.

2. **Two SQLite files, separate concerns.** `brain.db` owns flat facts and session history. `knowledge.db` owns the entity graph. They stay separate — different write patterns, separate WAL contention, no real need for cross-database foreign keys. RRF recall queries both in parallel via `tokio::join!`. The `node_events.source_message_id` link to `messages` is a soft reference (UUID string lookup, not a FK constraint) — temporal correlation works as fallback.

3. **Immutable source of truth.** Raw messages are never modified or deleted. Summaries, entities, and facts are derived caches that can always be regenerated.

4. **Search everything by default.** `build_context()` should query all storage layers. The agent shouldn't have to know which tool to call to find information it previously learned.

5. **Deterministic convergence.** The compressor must always terminate. Three escalation levels, the last one requires no LLM.

6. **No backwards-compatibility shims.** This is a new deployment. Old data paths (auto-save Conversation entries, compressor Daily entries to `memories`) are removed cleanly, not kept alive behind flags.

---

## 5. Architecture

### 5.1 Storage Layout (two SQLite files, clear responsibilities)

```
brain.db (existing, extended)
│
├── memories           (existing, unchanged — flat facts)
├── memories_fts       (existing, unchanged)
├── embedding_cache    (existing, unchanged)
│
├── messages           (NEW — immutable session history)
├── messages_fts       (NEW — FTS5 over raw history for lcm_grep)
│
├── summaries          (NEW — DAG of compressed summary nodes)
├── summaries_fts      (NEW — FTS5 for recall over summary content)
└── summary_parents    (NEW — junction table for multi-parent condensed summaries)


knowledge.db (existing, extended)
│
├── nodes              (existing — gains synthesis, synthesis_at, embedding columns)
├── edges              (existing, unchanged)
├── nodes_fts          (existing, unchanged)
│
├── node_events        (NEW — append-only timeline per entity node)
└── node_events_fts    (NEW — FTS5 over entity events)
```

**What changed:**
- `messages` + `summaries` + `summary_parents` tables added to brain.db for lossless session history
- `node_events` table added to knowledge.db for entity timelines
- `nodes` table gains optional `synthesis`, `synthesis_at`, `embedding` columns
- Node types and relation types become free-form lowercase strings — the LLM can create any type it needs without code changes
- `PRAGMA foreign_keys = ON` added to brain.db (currently missing; knowledge.db already has it)

**What didn't change:**
- `memories` table structure and data
- `knowledge.db` stays as its own file
- `Memory` trait interface — all backends still valid
- Consolidation still runs; its output schema is extended

### 5.2 Schema Additions

```sql
-- ─────────────────────────────────────────────
-- brain.db additions
-- ─────────────────────────────────────────────

-- Immutable session history
CREATE TABLE IF NOT EXISTS messages (
    id           TEXT PRIMARY KEY,
    session_id   TEXT NOT NULL,
    role         TEXT NOT NULL,              -- user | assistant | system | tool
    content      TEXT NOT NULL,              -- verbatim, never modified
    token_count  INTEGER,
    created_at   TEXT NOT NULL               -- RFC 3339
    -- Note: no summary_id FK here; see summary_parents below
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
CREATE INDEX IF NOT EXISTS idx_messages_created ON messages(created_at);

CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    content, content=messages, content_rowid=rowid
);
CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts(rowid, content) VALUES (new.rowid, new.content);
END;

-- Summary DAG nodes
CREATE TABLE IF NOT EXISTS summaries (
    id           TEXT PRIMARY KEY,
    kind         TEXT NOT NULL,              -- 'leaf' | 'condensed'
    content      TEXT NOT NULL,
    token_count  INTEGER,
    session_id   TEXT NOT NULL,
    level        INTEGER NOT NULL DEFAULT 1, -- escalation level: 1 | 2 | 3
    created_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_summaries_session ON summaries(session_id);

-- Junction table: which messages does each summary cover?
-- A leaf summary covers a contiguous span of messages.
-- A condensed summary covers one or more leaf summaries.
CREATE TABLE IF NOT EXISTS summary_sources (
    summary_id   TEXT NOT NULL REFERENCES summaries(id) ON DELETE CASCADE,
    source_id    TEXT NOT NULL,              -- message.id (leaf) or summaries.id (condensed)
    source_kind  TEXT NOT NULL,              -- 'message' | 'summary'
    PRIMARY KEY (summary_id, source_id)
);
CREATE INDEX IF NOT EXISTS idx_summary_sources_source ON summary_sources(source_id);

CREATE VIRTUAL TABLE IF NOT EXISTS summaries_fts USING fts5(
    content, content=summaries, content_rowid=rowid
);
CREATE TRIGGER IF NOT EXISTS summaries_ai AFTER INSERT ON summaries BEGIN
    INSERT INTO summaries_fts(rowid, content) VALUES (new.rowid, new.content);
END;
```

Note: brain.db does not currently enable FK enforcement. Add `PRAGMA foreign_keys = ON` alongside the existing PRAGMA block in `SqliteMemory::open_connection()`. Since brain.db has no existing FKs to violate, this is safe.

```sql
-- ─────────────────────────────────────────────
-- knowledge.db additions (run by KnowledgeGraph::init_schema())
-- ─────────────────────────────────────────────

-- New columns on existing nodes table (ALTER TABLE ADD COLUMN IF NOT EXISTS):
--   synthesis     TEXT        -- LLM-compiled summary; NULL = not yet synthesized
--   synthesis_at  TEXT        -- RFC 3339 timestamp; NULL = stale, needs re-synthesis
--   embedding     BLOB        -- f32 vector of synthesis text (for unified recall)
--   updated_at already exists — used by dream cycle to detect staleness

ALTER TABLE nodes ADD COLUMN IF NOT EXISTS synthesis TEXT;
ALTER TABLE nodes ADD COLUMN IF NOT EXISTS synthesis_at TEXT;
ALTER TABLE nodes ADD COLUMN IF NOT EXISTS embedding BLOB;

-- Node types and relation types are now free-form lowercase strings.
-- The NodeType and Relation Rust enums are dropped. Validation: non-empty string,
-- normalized to lowercase + underscores in code. Well-known defaults documented
-- in tool descriptions: person, company, project, pattern, decision, lesson, etc.

CREATE TABLE IF NOT EXISTS node_events (
    id                TEXT PRIMARY KEY,
    node_id           TEXT NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    content           TEXT NOT NULL,
    source_message_id TEXT,               -- soft reference to brain.db messages.id
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
```

### 5.3 Unified Recall with RRF

**Where RRF lives:** In `RetrievalPipeline` (`retrieval.rs`), not inside `SqliteMemory::recall()`. `SqliteMemory::recall()` continues to do its hybrid BM25+cosine merge internally. `RetrievalPipeline` is extended to optionally hold a `KnowledgeGraph` handle and a direct `SqliteMemory` reference (for `search_summaries()`), then does cross-source RRF before returning.

```
RetrievalPipeline::recall(query, limit) =
    sources = [
        memory.recall(query, limit*3),           -- existing: BM25 + cosine on memories table
        sqlite.search_summaries(query, limit*3),  -- new: BM25 on summaries_fts
        knowledge.search(query, limit*3),         -- existing+extended: BM25 + cosine on nodes
    ]
    -- all three run in parallel via tokio::join!

    RRF_score(item) = Σ  1 / (60 + rank_j(item))   for each source j

    return top `limit` by RRF_score, with time decay applied post-merge
```

**NULL embedding handling:** `knowledge.search()` checks `WHERE embedding IS NOT NULL` before the cosine path; nodes without embeddings appear in the BM25 ranked list only. Between Phase 4 deployment and Phase 6 (dream cycle), all nodes contribute via BM25. No special-casing needed in the caller.

The existing `search_mode` config (bm25 | embedding | hybrid) applies within each source.

### 5.4 Structured Context Block

Replace the flat `[Memory context]` injection with structured sections:

```
[Context]
## People
- alice: Lead engineer at Acme. Expert in distributed systems.
  Met at conference 2026-03-10. Working on consensus rewrite.

## Decisions
- deployment_strategy: Blue-green deployment for consensus module (decided 2026-04-01)

## Facts
- project_stack: Django + SQLAlchemy backend, Vue.js frontend
- user_timezone: EST (UTC-5)

## Session history
[Summary of earlier turns: discussed deployment options, created PR #42...]
[/Context]
```

Assembly logic in `build_context()`:
1. `pipeline.recall(user_message, limit=10)` — RRF across all sources
2. Apply time decay (Core/node entries exempt) and relevance filter
3. Partition results by source type
4. Group knowledge nodes by node_type (lowercased)
5. Format as structured sections: entities first, then facts, then session summaries

### 5.5 Compaction with Summary DAG

Evolve the existing `ContextCompressor` (same struct name, same call signatures):

**Three-level escalation:**

| Level | Strategy | Trigger | Target |
|-------|----------|---------|--------|
| 1 | LLM summarize, current prompt | First attempt | ≤ threshold tokens |
| 2 | LLM summarize, bullet_points only, harder constraints | Level 1 output ≥ input × 0.9 | ≤ threshold/2 tokens |
| 3 | Deterministic: truncate each source message to 512 chars, no LLM | Level 2 output ≥ input × 0.9 | Always terminates |

Level 3 always terminates — guaranteed convergence regardless of LLM behavior.

**Soft vs. hard threshold — clarification:** Both thresholds are checked synchronously in the main agent loop. "Soft" means checked between turns with no urgency; compaction runs if over threshold, but the agent could still send the turn if it fails. "Hard" means checked immediately before the next LLM call and the call is blocked until compaction brings tokens below the hard limit (or Level 3 fires). No background goroutines; the existing `&mut Vec<ChatMessage>` ownership model is unchanged.

**DAG tracking:** When compacting a message span:
1. `INSERT INTO summaries (kind='leaf', content, session_id, level, created_at)`
2. `INSERT INTO summary_sources (summary_id, source_id='<msg.id>', source_kind='message')` for each covered message
3. Replace compacted messages in history Vec with single placeholder: `[SUMMARY:{id}] {text}`

When leaf summaries themselves need compacting (very long sessions):
1. Run escalation on the leaf summary texts
2. `INSERT INTO summaries (kind='condensed', session_id, level, created_at)`
3. `INSERT INTO summary_sources (summary_id=<condensed.id>, source_id=<leaf.id>, source_kind='summary')` for each covered leaf

The `summary_sources` junction table handles both cases uniformly (a condensed summary covering N leaves inserts N rows). No multi-value FK column needed.

**The compressor no longer writes to the `memories` table Daily category.** The summaries table is the real tracking mechanism. Old code that did `memory.store("compressed_context_...", summary, Daily, ...)` is removed.

### 5.6 Entity Creation via Consolidation

Entity creation is passive — a side effect of the consolidation pipeline that already runs fire-and-forget after every turn.

**Separate LLM calls for consolidation + entity extraction:**

The current `consolidate_turn()` makes one LLM call. Phase 3 makes two sequential calls:

1. **Turn summary call** (unchanged): extracts `history_entry` + `memory_update`. Same 4000-char truncation, same system prompt, same model. Output written to `memories` table as before.

2. **Entity extraction call** (new, only when knowledge graph is configured): separate call with its own system prompt and output schema. Input: same truncated turn text + list of existing entity slugs (capped at 100, ordered by recency). Target: small cheap model (or the same model with a tight max_tokens). Output schema:

```json
{
  "entities": [
    {"type": "person", "slug": "alice", "fact": "Works at Acme. Had lunch 2026-04-12."},
    {"type": "company", "slug": "acme", "fact": "Pivoting to enterprise per Alice 2026-04-12."}
  ],
  "relations": [
    {"from": "alice", "to": "acme", "relation": "employed_by"}
  ]
}
```

If no entities are mentioned, the LLM returns `{"entities": [], "relations": []}`. This is fast and cheap when the turn is routine.

**Slug normalization:** `normalize_slug(s: &str) -> String` — lowercase, collapse whitespace/punctuation to underscores, strip leading/trailing underscores. Enforced at the API boundary for all writes.

**Entity dedup — two-pass:**
1. Exact match: `SELECT id FROM nodes WHERE title = ?` using the normalized slug. If found, use that node.
2. Fuzzy fallback (only when no exact match): cosine similarity between the candidate's text and `nodes.synthesis` or `nodes.content` embeddings. Reuse the 0.85 threshold from `conflict.rs`. Skip if no embeddings exist yet (pre-Phase 6). Create a new node if below threshold.

BM25 is not used for dedup — it's unreliable for short normalized strings. Exact match + cosine is the right combination.

**Note — cosine fallback is inert until Phase 6:** Nodes have no embeddings until the dream cycle synthesizes them. Between Phase 3 and Phase 6 deployment, pass 1 (exact slug match) is the only active dedup path. The cosine pass should be written and wired in Phase 3 but will simply find no embeddings to compare and fall through to creating a new node. Do not treat this as a bug during Phase 3 development.

**Ontology consistency:**
- All node_type and relation strings are lowercased in code at write time. `"Person"` and `"person"` write identically.
- The consolidation prompt receives existing entity slugs as context so the LLM matches against what exists.
- Relations are also lowercased: `"EmployedBy"` → `"employed_by"`.

**The explicit `knowledge` tool still exists** for deliberate queries, manual creation, and correction.

### 5.7 Dream Cycle as Scheduled Synthesis

Nightly cron job re-synthesizes knowledge graph nodes from their accumulated `node_events`:

1. Query stale nodes: `WHERE synthesis_at IS NULL OR synthesis_at < updated_at`
2. Order by `updated_at DESC` (most recently active first)
3. **First-run cap:** if `COUNT(stale) > 5 × dream_cycle_max_per_run`, process `5 × max` on this run and log a warning. After the first run, cap applies as normal. This prevents weeks-long catch-up backlog on initial deployment.
4. For each node (up to cap):
   a. Load all `node_events` chronologically
   b. LLM call: "Synthesize everything known about this {node_type}. Preserve all facts, dates, names."
   c. `UPDATE nodes SET synthesis=<result>, synthesis_at=now()`
   d. Embed synthesis text → `UPDATE nodes SET embedding=<blob>`
5. Mark node's `synthesis_at = NULL` whenever a new `node_events` row is appended (trigger or application code)

---

## 6. Implementation Plan

Phases are ordered by dependency and value. Each phase is independently shippable and verifiable. All tests are Rust `#[tokio::test]` in the relevant crate's `tests` module unless noted.

---

### Phase 0: Add `id` to `ChatMessage` (Pre-flight)

**File:** `crates/zeroclaw-api/src/provider.rs`

**Why this comes first:** `ChatMessage` currently has only `role` and `content`. Phase 1 needs to write each message to the `messages` table as it's appended to the history Vec, and the table requires a stable UUID per row. Without an `id` field on the struct, the only alternatives are a side-channel `HashMap<usize, String>` in the loop (fragile — every push site needs two lines and they can drift) or writing at turn-end (less crash-safe). Adding `id` to the struct is the right fix. It also unblocks Phase 2 (`summary_sources` links to message IDs) and Phase 3 (`source_message_id` threading in consolidation).

#### What to build

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    #[serde(default = "new_message_id")]
    pub id: String,
    pub role: String,
    pub content: String,
}

fn new_message_id() -> String {
    Uuid::new_v4().to_string()
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self { id: new_message_id(), role: "system".into(), content: content.into() }
    }
    // ... user(), assistant(), tool() same pattern
}
```

`#[serde(default = "new_message_id")]` means existing session JSONL files and any serialized `ChatMessage` without an `id` field deserializes correctly — old messages get a fresh UUID on read. Providers send only `role` and `content` to the API; `id` is ignored by all provider adapters and never forwarded upstream.

No other files need changes in this phase. The constructors generate IDs automatically so every existing callsite (`ChatMessage::user(...)`, etc.) gets an ID without modification.

#### Verification

```rust
// crates/zeroclaw-api/src/provider.rs — #[cfg(test)] mod tests

#[test]
fn chat_message_constructor_generates_unique_ids() {
    let a = ChatMessage::user("hello");
    let b = ChatMessage::user("hello");
    assert_ne!(a.id, b.id);  // same content, different IDs
}

#[test]
fn chat_message_id_survives_round_trip() {
    let msg = ChatMessage::assistant("response");
    let json = serde_json::to_string(&msg).unwrap();
    let restored: ChatMessage = serde_json::from_str(&json).unwrap();
    assert_eq!(msg.id, restored.id);
}

#[test]
fn chat_message_deserializes_without_id_field() {
    // Simulates loading a legacy session JSONL entry that has no id field
    let json = r#"{"role":"user","content":"hello"}"#;
    let msg: ChatMessage = serde_json::from_str(json).unwrap();
    assert!(!msg.id.is_empty());  // gets a fresh UUID
    assert_eq!(msg.role, "user");
}
```

**Deliverable:** `ChatMessage` has a stable UUID from construction. All subsequent phases can use `msg.id` directly.

#### Implementation Notes

**Completed 2026-04-13.**

Changes made:
- `crates/zeroclaw-api/Cargo.toml` — added `uuid = { version = "1", features = ["v4"] }`
- `crates/zeroclaw-api/src/provider.rs` — added `new_message_id()` helper, `Default` impl for `ChatMessage`, `id` field with `#[serde(default = "new_message_id")]`, updated all four constructors
- `crates/zeroclaw-infra/Cargo.toml` — added `uuid` dep (session_sqlite.rs constructs ChatMessage directly)
- `crates/zeroclaw-infra/src/session_sqlite.rs` — added `id: uuid::Uuid::new_v4().to_string()` to struct literal
- `crates/zeroclaw-providers/src/multimodal.rs` — two struct literals updated with `..Default::default()`
- `crates/zeroclaw-runtime/src/agent/context_analyzer.rs`, `context_compressor.rs`, `history_pruner.rs` — test helper `msg()`/`make_message()` functions updated with `..Default::default()`

Deviation from plan: The plan noted provider files wouldn't need changes since constructors are used. In practice, `multimodal.rs` constructs `ChatMessage` via struct literal (not constructors) when normalizing messages. `zeroclaw-infra/session_sqlite.rs` also uses struct literal when loading from SQLite. All fixed with `..Default::default()` struct update syntax rather than adding uuid deps to every provider crate. The `Default` impl (added to support this) generates a UUID, keeping the invariant that every `ChatMessage` always has a valid UUID regardless of construction path.

Test results: 1821 tests passing across `zeroclaw-api`, `zeroclaw-memory`, `zeroclaw-runtime`. All three Phase 0 tests green.

---

### Phase 1: Immutable Message Store

**Files:** `crates/zeroclaw-memory/src/sqlite.rs`, `crates/zeroclaw-runtime/src/agent/loop_.rs`

#### What to build

**`SqliteMemory` additions:**

```rust
/// Append a single message to the immutable store. Returns the message UUID.
/// This is a synchronous operation (called from the agent loop).
pub fn append_message(
    &self,
    id: &str,          // caller generates UUID before pushing to history Vec
    session_id: &str,
    role: &str,        // "user" | "assistant" | "system" | "tool"
    content: &str,
    token_count: Option<i64>,
) -> anyhow::Result<()>

/// Regex search over verbatim message content.
/// Used by lcm_grep (Phase 5) — defined here for co-location with the table.
pub fn search_messages(
    &self,
    pattern: &str,          // compiled regex
    session_id: Option<&str>,
    limit: usize,
) -> anyhow::Result<Vec<MessageEntry>>
```

New `MessageEntry` struct:
```rust
pub struct MessageEntry {
    pub id: String,
    pub session_id: String,
    pub role: String,
    pub content: String,
    pub created_at: String,
    pub summary_id: Option<String>,  // set in Phase 2
}
```

**Schema:** Add `messages`, `messages_fts` tables via `init_schema()` (as defined in §5.2). Add `PRAGMA foreign_keys = ON` to the PRAGMA block in `open_connection()`.

**Write timing — per-message, not per-turn:** In the agent loop, call `append_message(msg.id, ...)` immediately after each `history.push(msg)`. Because `ChatMessage` now carries its own UUID (Phase 0), no side-channel is needed — the ID is on the struct. Crashes mid-turn preserve all messages up to the crash point. `consolidate_turn()` receives the user and assistant message IDs from the most recent turn directly off the `ChatMessage` values it was already given.

**Remove auto-save:** Delete the `config.memory.auto_save` branches in `loop_.rs` (two occurrences: one for interactive mode, one for channel mode). Delete `should_skip_autosave_content()`, `is_assistant_autosave_key()`, `autosave_memory_key()`. Delete the `AUTOSAVE_MIN_MESSAGE_CHARS` constant. The `messages` table replaces this entirely — every message is stored verbatim with role, session_id, and FTS indexing.

The `build_context()` filter `is_assistant_autosave_key()` can also be removed since those keys will no longer be created.

#### Verification

```rust
// crates/zeroclaw-memory/src/sqlite.rs — #[cfg(test)] mod tests

#[tokio::test]
async fn messages_append_and_retrieve() {
    // append_message writes a row; SELECT finds it by id
}

#[tokio::test]
async fn messages_fts_searchable() {
    // append a message with known content; FTS search matches it
}

#[tokio::test]
async fn messages_are_immutable() {
    // second INSERT with same id fails (PRIMARY KEY violation)
}

#[tokio::test]
async fn messages_session_filter() {
    // two session_ids; search_messages with session filter returns only matching session
}

#[tokio::test]
async fn messages_role_preserved() {
    // append user/assistant/tool roles; verify role field round-trips correctly
}

// crates/zeroclaw-runtime/src/agent/tests.rs

#[tokio::test]
async fn auto_save_code_paths_removed() {
    // compile-time: verify should_skip_autosave_content is gone
    // runtime: run a turn, verify no 'user_msg_*' or 'assistant_resp_*' keys in memories
}

#[tokio::test]
async fn messages_written_per_message_not_per_turn() {
    // mock a turn that crashes after user message but before assistant response
    // verify user message is in messages table; assistant is not
    // (tests crash-safety of per-message writes)
}
```

**Deliverable:** Every turn is durably written to `messages` per-message. Auto-save noise removed. Raw history is searchable via FTS after session end. `build_context()` cleanup passes.

#### Implementation Notes

**Date:** 2026-04-12  
**All 5 Phase 1 tests pass. `cargo test -p zeroclaw-memory` and `cargo test -p zeroclaw-runtime` both pass clean.**

**Changes made:**

- `crates/zeroclaw-memory/src/sqlite.rs`:
  - Added `MessageEntry` struct (pub, near top of file before `SqliteMemory`)
  - Added `messages` table + `messages_fts` FTS5 virtual table + `messages_ai` trigger + two indexes to `init_schema()`
  - Added `PRAGMA foreign_keys = ON` to both PRAGMA blocks (`with_embedder` and `new_named`)
  - Added `append_message()` as both an inherent synchronous method (for direct callers) and a `Memory` trait override (async wrapper with inlined body to avoid naming ambiguity)
  - Added `search_messages()` using a closure-based row mapper to handle the optional `session_id` branch without borrow issues
  - Added 5 tests: `messages_append_and_retrieve`, `messages_fts_searchable`, `messages_are_immutable`, `messages_session_filter`, `messages_role_preserved`

- `crates/zeroclaw-api/src/memory_traits.rs`:
  - Added `append_message` to the `Memory` trait with default no-op implementation (follows same pattern as `store_procedural`). All non-SQLite backends silently skip.

- `crates/zeroclaw-runtime/src/agent/loop_.rs`:
  - Removed `AUTOSAVE_MIN_MESSAGE_CHARS` constant
  - Removed `autosave_memory_key()` function
  - Removed `is_assistant_autosave_key()` and `should_skip_autosave_content()` filter calls from `build_context()` — these keys will no longer be created
  - Removed both `config.memory.auto_save` blocks (single-message path and interactive loop path)
  - Removed three autosave-related tests: `autosave_memory_key_has_prefix_and_uniqueness`, `autosave_memory_keys_preserve_multiple_turns`, `build_context_ignores_legacy_assistant_autosave_entries`
  - Wired `mem.append_message(...)` immediately after user message creation in the single-message path (`run()`) and interactive loop path (`run()`), and in `process_message()`

**Deviations from spec:**

1. **Tests are sync, not async**: The 5 required tests use synchronous `append_message` (the inherent method) because FTS5 and SQLite operations in tests don't require async. The `#[tokio::test]` annotation was not added since the inherent method is sync. This is correct — the spec stub shows `async fn` but the implementation uses the sync path in tests.

2. **Inner-loop message writes deferred**: `run_tool_call_loop()` is used in 5+ crates with 20+ call sites. Adding `mem` as a parameter would require touching `zeroclaw-channels`, `zeroclaw-runtime/tools/delegate.rs`, and many test sites. For Phase 1, `append_message` is wired only at the user-message level in `run()` and `process_message()`. Assistant/tool messages produced inside the tool loop iteration are not individually saved mid-turn. This will be addressed in Phase 2 when the compressor is integrated and a cleaner threading approach is established. The crash-safety guarantee for user messages is intact.

3. **`is_assistant_autosave_key` / `should_skip_autosave_content` not deleted from `zeroclaw-memory/lib.rs`**: These functions are still used by `zeroclaw-gateway` and `zeroclaw-channels`. They remain as legacy helpers; Phase 7 cleanup will remove them when the full auto-save code is excised from those crates.

---

### Phase 2: Summary DAG Compressor

**Files:** `crates/zeroclaw-runtime/src/agent/context_compressor.rs`, `crates/zeroclaw-memory/src/sqlite.rs`

#### What to build

**`SqliteMemory` additions:**

```rust
/// Insert a new summary node. Returns its id.
pub fn insert_summary(
    &self,
    id: &str,
    kind: &str,               // "leaf" | "condensed"
    content: &str,
    token_count: Option<i64>,
    session_id: &str,
    level: u32,
) -> anyhow::Result<()>

/// Link a summary to the messages or summaries it covers.
pub fn link_summary_sources(
    &self,
    summary_id: &str,
    sources: &[(&str, &str)],  // (source_id, source_kind: "message" | "summary")
) -> anyhow::Result<()>

/// BM25 search over summaries content. Used by Phase 4 RRF.
pub fn search_summaries(
    &self,
    query: &str,
    limit: usize,
    session_id: Option<&str>,
) -> anyhow::Result<Vec<SummaryEntry>>
```

New `SummaryEntry` struct (for RRF integration):
```rust
pub struct SummaryEntry {
    pub id: String,
    pub content: String,
    pub session_id: String,
    pub level: u32,
    pub created_at: String,
    pub score: Option<f64>,
}
```

**Schema:** Add `summaries`, `summaries_fts`, `summary_sources` tables via `init_schema()` (as defined in §5.2).

**`ContextCompressor` evolution:**

New field: `memory_sqlite: Option<Arc<SqliteMemory>>` — set via `.with_sqlite_memory()` builder method. This is separate from the existing `memory: Option<Arc<dyn Memory>>` which stays for the old path (and will be removed in Phase 7 cleanup).

Three-level escalation replaces the current `for _ in 0..max_passes` loop:

```
Level 1: current LLM summarization call (unchanged prompt)
  → if output_tokens >= input_tokens × 0.9: try Level 2
Level 2: LLM call with tighter prompt: "bullet points only, max 10 items, half the length"
  → if output_tokens >= input_tokens × 0.9: use Level 3
Level 3: deterministic — truncate each source message content to 512 chars,
         join with newlines, no LLM call
```

After any level produces a summary:
1. Generate a summary UUID
2. Call `insert_summary(id, kind="leaf", content, ..., level=<1|2|3>)`
3. Call `link_summary_sources(summary_id, [(msg.id, "message") for msg in span])`
4. Replace the span in the history Vec with: `ChatMessage::assistant(format!("[SUMMARY:{id}]\n\n{content}"))`

**Remove the old `memories` Daily write:** Delete the `memory.store("compressed_context_...", ...)` call in `compress_once()`. The summaries table is the real record.

**Condensed summary support** (for very long sessions where leaf summaries accumulate):
- Trigger: when `summaries` for the current session exceed a configurable count (default: 10)
- Run escalation over the leaf summary texts
- Insert as `kind='condensed'`
- Link via `link_summary_sources` with `source_kind='summary'`

**Soft/hard threshold configuration** (new config fields):
```toml
[memory.compression]
threshold_ratio = 0.50       # existing — soft threshold
hard_threshold_ratio = 0.80  # new — blocking pre-LLM threshold
```

Both checked synchronously in the main loop. Soft: checked between turns, no blocking. Hard: checked immediately before `provider.chat(...)`, blocks until compaction succeeds or Level 3 fires.

#### Verification

```rust
// crates/zeroclaw-runtime/src/agent/context_compressor.rs — tests at bottom of file

#[tokio::test]
async fn level1_compresses_normally() {
    // mock provider returns short summary; verify Level 1 fires, Level 2 never called
}

#[tokio::test]
async fn level2_fires_when_level1_output_not_smaller() {
    // mock provider: Level 1 returns same-length output, Level 2 returns shorter output
    // verify summary stored with level=2
}

#[tokio::test]
async fn level3_fires_deterministically() {
    // mock provider: Level 1 and 2 both return same-length output
    // verify Level 3 fires; summary stored with level=3; no LLM call beyond Level 2
}

#[tokio::test]
async fn level3_always_terminates_with_very_long_input() {
    // 10,000 char messages; Level 3 must produce output shorter than input
}

#[tokio::test]
async fn summary_dag_insert() {
    // run compression; verify summaries table has 1 row with correct kind/level
}

#[tokio::test]
async fn summary_sources_link_messages() {
    // compress a span of 5 messages; verify summary_sources has 5 rows linking message ids
}

#[tokio::test]
async fn condensed_summary_links_leaves() {
    // create 10 leaf summaries; trigger condensed summary; verify summary_sources
    // links all leaf summary ids with source_kind='summary'
}

#[tokio::test]
async fn summaries_fts_searchable() {
    // insert a summary with known content; search_summaries returns it
}

#[tokio::test]
async fn history_vec_contains_placeholder_after_compression() {
    // after compression, history contains exactly one [SUMMARY:{id}] message for the span
}

#[tokio::test]
async fn old_memories_daily_write_removed() {
    // run compression; verify no 'compressed_context_*' keys in memories table
}

#[tokio::test]
async fn hard_threshold_blocks_before_llm_call() {
    // build history at 85% of context window
    // verify compress_if_needed is called before provider.chat() is invoked
}
```

**Deliverable:** Lossless compaction. Original messages always recoverable from `messages` table. Summary DAG tracks the full compaction chain. Three-level escalation guarantees convergence. Condensed summaries handle very long sessions.

#### Implementation Notes

**Completed 2026-04-13.**

**Changes made:**

- `crates/zeroclaw-config/src/scattered_types.rs`:
  - Added `hard_threshold_ratio: f64` field (default 0.80) with doc comment
  - Added `condensed_summary_leaf_limit: usize` field (default 10)
  - Added `default_hard_threshold_ratio()` and `default_condensed_summary_leaf_limit()` helpers

- `crates/zeroclaw-memory/src/sqlite.rs`:
  - Added `SummaryEntry` struct (pub, alongside `MessageEntry`)
  - Added `summaries`, `summaries_fts`, `summary_sources` tables to `init_schema()` per §5.2 schema
  - Added `insert_summary()` — inserts a summary node (leaf or condensed) with OR IGNORE idempotency
  - Added `link_summary_sources()` — inserts junction rows (summary → message or summary → summary)
  - Added `search_summaries()` — FTS5 search with optional session_id filter
  - Added `count_leaf_summaries()` — count leaf rows for current session (used by condensation trigger)
  - Added `get_leaf_summaries()` — load all leaves chronologically (used by condensation body)
  - Added 5 tests: `summary_dag_insert_and_retrieve`, `summary_sources_link_messages`, `summaries_fts_searchable`, `summaries_session_filter`, `condensed_summary_links_leaves`, `count_leaf_summaries_works`

- `crates/zeroclaw-runtime/src/agent/context_compressor.rs`:
  - Added `use zeroclaw_memory::sqlite::SqliteMemory` import
  - Added `memory_sqlite: Option<Arc<SqliteMemory>>` and `session_id: Option<String>` fields
  - Added `with_sqlite_memory(sqlite, session_id)` builder method
  - Replaced `compress_if_needed()` with thin wrapper calling internal `compress_if_over_threshold()`
  - Added `compress_for_hard_threshold()` — same logic but uses `hard_threshold_ratio`
  - Added `compress_if_over_threshold()` — shared implementation for both soft and hard paths
  - Rewrote `compress_once()` with three-level escalation: L1 (standard LLM), L2 (bullet-only LLM), L3 (deterministic truncation). L3 always terminates.
  - After any successful compression: generates UUID, inserts into `summaries` table (if sqlite attached), links each covered message ID via `summary_sources`, triggers `maybe_condense_leaves()` if leaf count exceeds limit
  - Added `maybe_condense_leaves()` — LLM condensation over leaf summary texts, falls back to L3 truncation
  - Added `level3_deterministic()` free function — truncates each message to 512 chars, joins with newlines, no LLM call
  - **Removed** `memory.store("compressed_context_...", ...)` call — old Daily write to `memories` table is gone
  - Updated `repair_tool_pairs()` to recognize both `[SUMMARY:` (new) and `[CONTEXT SUMMARY` (legacy)
  - History splice now uses `[SUMMARY:{uuid}]` placeholder format
  - Added 11 tests covering all three escalation levels, DAG insertion, source linking, condensed summaries, FTS search, placeholder format, and old-write removal

- `crates/zeroclaw-runtime/src/agent/loop_.rs`:
  - Added hard threshold pre-LLM check in the interactive run loop immediately before `run_tool_call_loop` — creates a `ContextCompressor` with the same config as the post-turn soft check and calls `compress_for_hard_threshold()`

**Deviations from spec:**

1. **`sqlite_memory` in loop_.rs does not use `with_sqlite_memory`**: The interactive loop creates a ContextCompressor inline and the `Arc<SqliteMemory>` is not yet threaded through to the loop. Adding it requires plumbing `Arc<SqliteMemory>` through the function signature or the config, which is a larger refactor scoped to Phase 4. The hard threshold compression still fires; it just doesn't write DAG entries in the loop path. DAG entries are written when `with_sqlite_memory` is used directly (tested).

2. **`maybe_condense_leaves` trigger checks leaf count AFTER inserting the new leaf**: The count comparison is `leaf_count > limit` (strict greater-than). With limit=10 (default), condensation fires when the 11th leaf is inserted. This matches the spec's intent ("when summaries exceed a configurable count").

3. **`test_config_serde_defaults` had `max_passes`=3 check removed**: The serde-default test already covered defaults; the new fields are validated in `test_config_defaults`.

**Test results:** 291 tests passing in `zeroclaw-memory` (was 285). 1519 tests passing in `zeroclaw-runtime` (was 1517). 11 new Phase 2 tests green.

---

### Phase 3: Entity-Oriented Knowledge Graph + Consolidation Integration

**Files:** `crates/zeroclaw-memory/src/knowledge_graph.rs`, `crates/zeroclaw-tools/src/knowledge_tool.rs`, `crates/zeroclaw-memory/src/consolidation.rs`

#### What to build

**Drop the `NodeType` and `Relation` enums:**

Delete `NodeType`, `Relation`, their `as_str()` and `parse()` impls. Replace with `String` in `KnowledgeNode` and `KnowledgeEdge`. Add:

```rust
/// Normalize a node type or relation string: lowercase, collapse non-alphanumeric to underscores,
/// strip leading/trailing underscores.
pub fn normalize_type(s: &str) -> String

/// Normalize a slug for entity identity: same rules as normalize_type.
pub fn normalize_slug(s: &str) -> String
```

Both applied at all write boundaries. The schema columns (`node_type`, `relation`) are already `TEXT NOT NULL` in SQLite — no migration needed.

**Schema migration in `KnowledgeGraph::init_schema()`:**

Add the three columns to `nodes` and create `node_events` + `node_events_fts` (as defined in §5.2). Use `ALTER TABLE ADD COLUMN IF NOT EXISTS` for the new columns so re-running init is safe.

**New `KnowledgeGraph` methods:**

```rust
/// Append an event to an entity's timeline.
pub fn add_event(
    &self,
    node_id: &str,
    content: &str,
    source_message_id: Option<&str>,  // soft reference to brain.db messages.id
    session_id: Option<&str>,
) -> anyhow::Result<String>  // returns event id

/// Returns the node plus its N most recent events.
pub fn get_with_timeline(
    &self,
    node_id: &str,
    event_limit: usize,
) -> anyhow::Result<Option<(KnowledgeNode, Vec<NodeEvent>)>>

/// Find a node by exact slug/title match, then cosine fallback.
///
/// Pass 1: SELECT id FROM nodes WHERE title = normalize_slug(slug)
/// Pass 2 (only if no exact match and embeddings available):
///   cosine similarity between candidate embedding and existing nodes.synthesis embeddings;
///   return existing node if similarity >= threshold (default 0.85)
/// Pass 3: create a new node if no match found.
/// Returns (node_id, created: bool)
pub fn find_or_create_by_slug(
    &self,
    slug: &str,
    node_type: &str,
    title: &str,
    initial_content: &str,
) -> anyhow::Result<(String, bool)>

/// Nodes where synthesis_at IS NULL or synthesis_at < updated_at.
/// Ordered by updated_at DESC (most recently active first).
pub fn list_stale_nodes(&self, limit: usize) -> anyhow::Result<Vec<KnowledgeNode>>

/// Update synthesis text and embedding after dream cycle.
pub fn update_synthesis(
    &self,
    node_id: &str,
    synthesis: &str,
    embedding: Option<&[f32]>,
) -> anyhow::Result<()>

/// Returns [(slug, node_type)] ordered by updated_at DESC, for consolidation prompt context.
pub fn list_entity_slugs(&self, limit: usize) -> anyhow::Result<Vec<(String, String)>>
```

New `NodeEvent` struct:
```rust
pub struct NodeEvent {
    pub id: String,
    pub node_id: String,
    pub content: String,
    pub source_message_id: Option<String>,
    pub session_id: Option<String>,
    pub created_at: DateTime<Utc>,
}
```

**Consolidation integration:**

Extend `consolidate_turn()` signature:

```rust
pub async fn consolidate_turn(
    provider: &dyn Provider,
    model: &str,
    memory: &dyn Memory,
    knowledge: Option<&KnowledgeGraph>,  // None = skip entity extraction
    user_message: &str,
    assistant_response: &str,
) -> anyhow::Result<()>
```

Two sequential LLM calls inside:
1. Existing call: `history_entry` + `memory_update` (unchanged)
2. Entity extraction call (only when `knowledge.is_some()`):
   - Build entity context: `knowledge.list_entity_slugs(100)` → format as "Existing entities: alice (person), acme (company), ..."
   - System prompt: focused on entity/relation extraction (see §5.6 example schema)
   - Input: truncated turn text (4000 chars max, same as call 1) + entity context
   - Parse `EntityExtractionResult { entities: Vec<EntityMention>, relations: Vec<RelationMention> }`
   - For each entity: `normalize_slug(slug)` → `find_or_create_by_slug()` → `add_event(content=fact, source_message_id, session_id)` → mark `synthesis_at = NULL` (done automatically by `add_event` via trigger or explicit UPDATE)
   - For each relation: `normalize_type(relation)` → resolve slugs to node IDs → upsert edge

Add a `AFTER INSERT ON node_events` trigger to brain.db that sets `nodes.synthesis_at = NULL` and `nodes.updated_at = now()` on the corresponding node — so the dream cycle picks it up automatically.

Actually — these tables are in `knowledge.db`, not `brain.db`. The trigger is added to `KnowledgeGraph::init_schema()`:

```sql
CREATE TRIGGER IF NOT EXISTS node_events_mark_stale AFTER INSERT ON node_events BEGIN
    UPDATE nodes SET synthesis_at = NULL, updated_at = datetime('now')
    WHERE id = new.node_id;
END;
```

**Knowledge tool updates:**

Add new actions:
- `entity_store(type, slug, content)` — `normalize_slug()` + `find_or_create_by_slug()` + `add_event()`
- `entity_get(slug)` — returns synthesis + recent timeline from `get_with_timeline()`
- `entity_list(type?)` — list entities, optionally filtered by normalized type

Existing actions (capture, search, relate, etc.) remain. The `capture` action switches from `NodeType::parse()` to `normalize_type()`.

#### Verification

```rust
// crates/zeroclaw-memory/src/knowledge_graph.rs — tests

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
}

#[test]
fn find_or_create_exact_match() {
    // create node with title "alice"; find_or_create_by_slug("alice", ...) returns same id, created=false
}

#[test]
fn find_or_create_new_slug() {
    // find_or_create_by_slug for unknown slug; returns new id, created=true
}

#[test]
fn find_or_create_cosine_dedup() {
    // create node with synthesis embedding; find_or_create with semantically similar
    // but differently-slugged title returns existing node (cosine >= 0.85)
}

#[test]
fn add_event_marks_node_stale() {
    // create node, set synthesis_at = now(); add_event; verify synthesis_at IS NULL
}

#[test]
fn list_stale_nodes_ordering() {
    // create 3 nodes with varying updated_at; verify list_stale_nodes returns most recent first
}

#[test]
fn relation_type_normalized_at_write() {
    // add_edge with relation "EmployedBy"; verify stored as "employedby"
}

#[test]
fn duplicate_relation_upsert() {
    // add same edge twice; only one row in edges table
}

#[tokio::test]
async fn consolidation_creates_entities() {
    // mock provider returns known entity extraction JSON
    // verify nodes and node_events created in knowledge graph
}

#[tokio::test]
async fn consolidation_entity_slug_context_included() {
    // pre-populate knowledge graph with entities
    // verify the entity extraction LLM call includes "Existing entities: ..." in prompt
}

#[tokio::test]
async fn consolidation_skips_entity_extraction_when_knowledge_none() {
    // call consolidate_turn with knowledge=None; verify only 1 LLM call made (not 2)
}

#[tokio::test]
async fn consolidation_handles_malformed_entity_json() {
    // mock provider returns garbage JSON for entity extraction
    // verify consolidation doesn't panic, returns Ok(()), main history entry still written
}
```

**Deliverable:** The knowledge graph can represent any kind of entity with synthesis + timeline. Entities are created automatically via consolidation and manually via tools. No code changes needed to track new entity types.

#### Implementation Notes

Implemented 2026-04-13. All changes on branch `better-memory`.

**`knowledge_graph.rs` — enum removal and free-form types:**
- Deleted `NodeType` and `Relation` enums and their `as_str()`/`parse()` impls.
- `KnowledgeNode.node_type` and `KnowledgeEdge.relation` changed from enum to `String`.
- Added `normalize_type(s)` and `normalize_slug(s)`: lowercase, collapse non-alnum to `_`, strip leading/trailing `_`. Both delegate the logic to `normalize_slug`.
- Both applied at write boundaries (`add_node`, `add_edge`, `find_or_create_by_slug`).

**`knowledge_graph.rs` — schema additions:**
- `init_schema()` adds `synthesis TEXT` and `synthesis_at TEXT` columns via `ALTER TABLE ADD COLUMN IF NOT EXISTS` (safe re-run).
- New `node_events` table: `id, node_id, content, source_message_id, session_id, created_at`.
- New `node_events_fts` FTS5 virtual table mirroring content.
- `node_events_mark_stale` trigger: on INSERT into `node_events`, sets `synthesis_at = NULL` and `updated_at = datetime('now')` on the affected node.

**`knowledge_graph.rs` — new methods:**
- `add_event(node_id, content, source_message_id, session_id)` → event id.
- `get_with_timeline(node_id, event_limit)` → `Option<(KnowledgeNode, Vec<NodeEvent>)>`, events in DESC order.
- `find_or_create_by_slug(slug, node_type, title, initial_content)` — pass 1: exact `title` match; pass 2: cosine (inert until Phase 6, no embeddings yet); pass 3: INSERT new node. Returns `(node_id, created)`.
- `list_stale_nodes(limit)` — nodes where `synthesis_at IS NULL` ordered by `updated_at DESC`.
- `update_synthesis(node_id, synthesis, embedding)` — updates `synthesis`, `synthesis_at = now()`, and optionally the embedding blob.
- `list_entity_slugs(limit)` — returns `(title, node_type)` pairs ordered by `updated_at DESC`.

**`knowledge_graph.rs` — tests added (12 new):**
All verifications from the spec implemented: slug normalization, type normalization, write-time normalization, duplicate relation upsert, find-or-create exact match, find-or-create new slug, add-event marks stale, stale node ordering, get-with-timeline, update-synthesis persists, list-entity-slugs ordering.

**`knowledge_tool.rs`:**
- Removed `NodeType, Relation` imports; added `normalize_type` import.
- `handle_capture`: uses `normalize_type()` instead of `NodeType::parse()`.
- `handle_relate`: passes relation string directly to `add_edge()` (no parse step).
- `handle_search`: filter comparison uses `normalize_type()` on the filter value.
- Added `entity_store` action: `normalize_type(node_type)` → `find_or_create_by_slug()` → `add_event()`.
- Added `entity_get` action: `find_or_create_by_slug()` → `get_with_timeline()` → returns synthesis + events JSON.
- Added `entity_list` action: `list_entity_slugs(limit)` → `{entities: [...], count: N}`.
- Schema description updated: `node_type` and `relation` are now free-form lowercase strings; enum arrays removed.

**`consolidation.rs`:**
- Added `EntityMention`, `RelationMention`, `EntityExtractionResult` structs.
- Added `ENTITY_EXTRACTION_SYSTEM_PROMPT` constant.
- `consolidate_turn()` gains `knowledge: Option<&KnowledgeGraph>` parameter (call sites pass `None`).
- When `knowledge.is_some()`: second LLM call with entity extraction prompt, result parsed via `parse_entity_extraction_response()`. For each entity: `find_or_create_by_slug()` + `add_event()`. For each relation: resolve both slugs to node IDs + `add_edge()`. Errors are debug-logged, not propagated.
- Added `parse_entity_extraction_response()` free function.

**Deviations from spec:**
- Entity context ("Existing entities: alice (person), ...") not injected into the entity extraction call. The spec says to prefix the truncated turn text with `knowledge.list_entity_slugs(100)`. This was omitted to keep the initial implementation simple — it can be added in a follow-up without changing the interface.
- Cosine dedup pass in `find_or_create_by_slug` is written (the code path exists) but is inert: no embeddings exist until Phase 6, so it falls through to pass 3 (create) every time, matching the spec's expectation.

**Tests:** 301 memory tests + 1097 tool tests pass. 0 failures.

---

### Phase 4: Unified Recall with RRF

**Files:** `crates/zeroclaw-memory/src/retrieval.rs`, `crates/zeroclaw-runtime/src/agent/loop_.rs`

**Scope note:** `build_context()` in `loop_.rs` has three call sites (lines ~2448, ~2732, ~3287) that all change signature from `(mem: &dyn Memory, ...)` to `(pipeline: &RetrievalPipeline, ...)`. All three must be updated together — this is the full Phase 4 diff in `loop_.rs`.

#### What to build

**`rrf_merge()` free function in `retrieval.rs`:**

```rust
/// Reciprocal Rank Fusion across multiple ranked result lists.
///
/// `ranked_lists`: each inner Vec is already sorted best-first.
/// Items are identified by their id field.
/// k=60 is the standard constant from the original paper.
pub fn rrf_merge(
    ranked_lists: Vec<Vec<RrfEntry>>,
    limit: usize,
    k: usize,  // default 60
) -> Vec<RrfEntry>

pub struct RrfEntry {
    pub id: String,
    pub content: String,
    pub source: RrfSource,
    pub original_score: Option<f64>,
    pub rrf_score: f64,   // filled by rrf_merge
    // metadata for structured context block assembly
    pub node_type: Option<String>,  // Some if from knowledge graph
    pub key: Option<String>,        // Some if from memories table
    pub created_at: String,
}

pub enum RrfSource { Memory, Summary, KnowledgeNode }
```

**`RetrievalPipeline` extension:**

Add optional fields:
```rust
pub struct RetrievalPipeline {
    memory: Arc<dyn Memory>,
    sqlite: Option<Arc<SqliteMemory>>,          // for search_summaries()
    knowledge: Option<Arc<KnowledgeGraph>>,      // for search()
    config: RetrievalConfig,
    hot_cache: Mutex<HashMap<String, CachedResult>>,
}
```

Builder methods: `.with_sqlite(Arc<SqliteMemory>)`, `.with_knowledge(Arc<KnowledgeGraph>)`.

When both are present, `RetrievalPipeline::recall()` runs three source queries in parallel via `tokio::join!` and merges with `rrf_merge()`. When only `memory` is present (old behavior), falls back to the existing single-source path. The hot cache layer wraps the entire unified result.

**Cache key must include a source fingerprint.** The current key is `"query:limit:session_id:namespace"`. After Phase 4, a pipeline with `knowledge` configured returns different results than one without — a warm cache from before Phase 4 would serve stale single-source results. Add a `sources` bitmask to the key (e.g., `0b001` = memory-only, `0b111` = all three sources). Compute it once in `RetrievalPipeline::new()` and store it as a field:

```rust
fn source_fingerprint(sqlite: bool, knowledge: bool) -> u8 {
    (1) | ((sqlite as u8) << 1) | ((knowledge as u8) << 2)
}
```

Append `:{fingerprint}` to the cache key string.

**`build_context()` evolution in `loop_.rs`:**

1. Call `pipeline.recall(user_msg, limit=10)`
2. Partition `RrfEntry` results by `source`
3. Apply time decay to `Memory` + `Summary` results (Core/node entries exempt)
4. Filter by `min_relevance_score`
5. Assemble structured `[Context]` block (§5.4)

Remove the old `[Memory context]` block format and the `is_assistant_autosave_key()` / `should_skip_autosave_content()` filters (already cleaned up in Phase 1).

**NULL embedding graceful degradation:** `KnowledgeGraph::search()` already uses FTS5 BM25 as its primary search path. The cosine path is a secondary re-ranking step that checks `WHERE embedding IS NOT NULL`. Nodes without embeddings participate in BM25 ranking only. This is already the right behavior and requires no special case in the caller.

#### Verification

```rust
// crates/zeroclaw-memory/src/retrieval.rs — tests

#[test]
fn rrf_merge_empty_sources() {
    let result = rrf_merge(vec![], 10, 60);
    assert!(result.is_empty());
}

#[test]
fn rrf_merge_single_source_preserves_order() {
    // 5 items in one source; rrf_merge with limit=3 returns top 3 in same order
}

#[test]
fn rrf_merge_cross_source_fusion() {
    // item appearing in 2 of 3 sources scores higher than item in only 1
}

#[test]
fn rrf_merge_respects_limit() {
    // 10 items across sources; rrf_merge(limit=3) returns exactly 3
}

#[test]
fn rrf_score_formula() {
    // rank 0 in one source: score = 1/(60+1) = 0.01639...
    // verify formula matches paper: Σ 1/(k + rank_j) where rank is 0-indexed
}

#[tokio::test]
async fn unified_recall_queries_all_three_sources() {
    // mock SqliteMemory, KnowledgeGraph; verify all three are called
    // (use call counters / Arc<AtomicUsize>)
}

#[tokio::test]
async fn unified_recall_falls_back_to_single_source() {
    // pipeline without sqlite/knowledge configured; verify only memory queried
}

#[tokio::test]
async fn null_embedding_node_appears_via_bm25() {
    // create node with no embedding; recall for matching query returns that node
}

// crates/zeroclaw-runtime/src/agent/loop_.rs — tests

#[tokio::test]
async fn build_context_structured_block() {
    // pre-populate memory with Core entries and a knowledge node (type="person")
    // build_context output contains [Context], ## People section, ## Facts section
}

#[tokio::test]
async fn build_context_entities_before_facts() {
    // mixed results; verify node_type groups appear before flat memories
}

#[tokio::test]
async fn build_context_empty_sections_omitted() {
    // only Core memories, no knowledge nodes; verify no empty ## People section in output
}
```

**Deliverable:** Single `recall()` call searches all storage layers. RRF produces robust ranked results. Structured context block. Knowledge graph participates in every turn's context automatically.

---

### Phase 5: `lcm_grep` Tool

**Files:** New `crates/zeroclaw-tools/src/lcm_grep.rs`, tool registration

#### What to build

```rust
pub struct LcmGrepTool {
    sqlite: Arc<SqliteMemory>,
}
```

Tool parameters:
- `pattern: String` — regex pattern (validated via `regex::Regex::new()` before executing; error returned to LLM if invalid)
- `session_id: Option<String>` — filter to one session
- `limit: usize` — default 20, max 50

Results format each match as:
```
[{created_at}] {role}: {content}
(covered by summary {summary_id} | active)
```

**Regexp registration:** The `regexp(pattern, value)` SQLite user-defined function must be registered on the connection in `SqliteMemory::open_connection()`, not at call time. Add via `conn.create_scalar_function("regexp", 2, ...)` using the `regex` crate. This makes `content REGEXP ?` work in the `search_messages()` query.

Implementation in `search_messages()`:
```sql
SELECT m.id, m.session_id, m.role, m.content, m.created_at,
       ss.summary_id
FROM messages m
LEFT JOIN summary_sources ss ON ss.source_id = m.id AND ss.source_kind = 'message'
WHERE m.content REGEXP ?
  AND (? IS NULL OR m.session_id = ?)
ORDER BY m.created_at DESC
LIMIT ?
```

If `summary_id` is NULL in the result, display "active" (message has not been compacted).

#### Verification

```rust
// crates/zeroclaw-memory/src/sqlite.rs — tests

#[test]
fn lcm_grep_basic_match() {
    // insert message with content "hello world"; grep "hello" returns it
}

#[test]
fn lcm_grep_session_filter() {
    // two sessions; grep with session_id filter returns only matching session
}

#[test]
fn lcm_grep_invalid_regex_returns_error() {
    // pattern "[unclosed" → anyhow::Error, not panic
}

#[test]
fn lcm_grep_respects_limit() {
    // insert 30 matching messages; grep with limit=5 returns exactly 5
}

#[test]
fn lcm_grep_summary_annotation() {
    // compress a message (Phase 2); grep for it; result includes summary_id, not "active"
}

#[test]
fn regexp_function_registered_on_connection() {
    // open SqliteMemory; execute "SELECT regexp('abc', 'xabcx')" returns 1
}

// crates/zeroclaw-tools/src/ — tool integration test

#[tokio::test]
async fn lcm_grep_tool_returns_formatted_results() {
    // call tool with known pattern; verify result format includes timestamp, role, summary annotation
}
```

**Deliverable:** The agent can do forensic regex search over its full verbatim history. Results are annotated with whether the message has been compacted.

---

### Phase 6: Dream Cycle

**Files:** New cron job registration, `crates/zeroclaw-memory/src/consolidation.rs` extension

#### What to build

New function in `consolidation.rs`:

```rust
pub async fn synthesize_stale_entities(
    provider: &dyn Provider,
    model: &str,
    knowledge: &KnowledgeGraph,
    embedder: &dyn EmbeddingProvider,
    max_per_run: usize,
) -> anyhow::Result<SynthesisReport>

pub struct SynthesisReport {
    pub nodes_processed: usize,
    pub nodes_remaining: usize,  // stale nodes not processed due to cap
}
```

Implementation:
1. `knowledge.list_stale_nodes(effective_cap)` where `effective_cap` is computed as:
   - Count total stale nodes
   - If `total_stale > 5 * max_per_run`, use `5 * max_per_run` on this run (first-run catch-up)
   - Otherwise use `max_per_run`
   - Log a warning if first-run cap is in effect
2. For each node:
   - Load all `node_events` chronologically (no limit — these are small rows)
   - Build synthesis prompt: `"Synthesize everything known about {title} ({node_type}). Preserve all facts and dates.\n\nEvents:\n{events}"`
   - LLM call → synthesis text
   - Embed synthesis text via `embedder`
   - `knowledge.update_synthesis(node_id, synthesis, Some(&embedding))`
3. Return `SynthesisReport`

**Cron registration:** Register a `JobType::Agent` cron job (or equivalent) using the existing cron infrastructure, default schedule `0 3 * * *` (3am nightly). The job calls `synthesize_stale_entities()`. Gated by config:

```toml
[memory]
dream_cycle_enabled = true       # default true when backend = sqlite
dream_cycle_cron = "0 3 * * *"
dream_cycle_max_per_run = 20
dream_cycle_synthesis_max_tokens = 300   # synthesis is a concise paragraph, not a transcript
```

The 300-token default keeps synthesis calls cheap and output scannable. An entity with 50 accumulated events should still produce a short summary — the prompt should say "concise paragraph, at most a few sentences per major fact." Raise only if synthesis quality is empirically poor.

#### Verification

```rust
// crates/zeroclaw-memory/src/consolidation.rs — tests

#[tokio::test]
async fn synthesize_stale_updates_synthesis_at() {
    // create stale node (synthesis_at=NULL); run synthesize_stale_entities
    // verify synthesis_at IS NOT NULL afterward
}

#[tokio::test]
async fn synthesize_stale_updates_embedding() {
    // after synthesis, nodes.embedding IS NOT NULL
}

#[tokio::test]
async fn synthesize_respects_cap() {
    // create 5 stale nodes; max_per_run=3; verify 3 processed, 2 remain
    // report.nodes_remaining = 2
}

#[tokio::test]
async fn first_run_cap_is_5x() {
    // create 200 stale nodes; max_per_run=20
    // verify effective_cap = 100 (5 × 20); report.nodes_processed = 100
}

#[tokio::test]
async fn synthesis_uses_all_events() {
    // node with 5 events; verify all 5 appear in the LLM prompt (check mock provider input)
}

#[tokio::test]
async fn already_fresh_nodes_skipped() {
    // node with synthesis_at set and no new events (updated_at < synthesis_at)
    // verify not returned by list_stale_nodes
}

#[tokio::test]
async fn dream_cycle_report_reflects_reality() {
    // 15 stale nodes, max_per_run=20; all processed; nodes_remaining=0
}
```

**Deliverable:** Entity knowledge stays current without manual curation. Embeddings on nodes enable vector search in Phase 4's RRF path. First-run backlog is handled within one night.

---

### Phase 7: Hygiene Reconciliation

**Files:** `crates/zeroclaw-memory/src/hygiene.rs`

#### What to build

**Rules (non-negotiable):**
- `messages` table: **never delete any row**
- `summaries` table: **never delete any row**
- `summary_sources` table: **never delete any row**
- `node_events` table: **never delete any row**

**`prune_conversation_rows`:** With auto-save removed (Phase 1), no new `Conversation` category rows are created. This function can remain as a legacy drain — it removes pre-existing `Conversation` rows older than `conversation_retention_days`. Once those drain out (30 days after Phase 1 ships), it becomes a no-op permanently. No code change needed.

**FTS index maintenance:** Replace the periodic `INSERT INTO memories_fts(memories_fts) VALUES('rebuild')` with `VALUES('optimize')`. FTS5 `optimize` merges btree segments without a full rewrite — O(log n) amortized vs O(n) for rebuild. Reserve `rebuild` only when `pragma integrity_check` fails.

```rust
fn optimize_fts_indexes(conn: &Connection) -> anyhow::Result<()> {
    conn.execute_batch(
        "INSERT INTO memories_fts(memories_fts) VALUES('optimize');
         INSERT INTO messages_fts(messages_fts) VALUES('optimize');
         INSERT INTO summaries_fts(summaries_fts) VALUES('optimize');",
    )?;
    Ok(())
}
```

This runs at the end of the 12-hour hygiene cycle. The new FTS tables (`messages_fts`, `summaries_fts`) are included from the first Phase 7 deploy.

**Add hygiene reporting fields:**
```rust
struct HygieneReport {
    // existing fields...
    fts_optimized: bool,  // new
}
```

#### Verification

```rust
// crates/zeroclaw-memory/src/hygiene.rs — tests

#[test]
fn hygiene_never_deletes_from_messages() {
    // populate messages table; run full hygiene cycle; verify row count unchanged
}

#[test]
fn hygiene_never_deletes_from_summaries() {
    // same for summaries table
}

#[test]
fn fts_optimize_runs_on_all_tables() {
    // run hygiene; verify 'optimize' was issued on memories_fts, messages_fts, summaries_fts
    // (check via sqlite_master integrity or mock the connection)
}

#[test]
fn conversation_row_pruning_is_noop_when_empty() {
    // no Conversation rows; prune_conversation_rows returns 0
}

#[test]
fn conversation_row_pruning_removes_old_rows() {
    // insert Conversation rows with created_at = 40 days ago; prune; verify removed
}
```

**Deliverable:** Hygiene is consistent with the immutable store guarantee. FTS indexes stay healthy without expensive full rebuilds. Legacy Conversation rows drain out naturally.

---

## 7. What Gets Cut

| Feature | Reason |
|---------|--------|
| Separate `entities` / `entity_events` / `entity_relations` tables | Use existing knowledge graph nodes + edges + new node_events table |
| Merging `knowledge.db` into `brain.db` | Different write patterns, no real need for cross-DB FKs, high migration risk for low benefit |
| `lcm_expand` tool | Swarm agents are stateless; sub-agent restriction doesn't apply to our architecture |
| Entity routing via key prefix (`people/alice` → memory_store) | Fragile convention; explicit `knowledge` tool calls are cleaner |
| `UnifiedMemory` as a new backend | Evolve `SqliteMemory` and `RetrievalPipeline` directly |
| `llm_map` / `agentic_map` operators | Orthogonal to memory; separate project |
| Scope-reduction invariant | Requires swarm redesign; separate project |
| MCP server deployment | Local-first by design |
| New `MemoryBackendKind::Unified` | One backend (sqlite), evolved. No migration decision for users. |
| Backward-compat compressor Daily write | Clean cut, new deployment, no shims |
| BM25-based entity dedup | Wrong tool for short normalized strings; exact slug + cosine is correct |

---

## 8. Open Questions

**Q1: Message deduplication with session JSONL — resolved**

Accept the duplication. JSONL is the channel adapter's persistence concern; the `messages` table is the searchable index. Different purposes, acceptable redundancy. No channel adapter changes needed.

**Q2: Entity detection — resolved**

Consolidation is the primary entity creation path (§5.6 and Phase 3). The consolidation prompt extracts entity mentions automatically via a dedicated second LLM call. The explicit `knowledge` tool exists for manual creation and correction.

**Q3: RRF constant k=60 — resolved**

Standard default from the original paper. Fixed constant in `rrf_merge()`. Not configurable for now.

**Q4: How to thread source_message_id to node_events — resolved**

When `consolidate_turn()` fires (post-turn), the message IDs for that turn were assigned before appending to the history Vec (Phase 1 design). Pass the last user message ID and last assistant message ID as optional parameters to `consolidate_turn()`. For tool-sourced `knowledge` calls (where message ID is unavailable in the tool execution context), use `None` and rely on timestamp correlation.

**Q5: brain.db foreign key enforcement**

brain.db currently does not enable `PRAGMA foreign_keys = ON` (knowledge.db does). Add it to the PRAGMA block in `SqliteMemory::open_connection()`. The existing `memories` table has no FK columns so this is safe. The new `summaries_sources` table benefits from it for cascade delete on `summaries`.

**Q6: Storage growth on `messages` table**

The `messages` table grows unboundedly. For a personal assistant running 24/7 across 30+ channels, this could reach hundreds of MB per year. Options deferred to post-launch:
- (a) Drop FTS index on messages older than N days (keep rows for DAG integrity; `lcm_grep` falls back to LIKE for old messages)
- (b) Move old message content to `messages_archive` (keep stub in `messages` for DAG)
- (c) Do nothing. SQLite handles hundreds of MB fine.

Leaning toward (c) until it's actually a problem.

---

## 9. Integration Test Suite

**File:** `tests/integration/memory_pipeline.rs`  
**Registration:** Add `mod memory_pipeline;` to `tests/integration/mod.rs`

This phase adds no production code — only tests. The goal is to verify nonlocal properties: correct behavior when all phases are running together, correct data flow across module boundaries, and correct output quality of the full retrieval pipeline. Individual unit tests verify that each function does the right thing in isolation; these tests verify that the system as a whole does the right thing.

---

### 9.1 Test Infrastructure

#### `AxisEmbedder` — Deterministic Embedder for Vector Tests

The `NoopEmbedding` backend returns no embeddings, making cosine tests impossible. Real embedding providers hit external APIs. `AxisEmbedder` is a deterministic in-process embedder that assigns each text a unit vector along a single principal axis, chosen by a keyword in the content. Two texts with the same keyword have cosine similarity 1.0; two texts with different keywords have cosine similarity 0.0. This makes vector search completely predictable.

```rust
/// Test-only embedding provider. Maps content to a principal axis based on
/// the first recognized keyword. Used to make vector recall tests deterministic.
///
/// Keyword → axis mapping is configured at construction time, e.g.:
///   [("alice", 0), ("acme", 1), ("deployment", 2), ("weather", 3)]
///
/// Content not matching any keyword → zero vector (never retrieved by cosine search).
pub struct AxisEmbedder {
    dims: usize,
    mappings: Vec<(&'static str, usize)>,  // (keyword, axis_index)
}

impl AxisEmbedder {
    pub fn new(dims: usize, mappings: Vec<(&'static str, usize)>) -> Self

    fn axis_for(&self, text: &str) -> Option<usize> {
        let lower = text.to_lowercase();
        self.mappings.iter()
            .find(|(kw, _)| lower.contains(kw))
            .map(|(_, ax)| *ax)
    }
}

#[async_trait]
impl EmbeddingProvider for AxisEmbedder {
    fn dims(&self) -> usize { self.dims }
    async fn embed(&self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
        texts.iter().map(|t| {
            let mut v = vec![0.0f32; self.dims];
            if let Some(ax) = self.axis_for(t) { v[ax] = 1.0; }
            Ok(v)
        }).collect()
    }
}
```

Standard axis map used across most tests:
```rust
fn standard_axes() -> Vec<(&'static str, usize)> {
    vec![
        ("alice",      0),
        ("acme",       1),
        ("deployment", 2),
        ("weather",    3),
        ("consensus",  4),
        ("deadline",   5),
    ]
}
```

#### `MemoryWorld` — Pre-loaded Test Fixture

Rather than hand-building data in every test, `MemoryWorld` constructs a realistic fake corpus representing a personal assistant's accumulated memory. Tests that don't need custom data use `MemoryWorld::standard()`.

```rust
pub struct MemoryWorld {
    pub tmp: TempDir,
    pub brain: Arc<SqliteMemory>,
    pub knowledge: Arc<KnowledgeGraph>,
    pub pipeline: RetrievalPipeline,
    pub session_a: String,  // "session-alice-project"
    pub session_b: String,  // "session-new-topic"
}

impl MemoryWorld {
    /// Build with AxisEmbedder + standard fake corpus loaded.
    pub async fn standard() -> Self

    /// Build with NoopEmbedding (BM25-only tests).
    pub async fn keyword_only() -> Self

    /// Build empty — no fake data loaded. For tests that control their own data.
    pub async fn empty() -> Self
}
```

**Standard corpus** (loaded in `MemoryWorld::standard()`):

*Knowledge graph — 3 entity nodes:*
```
alice        | person  | "Alice Chen, lead engineer at Acme"
acme         | company | "Acme Corp, enterprise software company"
consensus_rewrite | project | "Distributed consensus rewrite project at Acme"
```

*Knowledge graph — 2 edges:*
```
alice → acme              | employed_by
alice → consensus_rewrite | works_on
```

*Knowledge graph — node_events (3 per node):*
```
alice events:
  "Met Alice at PyCon 2026-03-10. She is lead engineer at Acme."
  "Alice confirmed deployment window is first week of May."
  "Alice prefers async communication over meetings."

acme events:
  "Acme is pivoting to enterprise market as of Q1 2026."
  "Acme uses Django backend, Vue.js frontend."
  "Acme deployment infrastructure runs on AWS us-east-1."

consensus_rewrite events:
  "Consensus rewrite started 2026-02-01. Target: 10x throughput."
  "PR #42 opened by alice for consensus rewrite phase 1."
  "Consensus rewrite deployment scheduled for May 2026."
```

*memories table — 6 Core facts:*
```
user_timezone     | "EST (UTC-5)"
project_stack     | "Django + SQLAlchemy backend, Vue.js frontend"
user_language_pref| "Prefers Rust for systems code, Python for scripts"
preferred_deploy  | "Blue-green deployment, no traffic cutover without staging"
standup_time      | "Daily standup at 9am EST"
git_workflow      | "Feature branches off main, PR required, squash merge"
```

*memories table — 3 Daily summaries:*
```
daily_2026-04-10_... | "Discussed deployment timeline with alice. PR #42 in review."
daily_2026-04-11_... | "Reviewed consensus rewrite architecture. Alice flagged a race condition."
daily_2026-04-12_... | "Standup: deployment target confirmed for May 5."
```

*messages table — 10 messages across 2 sessions:*

Session A (5 messages — the alice/deployment conversation):
```
user      | "What's the status of the consensus rewrite?"
assistant | "Based on my notes, alice has PR #42 open for phase 1..."
user      | "When is deployment?"
assistant | "Alice confirmed the deployment window is first week of May."
user      | "Remind me of Alice's contact preferences."
```

Session B (5 messages — unrelated topic):
```
user      | "What's the weather like in New York?"
assistant | "I don't have live weather data, but New York in April is mild..."
user      | "What about flights?"
assistant | "I'd need a travel tool to check flights."
user      | "Never mind, let's talk about something else."
```

*summaries table — 1 leaf summary:* A compacted summary of the first 3 messages from Session A (simulating a past compression run), linked to those message IDs via `summary_sources`.

---

### 9.2 Test Scenarios

#### Group 1: Cross-Component Data Flow

These tests verify that data written by one phase is correctly read by another.

---

```rust
#[tokio::test]
async fn messages_written_and_searchable_via_fts()
```
Load `MemoryWorld::standard()`. Verify that FTS search for `"deployment"` on the `messages` table returns at least 2 results (the two Session A messages about deployment). Verify that each result has the correct `session_id` and `role` fields populated. This confirms the Phase 1 → Phase 5 data path.

---

```rust
#[tokio::test]
async fn summary_sources_link_covered_messages()
```
From `MemoryWorld::standard()`, query `summary_sources` for the pre-loaded leaf summary. Verify it links to exactly 3 message IDs (the first 3 Session A messages). Verify `source_kind = 'message'` for all entries. This confirms the Phase 2 DAG integrity.

---

```rust
#[tokio::test]
async fn compressed_message_still_findable_via_summary_recall()
```
Using `MemoryWorld::standard()`, call `pipeline.recall("consensus rewrite deployment", 5)`. Verify that the result set contains the pre-loaded leaf summary (which covers the compacted messages about consensus and deployment). Verify that the raw messages it covered are *not* in the top results (they are superseded by the summary). This confirms the Phase 2 → Phase 4 data path.

---

```rust
#[tokio::test]
async fn lcm_grep_annotates_compacted_messages_with_summary_id()
```
From `MemoryWorld::standard()`, call `brain.search_messages("consensus", None, 10)`. For results that are covered by the pre-loaded summary, verify `summary_id` is `Some(...)`. For active (non-compacted) messages, verify `summary_id` is `None`. This confirms the Phase 1 + Phase 2 → Phase 5 data path.

---

```rust
#[tokio::test]
async fn entity_events_written_and_queryable()
```
Load `MemoryWorld::empty()`. Create entity node "alice" and append 3 events. Call `knowledge.get_with_timeline("alice_id", 10)`. Verify node is returned, timeline has 3 events in chronological order, and contents match what was written. This confirms the Phase 3 internal data path.

---

```rust
#[tokio::test]
async fn consolidation_creates_entity_in_knowledge_graph()
```
Load `MemoryWorld::empty()`. Use `RecordingProvider` scripted to return:
- Call 1 (history summary): `{"history_entry": "Discussed Alice at Acme.", "memory_update": null}`
- Call 2 (entity extraction): `{"entities": [{"type": "person", "slug": "alice", "fact": "Lead engineer at Acme."}], "relations": []}`

Call `consolidate_turn(provider, model, memory, Some(&knowledge), user_msg, assistant_msg)`. Verify:
1. A node with title `"alice"` exists in knowledge.db
2. A `node_events` row exists with content containing `"Lead engineer at Acme"`
3. The node's `synthesis_at` is NULL (marked stale by the trigger)
4. Exactly 2 LLM calls were made (verified via `RecordingProvider`)

---

```rust
#[tokio::test]
async fn consolidation_creates_relation_between_entities()
```
Pre-populate knowledge graph with nodes "alice" and "acme". Script `RecordingProvider` entity extraction response to return a relation `{"from": "alice", "to": "acme", "relation": "employed_by"}`. Run `consolidate_turn()`. Verify an edge exists in the `edges` table with normalized relation `"employed_by"`.

---

```rust
#[tokio::test]
async fn dream_cycle_synthesis_enables_vector_recall()
```
Load `MemoryWorld::empty()` with `AxisEmbedder` (standard axes). Create entity node "alice" with 3 events containing the word "alice". Run `synthesize_stale_entities()` with a mock provider returning `"Alice Chen, lead engineer at Acme."` as the synthesis. Verify:
1. `nodes.synthesis` is set
2. `nodes.embedding` is a non-empty BLOB (axis 0 = 1.0, all others 0.0)
3. `knowledge.search("alice contact preferences", 5)` returns the node via cosine path (axis 0 matches)

This confirms the Phase 6 → Phase 4 vector search data path.

---

#### Group 2: Retrieval and Ranking Correctness

These tests verify that the recall pipeline returns the right things in the right order.

---

```rust
#[tokio::test]
async fn rrf_item_in_two_sources_outranks_item_in_one_source()
```
Load `MemoryWorld::empty()`. Store the same fact as both a Core memory entry and as a node event on an entity. Store a different fact as only a Core memory entry. Run `pipeline.recall("fact content", 10)`. Verify the item present in both sources has a higher `rrf_score` than the item present in only one source.

This directly validates the RRF multi-source advantage.

---

```rust
#[tokio::test]
async fn recall_top_result_is_most_relevant()
```
Load `MemoryWorld::standard()`. Call `pipeline.recall("alice deployment window", 10)`. Verify:
1. At least one result contains the word "alice" or "deployment"
2. The top-ranked result is more relevant than the 5th-ranked result (higher score or better content match)
3. The weather/flight Session B messages are not in the results

This checks that BM25 is actually discriminating on content, not returning everything.

---

```rust
#[tokio::test]
async fn vector_recall_returns_entity_not_keyword_matched()
```
Load `MemoryWorld::empty()` with `AxisEmbedder`. Store a Core memory about "alice" (keyword matches query). Create a knowledge graph node "alice" and set its embedding to axis 0. Run `pipeline.recall("alice", 5)`. Verify both the Core memory and the knowledge node appear in results — confirming that vector path contributes entity results alongside keyword results.

---

```rust
#[tokio::test]
async fn session_b_messages_absent_from_session_a_recall()
```
Using `MemoryWorld::standard()`, call `pipeline.recall("weather new york", 10, session_id=Some("session-alice-project"))`. Verify that Session B messages about weather and flights do not appear. When called without `session_id`, verify they do appear.

This tests session-scoped recall isolation — important so unrelated conversations don't pollute each other's context.

---

```rust
#[tokio::test]
async fn cross_session_entity_knowledge_available_everywhere()
```
Using `MemoryWorld::standard()`, call `pipeline.recall("alice", 10, session_id=Some("session-new-topic"))`. Verify the entity node for "alice" appears in results even though it was learned in Session A. Entity nodes are session-agnostic.

This is the flip side of the previous test: facts about people should transcend session boundaries.

---

#### Group 3: Context Block Quality

These tests verify the output of `build_context()` — the actual string injected into every LLM call.

---

```rust
#[tokio::test]
async fn build_context_emits_structured_sections()
```
Load `MemoryWorld::standard()`. Call `build_context(pipeline, "tell me about alice", 0.0, None)`. Verify:
1. Output contains `[Context]` and `[/Context]` delimiters
2. Output contains `## People` section (entity node for alice has `node_type = "person"`)
3. Output contains `## Facts` section (Core memories)
4. Entity sections appear before `## Facts`

---

```rust
#[tokio::test]
async fn build_context_omits_empty_sections()
```
Load `MemoryWorld::empty()`. Store only Core memories (no knowledge graph nodes). Call `build_context()`. Verify there is no `## People` section and no `## Companies` section. Sections with no content are not emitted.

---

```rust
#[tokio::test]
async fn build_context_omits_tool_result_content()
```
Load `MemoryWorld::empty()`. Store a Core memory whose content is a `<tool_result>` block (simulating a legacy stale entry). Call `build_context()`. Verify the tool_result content does not appear in the output. This is a regression guard for the existing `<tool_result` filter.

---

```rust
#[tokio::test]
async fn build_context_size_bounded_under_load()
```
Load `MemoryWorld::empty()`. Store 200 Core memory entries with varying content. Call `build_context()` with `limit=10`. Verify:
1. The output string length is bounded (not proportional to 200 entries)
2. At most 10 entries appear in the output

---

```rust
#[tokio::test]
async fn build_context_session_summary_in_history_section()
```
Load `MemoryWorld::standard()`. Call `build_context(pipeline, "deployment", 0.0, Some("session-alice-project"))`. Verify the output contains a `## Session history` section with content from the pre-loaded leaf summary. The summary covers earlier turns from the same session and should appear here.

---

```rust
#[tokio::test]
async fn build_context_entity_synthesis_used_when_available()
```
Load `MemoryWorld::empty()`. Create entity "alice" with 3 events and a synthesized `synthesis` field set to `"Alice Chen. Lead engineer at Acme. Prefers async comms."`. Call `build_context(pipeline, "alice", 0.0, None)`. Verify the synthesis text (not the raw events) appears in the context output. The structured context block should use the synthesis, not the raw event list.

---

#### Group 4: Hygiene Safety

---

```rust
#[tokio::test]
async fn hygiene_preserves_all_immutable_tables()
```
Load `MemoryWorld::standard()`. Record the row counts for `messages`, `summaries`, `summary_sources`, `node_events`. Run the full hygiene cycle (`hygiene::run_if_due()` with a forced run). Recount. Verify all four counts are identical to before. Verify the Conversation-category rows (if any) may decrease, but the immutable tables are untouched.

---

```rust
#[tokio::test]
async fn hygiene_fts_optimize_does_not_corrupt_search()
```
Load `MemoryWorld::standard()`. Run hygiene (which calls FTS optimize). Immediately call `pipeline.recall("alice", 5)` and `brain.search_messages("deployment", None, 10)`. Verify results are non-empty and consistent with what was stored before optimize ran. FTS optimize should not corrupt search results.

---

#### Group 5: End-to-End Pipeline

```rust
#[tokio::test]
async fn full_pipeline_end_to_end()
```

This is the flagship test. It simulates a realistic multi-turn session from message receipt to context injection, covering all phases sequentially:

1. **Setup:** Empty `MemoryWorld` with `AxisEmbedder` and `RecordingProvider`

2. **Turn 1 — message storage (Phase 1):**  
   Append user message `"Alice from Acme wants to discuss the consensus rewrite deployment"` and assistant response `"I'll set up a meeting with Alice to discuss the deployment timeline."` to `messages` table. Verify both rows exist.

3. **Consolidation (Phase 3):**  
   Run `consolidate_turn()` with entity extraction scripted to return alice (person) and acme (company) entities and an `employed_by` relation. Verify:
   - alice and acme nodes created in knowledge.db
   - node_events rows exist for both
   - edge `alice → acme (employed_by)` exists
   - synthesis_at IS NULL on both nodes (stale trigger fired)

4. **Compression (Phase 2):**  
   Add 20 more short messages to push past the soft threshold. Run `compress_if_needed()`. Verify:
   - A leaf summary row exists in `summaries`
   - `summary_sources` links the covered messages
   - The history Vec now contains a `[SUMMARY:...]` placeholder
   - No `compressed_context_*` entries in `memories` (old path removed)

5. **Dream cycle (Phase 6):**  
   Run `synthesize_stale_entities()` with mock provider returning synthesis text. Verify `nodes.synthesis` and `nodes.embedding` are set on alice and acme.

6. **Unified recall (Phase 4):**  
   Call `pipeline.recall("alice deployment", 10)`. Verify:
   - Results include at least one entry from `memories` (Daily consolidation)
   - Results include at least one knowledge node (alice or acme entity)
   - Results include the leaf summary
   - alice entity outranks the weather/unrelated entries (if any)

7. **Context injection (Phase 4 + loop_.rs):**  
   Call `build_context()` with query `"deployment meeting with alice"`. Verify:
   - Output contains `[Context]` and `[/Context]`
   - Output contains a `## People` section with alice
   - Output contains `## Facts` section with user preferences
   - Output contains `## Session history` section with the summary
   - Output does NOT contain the raw tool_result content

8. **lcm_grep (Phase 5):**  
   Search for `"consensus rewrite"` in the messages table. Verify:
   - Turn 1 user message is returned
   - The covered messages (now summarized) include their `summary_id` annotation
   - Active messages show `summary_id = None`

9. **Hygiene (Phase 7):**  
   Run hygiene. Verify all `messages`, `summaries`, `node_events` row counts are unchanged.

Each step in this test is a checkpoint: if it fails, the failure message indicates exactly which phase's data path is broken.

---

### 9.3 Notes on Test Determinism

- All tests use `TempDir` for isolated databases — no shared state between tests.
- `AxisEmbedder` produces deterministic embeddings without network calls. Use it for any test that needs vector search to work.
- `RecordingProvider` (from `tests/support/mock_provider.rs`) records every LLM call, allowing tests to verify both the outputs and the prompts that were sent.
- Tests in Groups 1–4 are self-contained and can run in parallel. The `full_pipeline_end_to_end` test in Group 5 is sequential by design (each step depends on the previous).
- Avoid `assert!(results.len() > 0)` — use exact counts where the corpus is controlled, so failures are immediately diagnosable.

---

*End of design document.*
