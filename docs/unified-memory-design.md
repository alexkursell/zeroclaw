# Unified Memory Architecture — Revised Design

**Status:** Proposal v2 — revised after code review  
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

The `Memory` trait (`memory_traits.rs`) with 5 backends. The SQLite backend is the recommended default:

- **Schema:** `memories` table with id, key, content, category, embedding, timestamps, session_id, namespace, importance, superseded_by
- **Search:** Multi-stage pipeline (cache → FTS5 BM25 → cosine similarity → LIKE fallback) with hybrid merge using weighted linear combination (0.7 vector + 0.3 keyword)
- **Categories:** Core (evergreen, no decay), Daily (7-day half-life), Conversation (7-day half-life), Custom
- **Auto-save:** Every user message ≥20 chars stored as Conversation category (removed by this proposal — replaced by the `messages` table)
- **Consolidation:** Per-turn LLM extraction (`consolidation.rs`) writes Daily history entries and Core facts, with semantic conflict resolution (0.85 cosine threshold)
- **Conflict resolution:** Detects semantic duplicates, marks old entries `superseded_by` (logical delete)
- **Decay:** Exponential time decay applied at recall time, Core entries exempt
- **Importance:** Heuristic scoring (base by category + keyword boost) blended into final score
- **Hygiene:** Automated 12-hour job archives old daily memory files + session JSONLs (30d), purges archived files (90d), hard-deletes Conversation rows from brain.db (30d), prunes audit entries (90d)

### 1.3 Knowledge Graph (knowledge.db)

Separate SQLite database (`knowledge_graph.rs`, ~600 lines):

- **Node types:** Pattern, Decision, Lesson, Expert, Technology
- **Relations:** Uses, Replaces, Extends, AuthoredBy, AppliesTo
- **Search:** FTS5 on title/content/tags
- **Traversal:** Recursive CTE for multi-hop graph queries
- **Tool:** `knowledge` tool with actions: capture, search, relate, suggest, expert_find, lessons_extract, graph_stats
- **Cap:** max_nodes limit enforced at insert time

### 1.4 Session Persistence

Two mechanisms:
- **Channel sessions** (`session_store.rs`): Append-only JSONL, one `ChatMessage` per line. Already an immutable log. Not searchable.
- **Interactive CLI** (`history.rs`): Full JSON dump/restore of the history Vec. Overwritten each save. Loses data when `trim_history` runs.

### 1.5 Context Compressor

Multi-pass compression (`context_compressor.rs`, 764 lines):

- Triggered at 50% context window usage
- Protects first 3 + last 4 messages
- Pass 1: Trim tool results in non-protected messages
- Pass 2-4: LLM summarization of middle section (up to 3 passes)
- Fallback: Raw truncation if LLM fails/times out
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

The agent can manually call `memory_recall` and `knowledge` tools, but the automatic context injection — the thing that makes every response contextually aware — is blind to two of the three storage systems.

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
| Soft/hard thresholds | Async compaction below hard threshold, blocking above. Already partially there with the trigger ratio. |
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

2. **Two SQLite files, separate concerns.** `brain.db` owns flat facts and session history. `knowledge.db` owns the entity graph. They stay separate — different write patterns (bulk synthesis writes during dream cycle vs. frequent small writes during conversation), separate WAL contention, and no real need for cross-database foreign keys. RRF recall queries both in parallel via `tokio::join!` and merges in Rust. The `node_events.source_message_id` link to `messages` is a soft reference (UUID string lookup), not a hard FK constraint — temporal correlation (matching timestamps) works as fallback.

3. **Immutable source of truth.** Raw messages are never modified or deleted. Summaries, entities, and facts are derived caches that can always be regenerated.

4. **Search everything by default.** `build_context()` should query all storage layers. The agent shouldn't have to know which tool to call to find information it previously learned.

5. **Deterministic convergence.** The compressor must always terminate. Three escalation levels, the last one requires no LLM.

6. **Backward compatible.** Existing `brain.db` files gain new tables via `CREATE TABLE IF NOT EXISTS`. Existing `memories` data untouched. `knowledge.db` gains new columns and tables additively. Old config works, new config unlocks new features.

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
└── summaries          (NEW — DAG of compressed summary nodes)


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
- `messages` + `summaries` tables added to brain.db for lossless session history
- `node_events` table added to knowledge.db for entity timelines
- `nodes` table gains optional `synthesis`, `synthesis_at`, `embedding` columns
- Node types and relation types become free-form strings instead of hardcoded enums — the LLM can create any type it needs (person, company, restaurant, book, etc.) without code changes

**What didn't change:**
- `memories` table structure and data — untouched
- `knowledge.db` stays as its own file — no migration needed
- `Memory` trait interface — all backends still valid
- Auto-save, consolidation, hygiene — continue working

### 5.2 Schema Additions

```sql
-- ─────────────────────────────────────────────
-- Immutable session history
-- ─────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS messages (
    id           TEXT PRIMARY KEY,
    session_id   TEXT NOT NULL,
    role         TEXT NOT NULL,              -- user | assistant | system | tool
    content      TEXT NOT NULL,              -- verbatim, never modified
    token_count  INTEGER,
    created_at   TEXT NOT NULL,              -- RFC 3339
    summary_id   TEXT REFERENCES summaries(id)
                     DEFERRABLE INITIALLY DEFERRED
                                            -- NULL = active; non-NULL = covered by this summary
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
CREATE INDEX IF NOT EXISTS idx_messages_summary ON messages(summary_id);
CREATE INDEX IF NOT EXISTS idx_messages_created ON messages(created_at);

CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    content, content=messages, content_rowid=rowid
);
CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts(rowid, content) VALUES (new.rowid, new.content);
END;

-- ─────────────────────────────────────────────
-- Summary DAG (compaction tracking)
-- ─────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS summaries (
    id           TEXT PRIMARY KEY,
    kind         TEXT NOT NULL,              -- leaf | condensed
    content      TEXT NOT NULL,
    token_count  INTEGER,
    parent_id    TEXT REFERENCES summaries(id),
    session_id   TEXT NOT NULL,
    level        INTEGER NOT NULL DEFAULT 1, -- escalation level: 1 | 2 | 3
    created_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_summaries_session ON summaries(session_id);
CREATE INDEX IF NOT EXISTS idx_summaries_parent  ON summaries(parent_id);

```

**knowledge.db additions** (run by `KnowledgeGraph::init_schema()`):

```sql
-- ─────────────────────────────────────────────
-- Knowledge graph extensions (additive changes to existing schema)
-- ─────────────────────────────────────────────

-- New columns on existing nodes table (ALTER TABLE IF NOT EXISTS):
--   synthesis     TEXT        -- LLM-compiled summary; NULL = not yet synthesized
--   synthesis_at  TEXT        -- RFC 3339; NULL = stale, needs re-synthesis
--   embedding     BLOB        -- f32 vector of synthesis text (for unified recall)

-- Node types and relation types become free-form strings (see Phase 3).
-- The nodes.node_type and edges.relation columns are already TEXT in SQLite.
-- The Rust enums (NodeType, Relation) are replaced with validated strings.
-- Well-known defaults (person, company, project, pattern, decision, lesson, etc.)
-- are documented in tool descriptions, not enforced in code.

CREATE TABLE IF NOT EXISTS node_events (
    id                TEXT PRIMARY KEY,
    node_id           TEXT NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    content           TEXT NOT NULL,
    source_message_id TEXT,               -- soft reference to brain.db messages.id (UUID lookup)
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

Replace the weighted linear merge in the recall path with Reciprocal Rank Fusion across all sources:

```
recall(query, limit) =
    sources = [
        search_memories(query, limit*3),      -- existing: BM25 + cosine on memories table
        search_nodes(query, limit*3),          -- extended: BM25 + cosine on nodes table
        search_summaries(query, limit*3),      -- new: BM25 on summaries table
    ]

    RRF_score(item) = Σ  1 / (k + rank_j(item))   for each source j
                     where k = 60

    return top `limit` by RRF_score
```

This replaces the `0.7 * vector + 0.3 * keyword` merge in `vector::hybrid_merge`. RRF operates on rank position, not raw scores — no normalization needed, no weight tuning. The per-source search (BM25, cosine, hybrid) still runs as-is within each source; RRF merges across sources.

The three sources query two different SQLite files (`brain.db` and `knowledge.db`) in parallel via `tokio::join!`. Results are converted to a common `MemoryEntry` and merged in Rust. No cross-database SQL needed.

The existing `search_mode` config (bm25 | embedding | hybrid) still applies within each source.

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
1. `recall(user_message, limit=10)` — RRF across all sources
2. Apply time decay (Core/node entries exempt) and relevance filter
3. Partition results by source type
4. Group knowledge nodes by node_type
5. Format as structured sections: entities first, then facts, then session summaries

### 5.5 Compaction with Summary DAG

Evolve the existing `ContextCompressor` (don't replace it — same call sites, same signatures):

**Three-level escalation (formalizing the existing multi-pass):**

| Level | Strategy | Trigger |
|-------|----------|---------|
| 1 | LLM summarize, preserve_details mode, target T tokens | First attempt |
| 2 | LLM summarize, bullet_points mode, target T/2 tokens | Level 1 output ≥ input |
| 3 | Deterministic truncation to 512 chars, no LLM | Level 2 output ≥ input |

Level 3 always terminates — guaranteed convergence.

**DAG tracking:** When compacting a message span:
1. `INSERT INTO summaries (kind='leaf', content=S, session_id, level)`
2. `UPDATE messages SET summary_id=<id> WHERE id IN (<compacted>)`
3. Replace compacted messages in history Vec with summary placeholder

When leaf summaries themselves need compacting (very long sessions):
1. Run escalation on the leaf summary texts
2. `INSERT INTO summaries (kind='condensed', parent_id=<leaf_ids>)`
3. Update leaf rows' parent_id

**Thresholds:** Keep existing `threshold_ratio = 0.50` as soft (async between turns). Add `hard_threshold_ratio = 0.80` (blocking before next LLM call). Below soft threshold, zero overhead.

### 5.6 Entity Creation via Consolidation

The agent won't reliably call `knowledge entity_store(...)` during natural conversation. Entity creation must be passive — a side effect of the consolidation pipeline that already runs after every turn.

**Per-turn consolidation (evolved):** `consolidate_turn()` currently extracts `history_entry` (Daily) and `memory_update` (Core). Extend the output schema:

```json
{
  "history_entry": "Discussed Acme's enterprise pivot with Alice over lunch.",
  "memory_update": "Alice works at Acme. Acme is pivoting to enterprise.",
  "entities": [
    {"type": "person", "slug": "alice", "fact": "Works at Acme. Had lunch 2026-04-12."},
    {"type": "company", "slug": "acme", "fact": "Pivoting to enterprise per Alice 2026-04-12."}
  ],
  "relations": [
    {"from": "alice", "to": "acme", "relation": "employed_by"}
  ]
}
```

The consolidation code then:
1. For each entity: normalize the slug (lowercase, underscores, strip whitespace), search existing nodes by title/slug similarity (BM25 on `nodes_fts`), and either match an existing node or create a new one. Append a `node_events` row with the fact. Mark the node's `synthesis_at = NULL` (stale).
2. For each relation: resolve slugs to node IDs, upsert edge.

**Ontology consistency** is maintained by:
- **Slug normalization** in code: `"Alice from Acme"` → `"alice_from_acme"`. Enforced at the API boundary, not by convention.
- **Existing-entity context in the prompt**: The consolidation prompt receives the current list of entity slugs and types (e.g., `"Existing entities: alice (person), acme (company), zeroclaw (project)"`). This nudges the LLM to match against what exists rather than inventing new slugs. Capped at ~100 entities in the prompt to avoid context bloat; ordered by recency.
- **Dedup at write time**: Before creating a new node, BM25 search for similar titles. If similarity exceeds a threshold (reusing the 0.85 cosine pattern from `conflict.rs`), merge into the existing node instead.
- **Type normalization**: Lowercase in code. `"Person"` and `"person"` are the same.

**The explicit `knowledge` tool still exists** for deliberate queries, manual creation, and correction. But the passive consolidation path is what builds the graph over time.

### 5.7 Dream Cycle as Scheduled Synthesis

The dream cycle is the complement to consolidation: consolidation creates rough entity nodes with individual facts, the dream cycle periodically synthesizes them into coherent summaries.

**Nightly cron job:** Re-synthesize knowledge graph nodes from their accumulated `node_events`:

1. Query nodes where `synthesis_at IS NULL OR synthesis_at < updated_at`, ordered by update recency
2. For each (capped at 20 per run):
   a. Load all `node_events` chronologically
   b. LLM call: "Synthesize everything known about this {node_type}. Preserve all facts, dates, names."
   c. `UPDATE nodes SET synthesis=<result>, synthesis_at=now()`
   d. Embed synthesis text → `UPDATE nodes SET embedding=<blob>`

This uses the existing cron infrastructure (`cron/scheduler.rs` with `JobType::Agent`). The consolidation module gains a `synthesize_stale_entities()` function called by the cron job.

---

## 6. Implementation Plan

Phases are ordered by dependency and value. Each phase is independently shippable.

### Phase 1: Immutable Message Store

**Files:** `loop_.rs`, `sqlite.rs`, `history.rs`

**What:**
- Add `messages` and `messages_fts` tables to `init_schema()`
- Add `append_message(&self, role, content, session_id, token_count) -> String` to `SqliteMemory`
- In the agent loop, after each complete turn, write all messages to the `messages` table
- The `messages` table is the durable record; the in-memory `history` Vec remains the fast path
- FTS5 trigger keeps `messages_fts` in sync automatically
- Return the message UUID so downstream code (entity events in Phase 3) can link to it
- **Remove auto-save.** Set `auto_save = false` when the `messages` table is active. The `messages` table is a strictly better replacement — it stores every message verbatim with role, session_id, timestamps, and FTS indexing, versus auto-save which only stored user messages ≥20 chars as Conversation entries with UUID keys that the recall pipeline then had to filter out (`should_skip_autosave_content()`, `is_assistant_autosave_key()`). Those filters can also be removed.

**Key design decision:** The `messages` table supplements, not replaces, the session JSONL files. Channel sessions keep their append-only JSONL (it's the channel adapter's persistence). The `messages` table is the searchable index over all sessions. This avoids changing any channel adapter code.

**Where each message now lives and why:**

| Store | What it holds | Purpose |
|-------|--------------|---------|
| Session JSONL / session.json | Verbatim messages | Channel adapter persistence (unchanged) |
| `messages` table | Verbatim messages | Searchable immutable history, provenance links |
| `memories` table (via consolidation) | LLM-extracted summaries + facts | Cross-session knowledge (Daily + Core) |
| `node_events` table (via consolidation) | Entity-specific facts | Entity timelines |

The JSONL / `messages` duplication is accepted (different purposes, different systems — see Q1). The old auto-save Conversation entries are eliminated — they were a weaker version of what `messages` now provides.

**Deliverable:** Every turn is durably written to `messages`. Auto-save noise removed. Raw history is searchable after session end.

---

### Phase 2: Summary DAG Compressor

**Files:** `context_compressor.rs`, `sqlite.rs`

**What:**
- Add `summaries` table to `init_schema()`
- Evolve `ContextCompressor` (keep same struct name and call signatures):
  - Add three-level escalation (formalize existing multi-pass)
  - On compaction: INSERT summary into `summaries` table, UPDATE `messages.summary_id` for covered messages
  - Replace compacted messages in history Vec with placeholder `[SUMMARY:{id}] {text}`
  - Add condensed summary support: when leaf summaries accumulate, compact them too
- Add soft/hard threshold distinction (async vs blocking compaction)

**Key change from current behavior:** The `memories` table Daily entry still gets written (backward compat), but the `summaries` table is the real tracking mechanism with parent pointers and provenance.

**Deliverable:** Lossless compaction. Original messages always recoverable from `messages` table. Summary DAG tracks the compaction chain. Three-level escalation guarantees convergence.

---

### Phase 3: Entity-Oriented Knowledge Graph + Consolidation Integration

**Files:** `knowledge_graph.rs`, `knowledge_tool.rs`, `consolidation.rs`

**What — schema changes:**
- **Replace `NodeType` and `Relation` enums with free-form strings.** The `nodes.node_type` and `edges.relation` columns are already `TEXT` in SQLite — the enum is only enforced at the Rust API boundary. Drop `NodeType::parse()` / `Relation::parse()` validation and accept any string. Provide well-known defaults (`person`, `company`, `project`, `pattern`, `decision`, `lesson`, `expert`, `technology`, etc.) in tool descriptions and system prompts so the LLM knows what conventions exist, but don't enforce them in code. This means the agent can track restaurants, books, medical providers, or anything else without a code change.
- Add columns to `nodes`: `synthesis TEXT`, `synthesis_at TEXT`, `embedding BLOB`
- Add `node_events` table with soft FK to `nodes` (in knowledge.db) and soft reference to `messages` (in brain.db)

**What — KnowledgeGraph API extensions:**
- `add_event(node_id, content, source_message_id, session_id)` — append to timeline
- `get_with_timeline(node_id, event_limit)` — returns node + recent events
- `list_stale_nodes()` — nodes needing re-synthesis
- `update_synthesis(node_id, synthesis_text, embedding)` — called by dream cycle
- `find_or_create_by_slug(slug, node_type, title)` — dedup-aware upsert: BM25 search for similar slugs/titles, return existing node if similarity > threshold, else create. Used by consolidation.
- `list_entity_slugs(limit)` — returns `[(slug, node_type)]` ordered by recency, for injection into the consolidation prompt

**What — consolidation integration (the primary entity creation path):**
- Extend `ConsolidationResult` to include `entities: Vec<EntityMention>` and `relations: Vec<RelationMention>`
- Extend the consolidation LLM prompt (see §5.6) to extract entities and relations
- After extraction, for each entity: normalize slug → `find_or_create_by_slug()` → `add_event()` with the fact. Mark `synthesis_at = NULL`.
- After extraction, for each relation: resolve slugs to node IDs → upsert edge.
- The consolidation prompt receives existing entity slugs as context (capped at ~100, ordered by recency) so the LLM matches against what exists.

**What — knowledge tool extensions (the explicit/manual path):**
- `entity_store(type, slug, content)` — creates/updates entity node + appends event. `type` is any string.
- `entity_get(slug)` — returns synthesis + recent timeline
- `entity_list(type?)` — list entities, optionally filtered by type
- Existing knowledge tool actions (capture, search, relate, etc.) continue to work unchanged

**Key decisions:**

*Consolidation is the primary entity creation path.* The agent won't reliably call tools to create entities during natural conversation. Instead, the per-turn consolidation pipeline (which already runs fire-and-forget after every turn) extracts entity mentions and writes them to the knowledge graph automatically. The explicit `knowledge` tool exists for deliberate queries, manual creation, and correction — but the passive consolidation path is what builds the graph over time.

*No hardcoded types.* Node types and relation types are free-form strings, not enums. The schema is `TEXT NOT NULL`, validated only for non-empty. Well-known types are conventions, not constraints. This is critical for a personal assistant — you can't anticipate every kind of entity a user will care about.

*Ontology consistency via normalization + dedup.* Slug normalization (lowercase, underscores) is enforced in code. Existing-entity context in the consolidation prompt nudges the LLM toward consistency. Dedup at write time catches what the LLM misses.

**Deliverable:** The knowledge graph can represent any kind of entity with synthesis + timeline. Entities are created automatically via consolidation and manually via tools. No code changes needed to track new entity types. Existing knowledge graph features unchanged.

---

### Phase 4: Unified Recall with RRF

**Files:** `sqlite.rs`, `retrieval.rs`, `vector.rs`, `loop_.rs`

**What:**
- Add `rrf_merge(ranked_lists, k=60) -> Vec<MemoryEntry>` to `retrieval.rs`
- In `SqliteMemory::recall()`, query three sources in parallel:
  1. `memories` table (existing search pipeline)
  2. `nodes` table (BM25 on `nodes_fts` + cosine on `nodes.embedding`)
  3. `summaries` table (BM25 on summary content, filtered by session)
- Merge results with RRF instead of weighted linear combination
- Convert knowledge graph `SearchResult` to `MemoryEntry` for uniform handling
- Refactor `build_context()` to produce structured `[Context]` block with sections

**Deliverable:** Single `recall()` call searches all storage layers. RRF produces robust ranked results. Context injection is structured and includes entity knowledge.

---

### Phase 5: `lcm_grep` Tool

**Files:** New `crates/zeroclaw-tools/src/lcm_grep.rs`, tool registration

**What:**
- Regex search over the verbatim `messages` table
- Parameters: `pattern` (regex), `session_id` (optional filter), `limit` (default 20, max 50)
- Results annotated with covering summary_id (or "active" if NULL)
- Register SQLite `regexp` function via rusqlite
- Available to all agents (no sub-agent restriction)

**Why no `lcm_expand`:** The swarm tool spawns stateless agents — they get a single prompt, not a full tool-calling loop with history. There's no meaningful sub-agent context to restrict `lcm_expand` within. Instead, `lcm_grep` returns enough context (matching messages with their summary annotations) for the agent to decide if it needs more detail, and it can grep again with a narrower pattern.

**Deliverable:** The agent can do forensic regex search over its full verbatim history. "What did I say about the deployment on Tuesday?" becomes answerable even after compaction.

---

### Phase 6: Dream Cycle

**Files:** New cron job registration, `consolidation.rs` extension

**What:**
- Add `synthesize_stale_entities()` function to `consolidation.rs`
- Register automatic cron job (default `0 3 * * *`) when memory backend is sqlite
- Job logic:
  1. Query stale entity nodes (synthesis_at IS NULL OR < updated_at)
  2. For each (capped at 20/run): load events → LLM synthesis → update node
  3. Embed synthesis text for vector search
- Config: `dream_cycle_enabled`, `dream_cycle_cron`, `dream_cycle_max_per_run`, `synthesis_max_words`

**Deliverable:** Entity knowledge stays current without manual curation. After a few days of use, entity nodes contain rich, accurate summaries.

---

### Phase 7: Hygiene Reconciliation

**Files:** `hygiene.rs`

**What — align existing hygiene with the new storage model:**

- **`prune_conversation_rows`**: With auto-save removed (Phase 1), no new Conversation rows are created. This pruner becomes a legacy cleanup — it can drain remaining old Conversation rows and then become a no-op. No change needed; it naturally winds down.
- **`purge_session_archives`**: Session JSONL archives are still deletable — the `messages` table is now the authoritative searchable history, so archived JSONLs are redundant. No change needed.
- **`messages` table**: The hygiene job must **never delete from `messages`**. This is the immutable store. The lossless guarantee depends on it.
- **`summaries` table**: Never deleted. Summaries are small relative to raw messages.
- **`node_events` table**: Never deleted. Entity event timelines are the input to dream cycle synthesis.

**What — new: FTS index maintenance:**

- After legacy Conversation rows are pruned, their `memories_fts` entries become orphaned. Add a periodic `INSERT INTO memories_fts(memories_fts) VALUES('rebuild')` to the hygiene job to keep the FTS index consistent.

**What — storage growth on `messages`:**

The `messages` table grows unboundedly. For a personal assistant running 24/7 across 30+ channels, this could reach hundreds of MB per year. Options for future work (not this proposal):
- (a) Drop the FTS index on messages older than N days (keep rows, remove from `messages_fts`). Saves index size; `lcm_grep` falls back to slower `LIKE` for old messages.
- (b) Move old message content to a `messages_archive` table (keep id/session_id/summary_id/created_at stub in `messages` for DAG integrity). `lcm_grep` gains `include_archived` flag.
- (c) Do nothing. SQLite handles hundreds of MB fine. Revisit when it's actually a problem.

Leaning toward (c) for now. Hundreds of MB is not a problem for SQLite on modern hardware. This can be revisited if real-world usage shows otherwise.

**What — decay:**

Time decay (`decay.rs`) still applies to non-Core entries at recall time. With auto-save removed, the Conversation category effectively goes dormant. Decay continues to apply to:
- Daily entries from consolidation (session summaries) — correct, older summaries should rank lower
- Daily entries from the context compressor — correct, same reason
- Core entries — exempt, as before

No changes needed to the decay system.

**Deliverable:** Hygiene is consistent with the immutable store guarantee. No new deletion paths. Legacy cleanup winds down naturally.

---

## 7. What Gets Cut

| Feature | Reason |
|---------|--------|
| Separate `entities` / `entity_events` / `entity_relations` tables | Use existing knowledge graph nodes + edges + new node_events table |
| Merging `knowledge.db` into `brain.db` | Different write patterns, no real need for cross-DB FKs, high migration risk for low benefit |
| `lcm_expand` tool | Swarm agents are stateless; sub-agent restriction doesn't apply to our architecture |
| Entity routing via key prefix (`people/alice` → memory_store) | Fragile convention; explicit `knowledge` tool calls are cleaner |
| `UnifiedMemory` as a new backend | Evolve `SqliteMemory` directly; don't create a parallel backend |
| `llm_map` / `agentic_map` operators | Orthogonal to memory; separate project |
| Scope-reduction invariant | Requires swarm redesign; separate project |
| MCP server deployment | Local-first by design |
| New `MemoryBackendKind::Unified` | One backend (sqlite), evolved. No migration decision for users. |

---

## 8. Open Questions

**Q1: Message deduplication with session JSONL**

The `messages` table and channel session JSONL files will contain overlapping data. Options:
- (a) Accept the duplication. JSONL is the channel adapter's concern; `messages` table is the search index. Different purposes, acceptable redundancy.
- (b) Replace JSONL session store with reads from `messages` table. This changes the `SessionBackend` trait and all channel adapters.

Option (a) is much simpler and avoids a large refactor. The JSONL files are tiny relative to the SQLite data. Leaning toward (a).

**Q2: Entity detection — resolved**

Consolidation is the primary entity creation path (see §5.6 and Phase 3). The consolidation prompt extracts entity mentions automatically. The explicit `knowledge` tool exists for manual creation and correction. This is option (c) from the original proposal — hybrid automatic + manual — which is the right answer now that consolidation owns entity extraction.

**Q3: RRF constant k=60**

Standard default from the original paper. Probably fine as a fixed constant. If we want to tune it later, it's a single constant in one function. Not worth making configurable now.

**Q4: How to thread source_message_id to node_events**

When the agent calls the `knowledge` tool during a conversation turn, we need to link the resulting `node_events` row back to the `messages` row. The tool execution context (`execute_one_tool`) doesn't carry session state.

Options:
- (a) Thread a "current message ID" through the tool execution context. Requires adding a parameter to the tool execution path.
- (b) Use a task-local (tokio) to carry the current message ID. Set it in the agent loop before tool execution.
- (c) Accept NULL for `source_message_id` when called from tools. The temporal correlation (matching timestamps) is good enough for most uses.

Leaning toward (b) for cleanliness, with (c) as fallback.

---

*End of design document.*
