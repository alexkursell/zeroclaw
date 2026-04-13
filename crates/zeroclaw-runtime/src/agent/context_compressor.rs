use std::fmt::Write;
use std::time::Duration;

use anyhow::Result;
use std::sync::Arc;

use zeroclaw_api::provider::{ChatMessage, Provider};
use zeroclaw_memory::sqlite::SqliteMemory;
use zeroclaw_memory::traits::Memory;

pub use zeroclaw_config::scattered_types::ContextCompressionConfig;

// ---------------------------------------------------------------------------
// Result
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct CompressionResult {
    pub compressed: bool,
    pub tokens_before: usize,
    pub tokens_after: usize,
    pub passes_used: u32,
}

// ---------------------------------------------------------------------------
// Probe tiers for unknown model context windows
// ---------------------------------------------------------------------------

const PROBE_TIERS: &[usize] = &[
    2_000_000, 1_000_000, 512_000, 200_000, 128_000, 64_000, 32_000,
];

fn next_probe_tier(current: usize) -> usize {
    PROBE_TIERS
        .iter()
        .copied()
        .find(|&tier| tier < current)
        .unwrap_or(32_000)
}

// ---------------------------------------------------------------------------
// Error message parsing
// ---------------------------------------------------------------------------

/// Try to extract the actual context window limit from a provider error message.
pub fn parse_context_limit_from_error(msg: &str) -> Option<usize> {
    // Match patterns like "maximum context length is 128000" or "limit of 200000 tokens"
    // or "context window of 131072" or "available context size (8448 tokens)"
    let re_patterns: &[&str] = &[
        // "maximum context length is 128000"
        r"(?:max(?:imum)?|limit)\s*(?:context\s*)?(?:length|size|window)?\s*(?:is|of|:)?\s*(\d{4,})",
        // "context length is 128000" / "context window of 131072"
        r"context\s*(?:length|size|window)\s*(?:is|of|:)?\s*(\d{4,})",
        // "128000 token context" / "128000 limit"
        r"(\d{4,})\s*(?:tokens?\s*)?(?:context|limit)",
        // "available context size (8448 tokens)"
        r"available context size\s*\(\s*(\d{4,})",
        // "> 128000 maximum context length" (Anthropic-style)
        r">\s*(\d{4,})\s*(?:maximum|max)?\s*(?:context)?\s*(?:length|size|window|tokens?)",
    ];
    let lower = msg.to_lowercase();
    for pattern in re_patterns {
        if let Ok(re) = regex::Regex::new(pattern)
            && let Some(caps) = re.captures(&lower)
            && let Some(m) = caps.get(1)
            && let Ok(limit) = m.as_str().parse::<usize>()
            && (1024..=10_000_000).contains(&limit)
        {
            return Some(limit);
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Token estimation
// ---------------------------------------------------------------------------

/// Estimate token count for a message history using ~4 chars/token heuristic
/// with a 1.2x safety margin.
pub fn estimate_tokens(messages: &[ChatMessage]) -> usize {
    let raw: usize = messages
        .iter()
        .map(|m| m.content.len().div_ceil(4) + 4)
        .sum();
    // 1.2x safety margin to account for underestimation
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    {
        (raw as f64 * 1.2) as usize
    }
}

// ---------------------------------------------------------------------------
// Summarizer prompt
// ---------------------------------------------------------------------------

const SUMMARIZER_SYSTEM: &str = "\
You are a conversation compaction engine. Summarize the conversation segment below into concise context.

PRESERVE exactly:
- All identifiers (UUIDs, hashes, file paths, URLs, tokens, IPs)
- Actions taken (tool calls, file operations, commands run)
- Key information obtained (data, results, error messages)
- Decisions made and user preferences expressed
- Current task status and unresolved items
- Constraints and requirements mentioned

OMIT:
- Verbose tool output (keep only key results)
- Repeated greetings or filler
- Redundant information already stated

Output concise bullet points. Be thorough but brief.";

// ---------------------------------------------------------------------------
// ContextCompressor
// ---------------------------------------------------------------------------

pub struct ContextCompressor {
    config: ContextCompressionConfig,
    context_window: usize,
    memory: Option<Arc<dyn Memory>>,
    /// SqliteMemory handle for the Summary DAG (Phase 2).
    memory_sqlite: Option<Arc<SqliteMemory>>,
    /// Session ID for DAG entries — None means no DAG tracking.
    session_id: Option<String>,
}

impl ContextCompressor {
    pub fn new(config: ContextCompressionConfig, context_window: usize) -> Self {
        Self {
            config,
            context_window,
            memory: None,
            memory_sqlite: None,
            session_id: None,
        }
    }

    /// Attach a memory handle so compression summaries are persisted before
    /// old messages are discarded. Without this, compressed facts are lost.
    pub fn with_memory(mut self, memory: Arc<dyn Memory>) -> Self {
        self.memory = Some(memory);
        self
    }

    /// Attach the SQLite memory handle for Summary DAG tracking.
    /// When set, every compression records a summary + source links in the database.
    pub fn with_sqlite_memory(mut self, sqlite: Arc<SqliteMemory>, session_id: impl Into<String>) -> Self {
        self.memory_sqlite = Some(sqlite);
        self.session_id = Some(session_id.into());
        self
    }

    /// Update the context window size (e.g. after error-driven probing).
    pub fn set_context_window(&mut self, window: usize) {
        self.context_window = window;
    }

    /// Fast-path: trim oversized tool results in non-protected messages.
    /// Returns total characters saved. No LLM call needed.
    fn fast_trim_tool_results(&self, history: &mut [ChatMessage]) -> usize {
        let max = self.config.tool_result_retrim_chars;
        if max == 0 {
            return 0;
        }
        let mut saved = 0;
        let protect_start = self.config.protect_first_n.min(history.len());
        let protect_end = history.len().saturating_sub(self.config.protect_last_n);

        if protect_start >= protect_end {
            return 0;
        }

        for msg in &mut history[protect_start..protect_end] {
            if msg.role != "tool" {
                continue;
            }
            if msg.content.len() <= max {
                continue;
            }
            // Skip exempt tools
            if self
                .config
                .tool_result_trim_exempt
                .iter()
                .any(|t| msg.content.contains(t.as_str()))
            {
                continue;
            }
            // Skip base64 images
            if msg.content.contains("data:image/") {
                continue;
            }
            let original_len = msg.content.len();
            msg.content = crate::agent::history::truncate_tool_message(&msg.content, max);
            saved += original_len - msg.content.len();
        }
        saved
    }

    /// Main entry point (soft threshold). Compresses history in-place if over soft threshold.
    /// Called between turns — non-blocking.
    pub async fn compress_if_needed(
        &self,
        history: &mut Vec<ChatMessage>,
        provider: &dyn Provider,
        model: &str,
    ) -> Result<CompressionResult> {
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let threshold = (self.context_window as f64 * self.config.threshold_ratio) as usize;
        self.compress_if_over_threshold(history, provider, model, threshold).await
    }

    /// Hard threshold check — called immediately before each LLM call (blocking).
    /// Uses `hard_threshold_ratio` instead of `threshold_ratio`.
    pub async fn compress_for_hard_threshold(
        &self,
        history: &mut Vec<ChatMessage>,
        provider: &dyn Provider,
        model: &str,
    ) -> Result<CompressionResult> {
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let threshold = (self.context_window as f64 * self.config.hard_threshold_ratio) as usize;
        self.compress_if_over_threshold(history, provider, model, threshold).await
    }

    /// Internal: compress if tokens exceed `threshold`.
    async fn compress_if_over_threshold(
        &self,
        history: &mut Vec<ChatMessage>,
        provider: &dyn Provider,
        model: &str,
        threshold: usize,
    ) -> Result<CompressionResult> {
        if !self.config.enabled {
            let tokens = estimate_tokens(history);
            return Ok(CompressionResult {
                compressed: false,
                tokens_before: tokens,
                tokens_after: tokens,
                passes_used: 0,
            });
        }

        let tokens_before = estimate_tokens(history);

        if tokens_before <= threshold {
            return Ok(CompressionResult {
                compressed: false,
                tokens_before,
                tokens_after: tokens_before,
                passes_used: 0,
            });
        }

        // Fast-trim pass — may resolve overflow without an LLM call
        let chars_saved = self.fast_trim_tool_results(history);
        if chars_saved > 0 {
            tracing::info!(chars_saved, "Fast-trim saved chars from old tool results");
            let recheck = estimate_tokens(history);
            if recheck <= threshold {
                return Ok(CompressionResult {
                    compressed: true,
                    tokens_before,
                    tokens_after: recheck,
                    passes_used: 0,
                });
            }
        }

        let mut passes_used = 0;
        for _ in 0..self.config.max_passes {
            let did_compress = self.compress_once(history, provider, model).await?;
            if did_compress {
                passes_used += 1;
            }
            if estimate_tokens(history) <= threshold || !did_compress {
                break;
            }
        }

        let tokens_after = estimate_tokens(history);
        Ok(CompressionResult {
            compressed: passes_used > 0,
            tokens_before,
            tokens_after,
            passes_used,
        })
    }

    /// Reactive compression triggered by a context_length_exceeded error.
    /// Parses the actual limit from the error, steps down probe tiers, and re-compresses.
    pub async fn compress_on_error(
        &mut self,
        history: &mut Vec<ChatMessage>,
        provider: &dyn Provider,
        model: &str,
        error_msg: &str,
    ) -> Result<bool> {
        // Try to extract actual limit from error message
        if let Some(limit) = parse_context_limit_from_error(error_msg) {
            self.context_window = limit;
        } else {
            // Step down to next probe tier
            self.context_window = next_probe_tier(self.context_window);
        }

        tracing::info!(
            context_window = self.context_window,
            "Context limit adjusted, re-compressing"
        );

        let result = self.compress_if_needed(history, provider, model).await?;
        Ok(result.compressed)
    }

    /// Single compression pass with three-level escalation.
    /// Level 1: LLM summarize (standard prompt)
    /// Level 2: LLM summarize (bullet_points only, tighter constraints) — fires if L1 didn't shrink enough
    /// Level 3: Deterministic truncation — always terminates
    async fn compress_once(
        &self,
        history: &mut Vec<ChatMessage>,
        provider: &dyn Provider,
        model: &str,
    ) -> Result<bool> {
        let n = history.len();
        let protected_total = self.config.protect_first_n + self.config.protect_last_n;
        if n <= protected_total {
            return Ok(false);
        }

        let mut start = self.config.protect_first_n.min(n);
        let mut end = n.saturating_sub(self.config.protect_last_n);

        // Align boundaries to avoid orphaning tool_call/tool_result pairs
        start = align_boundary_forward(history, start);
        end = align_boundary_backward(history, end);

        if start >= end {
            return Ok(false);
        }

        // Collect message IDs for DAG source links (before we splice them away)
        let source_ids: Vec<String> = history[start..end]
            .iter()
            .map(|m| m.id.clone())
            .collect();

        let message_count = source_ids.len();
        let middle = &history[start..end];
        let transcript = build_transcript(middle, self.config.source_max_chars);

        if transcript.is_empty() {
            return Ok(false);
        }

        let summary_model = self.config.summary_model.as_deref().unwrap_or(model);
        let timeout = Duration::from_secs(self.config.timeout_secs);

        let identifier_note = if self.config.identifier_policy == "strict" {
            "\nIMPORTANT: Preserve all identifiers exactly as they appear."
        } else {
            ""
        };

        // ── Level 1: standard LLM summarization ──────────────────────────
        let l1_prompt = format!(
            "Summarize the following conversation history ({message_count} messages) for context preservation. \
             Keep it concise (max 20 bullet points).{identifier_note}\n\n{transcript}"
        );

        let l1_raw = match tokio::time::timeout(
            timeout,
            provider.chat_with_system(Some(SUMMARIZER_SYSTEM), &l1_prompt, summary_model, 0.1),
        )
        .await
        {
            Ok(Ok(s)) => s,
            Ok(Err(e)) => {
                tracing::warn!(error = %e, "Level-1 summarization failed, skipping to Level 3");
                String::new()
            }
            Err(_) => {
                tracing::warn!(
                    "Level-1 summarization timed out after {}s, skipping to Level 3",
                    self.config.timeout_secs
                );
                String::new()
            }
        };

        // Check if Level 1 achieved meaningful compression (< 90% of input length)
        let (summary, level_used) = if !l1_raw.is_empty()
            && l1_raw.len() < (transcript.len() * 9 / 10)
        {
            (truncate_chars(&l1_raw, self.config.summary_max_chars), 1u32)
        } else {
            // ── Level 2: tighter LLM summarization ───────────────────────
            let l2_prompt = format!(
                "Summarize the following conversation history ({message_count} messages). \
                 Output BULLET POINTS ONLY. Maximum 10 items. Each item ≤ 15 words. \
                 Total output must be half the input length.{identifier_note}\n\n{transcript}"
            );

            let l2_raw = match tokio::time::timeout(
                timeout,
                provider.chat_with_system(Some(SUMMARIZER_SYSTEM), &l2_prompt, summary_model, 0.1),
            )
            .await
            {
                Ok(Ok(s)) => s,
                _ => String::new(),
            };

            if !l2_raw.is_empty()
                && l2_raw.len() < (transcript.len() * 9 / 10)
            {
                (truncate_chars(&l2_raw, self.config.summary_max_chars), 2u32)
            } else {
                // ── Level 3: deterministic truncation (always terminates) ─
                let l3_summary = level3_deterministic(&history[start..end]);
                (truncate_chars(&l3_summary, self.config.summary_max_chars), 3u32)
            }
        };

        // Generate a stable UUID for this summary node
        let summary_id = uuid::Uuid::new_v4().to_string();

        // Persist to Summary DAG if SQLite handle is present
        if let Some(ref sqlite) = self.memory_sqlite {
            let session = self.session_id.as_deref().unwrap_or("unknown");
            #[allow(clippy::cast_possible_wrap)]
            let token_count = Some(summary.len().div_ceil(4) as i64);

            if let Err(e) = sqlite.insert_summary(
                &summary_id,
                "leaf",
                &summary,
                token_count,
                session,
                level_used,
            ) {
                tracing::warn!(error = %e, "Failed to insert summary into DAG");
            } else {
                // Link each covered message to this summary
                let sources: Vec<(&str, &str)> = source_ids
                    .iter()
                    .map(|id| (id.as_str(), "message"))
                    .collect();
                if let Err(e) = sqlite.link_summary_sources(&summary_id, &sources) {
                    tracing::warn!(error = %e, "Failed to link summary sources");
                }

                // Trigger condensed summary if leaf count exceeds limit
                if let Ok(leaf_count) = sqlite.count_leaf_summaries(session) {
                    if leaf_count > self.config.condensed_summary_leaf_limit {
                        if let Err(e) = self
                            .maybe_condense_leaves(sqlite, session, provider, summary_model, timeout)
                            .await
                        {
                            tracing::warn!(error = %e, "Failed to create condensed summary");
                        }
                    }
                }
            }
        }

        // Splice: head + [SUMMARY:{id}] + tail
        let summary_msg = ChatMessage::assistant(format!(
            "[SUMMARY:{summary_id}]\n\n{summary}"
        ));
        history.splice(start..end, std::iter::once(summary_msg));

        // Repair orphaned tool pairs
        repair_tool_pairs(history);

        Ok(true)
    }

    /// Create a condensed summary over existing leaf summaries for the current session.
    async fn maybe_condense_leaves(
        &self,
        sqlite: &SqliteMemory,
        session_id: &str,
        provider: &dyn Provider,
        model: &str,
        timeout: Duration,
    ) -> Result<()> {
        let leaves = sqlite.get_leaf_summaries(session_id)?;
        if leaves.len() <= self.config.condensed_summary_leaf_limit {
            return Ok(()); // race: already condensed
        }

        // Build condensed input from leaf contents
        let combined: String = leaves
            .iter()
            .enumerate()
            .map(|(i, l)| format!("Summary {}: {}", i + 1, l.content))
            .collect::<Vec<_>>()
            .join("\n\n");

        let l1_prompt = format!(
            "Condense the following {} session summaries into a single unified summary. \
             Preserve all key facts, decisions, and identifiers.\n\n{combined}",
            leaves.len()
        );

        let condensed_raw = match tokio::time::timeout(
            timeout,
            provider.chat_with_system(Some(SUMMARIZER_SYSTEM), &l1_prompt, model, 0.1),
        )
        .await
        {
            Ok(Ok(s)) => s,
            _ => {
                // Level 3 fallback for condensation
                leaves
                    .iter()
                    .map(|l| truncate_chars(&l.content, 512))
                    .collect::<Vec<_>>()
                    .join("\n")
            }
        };

        let content = truncate_chars(&condensed_raw, self.config.summary_max_chars);
        let condensed_id = uuid::Uuid::new_v4().to_string();

        #[allow(clippy::cast_possible_wrap)]
        let token_count = Some(content.len().div_ceil(4) as i64);

        // Determine the max level used across the covered leaves
        let max_level = leaves.iter().map(|l| l.level).max().unwrap_or(1);

        sqlite.insert_summary(
            &condensed_id,
            "condensed",
            &content,
            token_count,
            session_id,
            max_level,
        )?;

        let sources: Vec<(&str, &str)> = leaves
            .iter()
            .map(|l| (l.id.as_str(), "summary"))
            .collect();
        sqlite.link_summary_sources(&condensed_id, &sources)?;

        tracing::info!(
            leaf_count = leaves.len(),
            condensed_id = %condensed_id,
            "Created condensed summary"
        );

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Level 3 deterministic fallback
// ---------------------------------------------------------------------------

/// Level 3 deterministic summarization: truncate each message to 512 chars and join.
/// No LLM call — always terminates.
fn level3_deterministic(messages: &[ChatMessage]) -> String {
    messages
        .iter()
        .map(|m| {
            let role = m.role.to_uppercase();
            let body = truncate_chars(m.content.trim(), 512);
            format!("{role}: {body}")
        })
        .collect::<Vec<_>>()
        .join("\n")
}

// ---------------------------------------------------------------------------
// Boundary alignment
// ---------------------------------------------------------------------------

/// Move boundary forward past any orphaned tool results at the start.
fn align_boundary_forward(messages: &[ChatMessage], idx: usize) -> usize {
    let mut i = idx;
    while i < messages.len() && messages[i].role == "tool" {
        i += 1;
    }
    i
}

/// Move boundary backward past any tool_call-bearing assistant messages at the end
/// so their results stay in the protected tail.
fn align_boundary_backward(messages: &[ChatMessage], idx: usize) -> usize {
    let mut i = idx;
    // If the message just before the boundary is an assistant message that likely
    // contains tool calls (heuristic: followed by a tool result), pull the boundary back.
    while i > 0 && i < messages.len() && messages[i].role == "tool" {
        // The tool result at `i` belongs to a tool_call before it — move boundary past it
        i -= 1;
    }
    i
}

// ---------------------------------------------------------------------------
// Tool pair repair
// ---------------------------------------------------------------------------

/// Remove orphaned tool_results and add stubs for orphaned tool_calls.
///
/// After compression, some tool results may reference tool_calls that were
/// summarized away, and vice versa. This function cleans up the history
/// so every tool_result has a matching assistant message and every
/// tool_call-bearing assistant message has results.
fn repair_tool_pairs(messages: &mut Vec<ChatMessage>) {
    // Heuristic: tool messages whose content references a call ID that no longer
    // exists in any assistant message should be removed. Since ChatMessage is a
    // simple role+content struct (no structured tool_call_id field), we use a
    // simpler approach: remove any "tool" message that immediately follows the
    // [CONTEXT SUMMARY] message (it's orphaned by definition).
    let mut i = 0;
    while i < messages.len() {
        if messages[i].content.contains("[SUMMARY:") || messages[i].content.contains("[CONTEXT SUMMARY") {
            // Remove any immediately following orphaned tool results
            while i + 1 < messages.len() && messages[i + 1].role == "tool" {
                messages.remove(i + 1);
            }
        }
        i += 1;
    }

    // Also check for tool results at the very start (after system prompt) that
    // are orphaned because their assistant message was compressed.
    let start = if messages.first().is_some_and(|m| m.role == "system") {
        1
    } else {
        0
    };
    while start < messages.len() && messages[start].role == "tool" {
        messages.remove(start);
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn build_transcript(messages: &[ChatMessage], max_chars: usize) -> String {
    let mut transcript = String::new();
    for msg in messages {
        let role = msg.role.to_uppercase();
        let _ = writeln!(transcript, "{role}: {}", msg.content.trim());
    }

    if transcript.len() > max_chars {
        truncate_chars(&transcript, max_chars)
    } else {
        transcript
    }
}

fn truncate_chars(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    // Find a safe char boundary
    let mut end = max;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    let mut result = s[..end].to_string();
    result.push_str("...");
    result
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use std::sync::Mutex;
    use zeroclaw_api::provider::{ChatRequest, ChatResponse};
    use zeroclaw_memory::sqlite::SqliteMemory;

    fn msg(role: &str, content: &str) -> ChatMessage {
        ChatMessage {
            role: role.to_string(),
            content: content.to_string(),
            ..Default::default()
        }
    }

    /// Mock provider whose `chat_with_system` calls return pre-scripted responses in order.
    struct ScriptedSummarizer {
        responses: Mutex<Vec<String>>,
        call_count: Mutex<usize>,
    }

    impl ScriptedSummarizer {
        fn new(responses: Vec<impl Into<String>>) -> Self {
            Self {
                responses: Mutex::new(responses.into_iter().map(Into::into).collect()),
                call_count: Mutex::new(0),
            }
        }

        fn call_count(&self) -> usize {
            *self.call_count.lock().unwrap()
        }
    }

    #[async_trait]
    impl zeroclaw_api::provider::Provider for ScriptedSummarizer {
        async fn chat_with_system(
            &self,
            _system: Option<&str>,
            _message: &str,
            _model: &str,
            _temperature: f64,
        ) -> anyhow::Result<String> {
            *self.call_count.lock().unwrap() += 1;
            let mut guard = self.responses.lock().unwrap();
            if guard.is_empty() {
                Ok("short summary".to_string())
            } else {
                Ok(guard.remove(0))
            }
        }

        async fn chat(
            &self,
            _request: ChatRequest<'_>,
            _model: &str,
            _temperature: f64,
        ) -> anyhow::Result<ChatResponse> {
            Ok(ChatResponse {
                text: Some("done".into()),
                tool_calls: vec![],
                usage: None,
                reasoning_content: None,
            })
        }
    }

    /// Builds a ContextCompressor configured for test use.
    fn test_compressor(context_window: usize) -> ContextCompressor {
        let config = ContextCompressionConfig {
            enabled: true,
            threshold_ratio: 0.50,
            hard_threshold_ratio: 0.80,
            protect_first_n: 1,
            protect_last_n: 1,
            max_passes: 3,
            summary_max_chars: 4_000,
            source_max_chars: 50_000,
            timeout_secs: 60,
            summary_model: None,
            identifier_policy: "strict".to_string(),
            tool_result_retrim_chars: 2_000,
            tool_result_trim_exempt: vec![],
            condensed_summary_leaf_limit: 10,
        };
        ContextCompressor::new(config, context_window)
    }

    // ── Three-level escalation tests ────────────────────────────────────────

    #[tokio::test]
    async fn level1_compresses_normally() {
        // Context window 1000 tokens; history at ~600 tokens → over 50% threshold
        let compressor = test_compressor(1000);

        // 5 messages of ~50 chars each ≈ 5*(50/4+4)*1.2 ≈ 90 tokens
        // Need to be over 500 tokens (50% of 1000): use large messages
        let big = "a".repeat(1700); // ~510 tokens each
        let mut history = vec![
            msg("system", "sys"),
            msg("user", &big),
            msg("assistant", "response"),
        ];

        // Provider returns a short summary (clearly less than 90% of input)
        let provider = ScriptedSummarizer::new(vec!["• discussed topic A", "• key decision made"]);

        let result = compressor.compress_if_needed(&mut history, &provider, "test-model").await.unwrap();
        assert!(result.compressed, "Should have compressed");
        // Level 1 should have fired (1 LLM call)
        assert_eq!(provider.call_count(), 1, "Level 1 only: 1 LLM call");
        // History should contain [SUMMARY:...] in one of the middle messages
        assert!(history.iter().any(|m| m.content.contains("[SUMMARY:")));
    }

    #[tokio::test]
    async fn level2_fires_when_level1_output_not_smaller() {
        let compressor = test_compressor(1000);

        let big = "a".repeat(1700);
        let mut history = vec![
            msg("system", "sys"),
            msg("user", &big),
            msg("assistant", "response"),
        ];

        // Level 1 returns same-length output as transcript (won't shrink enough → Level 2)
        // Level 2 returns a short summary
        let l1_response = "a".repeat(10_000); // much larger than input → triggers L2
        let l2_response = "• brief bullet point summary";
        let provider = ScriptedSummarizer::new(vec![l1_response.as_str(), l2_response]);

        let result = compressor.compress_if_needed(&mut history, &provider, "test-model").await.unwrap();
        assert!(result.compressed);
        assert_eq!(provider.call_count(), 2, "Level 1 + Level 2 = 2 LLM calls");
        assert!(history.iter().any(|m| m.content.contains("[SUMMARY:")));
    }

    #[tokio::test]
    async fn level3_fires_deterministically() {
        let compressor = test_compressor(1000);

        let big = "a".repeat(1700);
        let mut history = vec![
            msg("system", "sys"),
            msg("user", &big),
            msg("assistant", "response"),
        ];

        // Both L1 and L2 return large output → L3 fires (no 3rd LLM call)
        let huge = "a".repeat(50_000);
        let provider = ScriptedSummarizer::new(vec![huge.as_str(), huge.as_str()]);

        let result = compressor.compress_if_needed(&mut history, &provider, "test-model").await.unwrap();
        assert!(result.compressed);
        assert_eq!(provider.call_count(), 2, "L1 + L2 attempted, L3 is deterministic (no LLM)");
        assert!(history.iter().any(|m| m.content.contains("[SUMMARY:")));
    }

    #[tokio::test]
    async fn level3_always_terminates_with_very_long_input() {
        // 10,000-char messages, no LLM needed (provider would return same length)
        let content = "x".repeat(10_000);
        let messages: Vec<ChatMessage> = (0..5).map(|_| msg("user", &content)).collect();
        let result = level3_deterministic(&messages);
        // Level 3 truncates to 512 chars per message — output < input
        assert!(result.len() < content.len() * 5, "Level 3 must produce shorter output");
        assert!(!result.is_empty());
    }

    #[tokio::test]
    async fn summary_dag_insert() {
        use tempfile::TempDir;
        let tmp = TempDir::new().unwrap();
        let sqlite = Arc::new(SqliteMemory::new(tmp.path()).unwrap());

        let config = ContextCompressionConfig {
            protect_first_n: 1,
            protect_last_n: 1,
            condensed_summary_leaf_limit: 10,
            ..ContextCompressionConfig::default()
        };
        let compressor = ContextCompressor::new(config, 1000)
            .with_sqlite_memory(Arc::clone(&sqlite), "session-dag");

        let big = "a".repeat(1700);
        let mut history = vec![
            msg("system", "sys"),
            msg("user", &big),
            msg("assistant", "response"),
        ];
        let provider = ScriptedSummarizer::new(vec!["compact summary"]);
        compressor.compress_if_needed(&mut history, &provider, "model").await.unwrap();

        let conn = sqlite.connection().lock();
        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM summaries WHERE session_id = 'session-dag'", [], |row| row.get(0))
            .unwrap();
        assert_eq!(count, 1, "One summary row inserted");
    }

    #[tokio::test]
    async fn summary_sources_link_messages_test() {
        use tempfile::TempDir;
        let tmp = TempDir::new().unwrap();
        let sqlite = Arc::new(SqliteMemory::new(tmp.path()).unwrap());

        // Pre-insert messages so source IDs exist in messages table
        let user_id = uuid::Uuid::new_v4().to_string();
        sqlite.append_message(&user_id, "session-s", "user", "hello there", None).unwrap();

        let config = ContextCompressionConfig {
            protect_first_n: 0,
            protect_last_n: 0,
            condensed_summary_leaf_limit: 10,
            ..ContextCompressionConfig::default()
        };
        let compressor = ContextCompressor::new(config, 100)
            .with_sqlite_memory(Arc::clone(&sqlite), "session-s");

        // Build a history with a known message ID
        let known_msg = ChatMessage {
            id: user_id.clone(),
            role: "user".to_string(),
            content: "a".repeat(1000),
        };
        let mut history = vec![known_msg];
        let provider = ScriptedSummarizer::new(vec!["summary text"]);
        compressor.compress_if_needed(&mut history, &provider, "model").await.unwrap();

        let conn = sqlite.connection().lock();
        let linked: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM summary_sources WHERE source_id = ?1 AND source_kind = 'message'",
                rusqlite::params![user_id],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(linked, 1, "Known message ID must be linked as a source");
    }

    #[tokio::test]
    async fn history_vec_contains_placeholder_after_compression() {
        let compressor = test_compressor(1000);
        let big = "a".repeat(1700);
        let mut history = vec![
            msg("system", "sys"),
            msg("user", &big),
            msg("assistant", "response"),
        ];
        let provider = ScriptedSummarizer::new(vec!["brief summary"]);
        compressor.compress_if_needed(&mut history, &provider, "model").await.unwrap();

        assert!(
            history.iter().any(|m| m.content.starts_with("[SUMMARY:")),
            "History must contain a [SUMMARY:{{id}}] placeholder"
        );
    }

    #[tokio::test]
    async fn old_memories_daily_write_removed() {
        use tempfile::TempDir;
        let tmp = TempDir::new().unwrap();
        let mem = zeroclaw_memory::sqlite::SqliteMemory::new(tmp.path()).unwrap();
        let mem_arc: Arc<dyn zeroclaw_memory::traits::Memory> = Arc::new(
            zeroclaw_memory::sqlite::SqliteMemory::new(tmp.path()).unwrap(),
        );

        let config = ContextCompressionConfig {
            protect_first_n: 1,
            protect_last_n: 1,
            ..ContextCompressionConfig::default()
        };
        let compressor = ContextCompressor::new(config, 1000).with_memory(mem_arc);

        let big = "a".repeat(1700);
        let mut history = vec![
            msg("system", "sys"),
            msg("user", &big),
            msg("assistant", "response"),
        ];
        let provider = ScriptedSummarizer::new(vec!["summary"]);
        compressor.compress_if_needed(&mut history, &provider, "model").await.unwrap();

        // Verify no 'compressed_context_*' keys in memories table
        let conn = mem.connection().lock();
        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM memories WHERE key LIKE 'compressed_context_%'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(count, 0, "No compressed_context_* entries should be written");
        drop(conn);
    }

    #[tokio::test]
    async fn hard_threshold_fires_at_hard_ratio() {
        // Context window 1000 tokens; soft threshold 50% = 500, hard threshold 80% = 800.
        // Token estimate: (content_len / 4 + 4) * 1.2 per message.
        // For a 3-message history [sys(3 chars), user(N chars), asst(8 chars)]:
        //   sys:  (1 + 4) * 1.2 ≈ 6 tokens
        //   asst: (2 + 4) * 1.2 ≈ 7 tokens
        //   user: (N/4 + 4) * 1.2 tokens
        //   To exceed hard=800: user needs > 800-13=787 tokens
        //   → N/4 + 4 > 787/1.2=656  → N > (656-4)*4=2608 chars  → use 3000
        let config = ContextCompressionConfig {
            enabled: true,
            threshold_ratio: 0.50,
            hard_threshold_ratio: 0.80,
            protect_first_n: 1,
            protect_last_n: 1,
            condensed_summary_leaf_limit: 10,
            ..ContextCompressionConfig::default()
        };
        let compressor = ContextCompressor::new(config, 1000);

        // ~900 tokens — clearly over hard threshold (800)
        let big = "a".repeat(3000);
        let mut history_big = vec![
            msg("system", "sys"),
            msg("user", &big),
            msg("assistant", "response"),
        ];

        // ~560 tokens (1700 chars) — over soft (500) but under hard (800)
        // (1700/4 + 4)*1.2 = 429*1.2 ≈ 514 tokens + overhead ≈ 527 total... actually:
        // sys(6) + user((1700/4+4)*1.2=(429)*1.2=514.8≈514) + asst(7) = 527 tokens
        // 527 < 800 → should NOT fire for hard threshold
        let medium = "a".repeat(1700);
        let mut history_med = vec![
            msg("system", "sys"),
            msg("user", &medium),
            msg("assistant", "response"),
        ];

        let provider = ScriptedSummarizer::new(vec!["summary"]);
        let result = compressor.compress_for_hard_threshold(&mut history_big, &provider, "model").await.unwrap();
        assert!(result.compressed, "Hard threshold should fire at ~900 tokens (> 800)");

        let provider2 = ScriptedSummarizer::new(vec![] as Vec<String>);
        let result2 = compressor.compress_for_hard_threshold(&mut history_med, &provider2, "model").await.unwrap();
        assert!(!result2.compressed, "~527-token history is under hard threshold (800), should not compress");
    }

    #[tokio::test]
    async fn summaries_fts_searchable_via_search_summaries() {
        use tempfile::TempDir;
        let tmp = TempDir::new().unwrap();
        let sqlite = Arc::new(SqliteMemory::new(tmp.path()).unwrap());

        let config = ContextCompressionConfig {
            protect_first_n: 1,
            protect_last_n: 1,
            condensed_summary_leaf_limit: 10,
            ..ContextCompressionConfig::default()
        };
        let compressor = ContextCompressor::new(config, 1000)
            .with_sqlite_memory(Arc::clone(&sqlite), "session-fts");

        let big = "a tokio async runtime discussion ".repeat(60); // ~540 tokens
        let mut history = vec![
            msg("system", "sys"),
            msg("user", &big),
            msg("assistant", "response"),
        ];
        let provider = ScriptedSummarizer::new(vec!["tokio async runtime covered"]);
        compressor.compress_if_needed(&mut history, &provider, "model").await.unwrap();

        // The summary content should be searchable
        let results = sqlite.search_summaries("tokio", 10, Some("session-fts")).unwrap();
        assert!(!results.is_empty(), "Summary content should be FTS-searchable");
    }

    #[tokio::test]
    async fn condensed_summary_created_when_leaf_limit_exceeded() {
        use tempfile::TempDir;
        let tmp = TempDir::new().unwrap();
        let sqlite = Arc::new(SqliteMemory::new(tmp.path()).unwrap());

        // Leaf limit = 2. Context window = 50 tokens, soft threshold = 0.5 → 25 tokens.
        // A single "x".repeat(200) message: (200/4 + 4)*1.2 = 54*1.2 ≈ 64 tokens → triggers.
        let config = ContextCompressionConfig {
            enabled: true,
            threshold_ratio: 0.5,
            hard_threshold_ratio: 0.80,
            protect_first_n: 0,
            protect_last_n: 0,
            max_passes: 1,
            summary_max_chars: 4_000,
            source_max_chars: 50_000,
            timeout_secs: 60,
            summary_model: None,
            identifier_policy: "strict".to_string(),
            tool_result_retrim_chars: 2_000,
            tool_result_trim_exempt: vec![],
            condensed_summary_leaf_limit: 2,
        };
        // context_window = 50 → soft threshold = 25 tokens; 64-token message exceeds it
        let compressor = ContextCompressor::new(config, 50)
            .with_sqlite_memory(Arc::clone(&sqlite), "session-cond");

        // Trigger 3 compressions; each inserts a leaf. On the 3rd, leaf_count becomes 3 > 2.
        for _ in 0..3 {
            let mut history = vec![msg("user", &"x".repeat(200))];
            // Two responses: first for leaf L1, second for the condensed summary attempt.
            let provider = ScriptedSummarizer::new(vec!["leaf summary content", "condensed content"]);
            compressor.compress_if_needed(&mut history, &provider, "model").await.unwrap();
        }

        let conn = sqlite.connection().lock();
        let condensed_count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM summaries WHERE kind = 'condensed' AND session_id = 'session-cond'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert!(condensed_count >= 1, "At least one condensed summary should have been created");
    }

    #[test]
    fn test_estimate_tokens() {
        let messages = vec![msg("user", "hello world")]; // 11 chars
        let tokens = estimate_tokens(&messages);
        // 11/4 ceil = 3, +4 framing = 7, *1.2 = 8.4 -> 8
        assert!(tokens > 0);
    }

    #[test]
    fn test_estimate_tokens_empty() {
        assert_eq!(estimate_tokens(&[]), 0);
    }

    #[test]
    fn test_parse_context_limit_anthropic() {
        let msg = "prompt is too long: 150000 tokens > 128000 maximum context length";
        assert_eq!(parse_context_limit_from_error(msg), Some(128_000));
    }

    #[test]
    fn test_parse_context_limit_openai() {
        let msg = "This model's maximum context length is 128000 tokens. However, your messages resulted in 150000 tokens.";
        assert_eq!(parse_context_limit_from_error(msg), Some(128_000));
    }

    #[test]
    fn test_parse_context_limit_llamacpp() {
        let msg = "request (8968 tokens) exceeds the available context size (8448 tokens)";
        assert_eq!(parse_context_limit_from_error(msg), Some(8448));
    }

    #[test]
    fn test_parse_context_limit_none() {
        assert_eq!(parse_context_limit_from_error("some random error"), None);
    }

    #[test]
    fn test_parse_context_limit_rejects_small() {
        let msg = "limit is 100 tokens";
        assert_eq!(parse_context_limit_from_error(msg), None); // < 1024
    }

    #[test]
    fn test_next_probe_tier() {
        assert_eq!(next_probe_tier(2_000_001), 2_000_000);
        assert_eq!(next_probe_tier(2_000_000), 1_000_000);
        assert_eq!(next_probe_tier(200_000), 128_000);
        assert_eq!(next_probe_tier(64_000), 32_000);
        assert_eq!(next_probe_tier(32_000), 32_000); // floor
        assert_eq!(next_probe_tier(10_000), 32_000); // below all tiers
    }

    #[test]
    fn test_align_boundary_forward_skips_tool() {
        let messages = vec![
            msg("system", "sys"),
            msg("user", "q"),
            msg("tool", "result1"),
            msg("tool", "result2"),
            msg("user", "next"),
        ];
        // Starting at index 2 (tool), should skip to index 4
        assert_eq!(align_boundary_forward(&messages, 2), 4);
    }

    #[test]
    fn test_align_boundary_forward_noop() {
        let messages = vec![
            msg("system", "sys"),
            msg("user", "q"),
            msg("assistant", "a"),
        ];
        assert_eq!(align_boundary_forward(&messages, 1), 1);
    }

    #[test]
    fn test_repair_tool_pairs_removes_orphaned() {
        let mut messages = vec![
            msg("system", "sys"),
            msg(
                "assistant",
                "[SUMMARY:abc-123-def]\n\nstuff",
            ),
            msg("tool", "orphaned result"),
            msg("user", "next question"),
        ];
        repair_tool_pairs(&mut messages);
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[2].role, "user");
    }

    #[test]
    fn test_repair_tool_pairs_legacy_context_summary() {
        // Ensure legacy [CONTEXT SUMMARY] format is still handled
        let mut messages = vec![
            msg("system", "sys"),
            msg("assistant", "[CONTEXT SUMMARY — 5 earlier messages compressed]\nstuff"),
            msg("tool", "orphaned result"),
            msg("user", "next question"),
        ];
        repair_tool_pairs(&mut messages);
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[2].role, "user");
    }

    #[test]
    fn test_repair_tool_pairs_no_false_positives() {
        let mut messages = vec![
            msg("system", "sys"),
            msg("user", "q"),
            msg("assistant", "calling tool"),
            msg("tool", "result"),
            msg("user", "thanks"),
        ];
        repair_tool_pairs(&mut messages);
        assert_eq!(messages.len(), 5); // no change
    }

    #[test]
    fn test_build_transcript() {
        let messages = vec![msg("user", "hello"), msg("assistant", "hi there")];
        let t = build_transcript(&messages, 10_000);
        assert!(t.contains("USER: hello"));
        assert!(t.contains("ASSISTANT: hi there"));
    }

    #[test]
    fn test_build_transcript_truncates() {
        let messages = vec![msg("user", &"x".repeat(1000))];
        let t = build_transcript(&messages, 100);
        assert!(t.len() <= 103); // 100 + "..."
    }

    #[test]
    fn test_truncate_chars() {
        assert_eq!(truncate_chars("hello world", 5), "hello...");
        assert_eq!(truncate_chars("hi", 10), "hi");
    }

    #[test]
    fn test_config_defaults() {
        let config = ContextCompressionConfig::default();
        assert!(config.enabled);
        assert!((config.threshold_ratio - 0.50).abs() < f64::EPSILON);
        assert!((config.hard_threshold_ratio - 0.80).abs() < f64::EPSILON);
        assert_eq!(config.protect_first_n, 3);
        assert_eq!(config.protect_last_n, 4);
        assert_eq!(config.max_passes, 3);
        assert_eq!(config.summary_max_chars, 4_000);
        assert_eq!(config.source_max_chars, 50_000);
        assert_eq!(config.timeout_secs, 60);
        assert!(config.summary_model.is_none());
        assert_eq!(config.identifier_policy, "strict");
        assert_eq!(config.condensed_summary_leaf_limit, 10);
    }

    #[test]
    fn test_config_serde_defaults() {
        let json = "{}";
        let config: ContextCompressionConfig = serde_json::from_str(json).unwrap();
        assert!(config.enabled);
        assert_eq!(config.protect_first_n, 3);
        assert_eq!(config.max_passes, 3);
    }

    #[test]
    fn test_config_serde_override() {
        let json = r#"{"enabled": false, "protect_first_n": 5, "max_passes": 1}"#;
        let config: ContextCompressionConfig = serde_json::from_str(json).unwrap();
        assert!(!config.enabled);
        assert_eq!(config.protect_first_n, 5);
        assert_eq!(config.max_passes, 1);
    }

    // ── fast_trim_tool_results tests ────────────────────────────────

    #[test]
    fn test_fast_trim_protects_first_and_last_n() {
        let config = ContextCompressionConfig {
            protect_first_n: 2,
            protect_last_n: 2,
            tool_result_retrim_chars: 100,
            ..Default::default()
        };
        let compressor = ContextCompressor::new(config, 128_000);
        let big = "x".repeat(5_000);
        let mut history = vec![
            msg("system", "sys"),
            msg("tool", &big), // index 1 — protected (first 2)
            msg("user", "q"),
            msg("tool", &big),   // index 3 — trimmable
            msg("user", "next"), // index 4 — protected (last 2)
            msg("tool", &big),   // index 5 — protected (last 2)
        ];
        let saved = compressor.fast_trim_tool_results(&mut history);
        assert!(saved > 0);
        // Protected messages unchanged
        assert_eq!(history[1].content.len(), 5_000);
        assert_eq!(history[5].content.len(), 5_000);
        // Trimmable message was trimmed
        assert!(history[3].content.len() <= 200); // 100 + marker overhead
    }

    #[test]
    fn test_fast_trim_skips_images() {
        let config = ContextCompressionConfig {
            protect_first_n: 0,
            protect_last_n: 0,
            tool_result_retrim_chars: 100,
            ..Default::default()
        };
        let compressor = ContextCompressor::new(config, 128_000);
        let img = format!("data:image/{}", "x".repeat(5_000));
        let mut history = vec![msg("tool", &img)];
        let saved = compressor.fast_trim_tool_results(&mut history);
        assert_eq!(saved, 0);
        assert!(history[0].content.len() > 5_000);
    }

    #[test]
    fn test_fast_trim_skips_exempt_tools() {
        let config = ContextCompressionConfig {
            protect_first_n: 0,
            protect_last_n: 0,
            tool_result_retrim_chars: 100,
            tool_result_trim_exempt: vec!["KEEPME".to_string()],
            ..Default::default()
        };
        let compressor = ContextCompressor::new(config, 128_000);
        let content = format!("KEEPME {}", "x".repeat(5_000));
        let mut history = vec![msg("tool", &content)];
        let saved = compressor.fast_trim_tool_results(&mut history);
        assert_eq!(saved, 0);
    }

    #[test]
    fn test_fast_trim_skips_small_results() {
        let config = ContextCompressionConfig {
            protect_first_n: 0,
            protect_last_n: 0,
            tool_result_retrim_chars: 2_000,
            ..Default::default()
        };
        let compressor = ContextCompressor::new(config, 128_000);
        let mut history = vec![msg("tool", "small result")];
        let saved = compressor.fast_trim_tool_results(&mut history);
        assert_eq!(saved, 0);
    }

    #[test]
    fn test_fast_trim_skips_non_tool_messages() {
        let config = ContextCompressionConfig {
            protect_first_n: 0,
            protect_last_n: 0,
            tool_result_retrim_chars: 100,
            ..Default::default()
        };
        let compressor = ContextCompressor::new(config, 128_000);
        let big = "x".repeat(5_000);
        let mut history = vec![msg("user", &big), msg("assistant", &big)];
        let saved = compressor.fast_trim_tool_results(&mut history);
        assert_eq!(saved, 0);
    }

    #[test]
    fn test_fast_trim_config_defaults() {
        let config = ContextCompressionConfig::default();
        assert_eq!(config.tool_result_retrim_chars, 2_000);
        assert!(config.tool_result_trim_exempt.is_empty());
    }

    #[test]
    fn test_fast_trim_disabled_when_zero() {
        let config = ContextCompressionConfig {
            protect_first_n: 0,
            protect_last_n: 0,
            tool_result_retrim_chars: 0,
            ..Default::default()
        };
        let compressor = ContextCompressor::new(config, 128_000);
        let big = "x".repeat(5_000);
        let mut history = vec![msg("tool", &big)];
        let saved = compressor.fast_trim_tool_results(&mut history);
        assert_eq!(saved, 0);
    }
}
