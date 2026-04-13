//! `lcm_grep` tool — regex search over verbatim conversation history.
//!
//! Allows the agent to do forensic searches over the full message log stored
//! in the brain.db `messages` table, annotated with whether each message has
//! been covered by a summary (compacted) or is still "active" in the window.

use async_trait::async_trait;
use serde_json::json;
use std::fmt::Write;
use std::sync::Arc;
use zeroclaw_api::tool::{Tool, ToolResult};
use zeroclaw_memory::SqliteMemory;

/// Maximum results the tool will ever return, regardless of caller input.
const MAX_LIMIT: usize = 50;

/// Regex search over the full verbatim conversation history.
pub struct LcmGrepTool {
    sqlite: Arc<SqliteMemory>,
}

impl LcmGrepTool {
    pub fn new(sqlite: Arc<SqliteMemory>) -> Self {
        Self { sqlite }
    }
}

#[async_trait]
impl Tool for LcmGrepTool {
    fn name(&self) -> &str {
        "lcm_grep"
    }

    fn description(&self) -> &str {
        "Regex search over the full verbatim conversation history stored in the message log. Returns matching messages annotated with whether each has been compacted into a summary or is still active in the context window."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regular expression pattern to search message content (validated before executing)"
                },
                "session_id": {
                    "type": "string",
                    "description": "Filter results to a specific session ID (optional)"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default 20, max 50)",
                    "minimum": 1,
                    "maximum": 50
                }
            },
            "required": ["pattern"]
        })
    }

    async fn execute(&self, args: serde_json::Value) -> anyhow::Result<ToolResult> {
        let pattern = match args.get("pattern").and_then(|v| v.as_str()) {
            Some(p) => p,
            None => {
                return Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some("missing required parameter 'pattern'".into()),
                });
            }
        };

        let session_id = args.get("session_id").and_then(|v| v.as_str());
        let limit = args
            .get("limit")
            .and_then(|v| v.as_u64())
            .map(|n| (n as usize).min(MAX_LIMIT))
            .unwrap_or(20);

        match self.sqlite.grep_messages(pattern, session_id, limit) {
            Ok(entries) => {
                if entries.is_empty() {
                    return Ok(ToolResult {
                        success: true,
                        output: format!("No messages matched pattern `{pattern}`."),
                        error: None,
                    });
                }

                let mut output = String::new();
                let _ = writeln!(output, "{} result(s) for pattern `{pattern}`:\n", entries.len());

                for entry in &entries {
                    let annotation = match &entry.summary_id {
                        Some(sid) => format!("covered by summary {sid}"),
                        None => "active".to_string(),
                    };
                    let _ = writeln!(
                        output,
                        "[{}] {}: {}\n({})",
                        entry.created_at, entry.role, entry.content, annotation
                    );
                    output.push('\n');
                }

                Ok(ToolResult {
                    success: true,
                    output,
                    error: None,
                })
            }
            Err(e) => Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!("lcm_grep failed: {e}")),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn test_tool() -> (TempDir, LcmGrepTool) {
        let tmp = TempDir::new().unwrap();
        let sqlite = Arc::new(SqliteMemory::new(tmp.path()).unwrap());
        (tmp, LcmGrepTool::new(sqlite.clone()))
    }

    fn insert_msg(sqlite: &SqliteMemory, id: &str, session: &str, role: &str, content: &str) {
        sqlite.append_message(id, session, role, content, None).unwrap();
    }

    #[tokio::test]
    async fn lcm_grep_tool_returns_formatted_results() {
        let (tmp, tool) = test_tool();
        let sqlite = SqliteMemory::new(tmp.path()).unwrap();
        insert_msg(&sqlite, "m1", "sess", "user", "the answer is 42");

        let result = tool
            .execute(json!({ "pattern": "answer is \\d+" }))
            .await
            .unwrap();

        assert!(result.success, "expected success: {:?}", result.error);
        assert!(result.output.contains("answer is 42"), "output: {}", result.output);
        assert!(result.output.contains("active"), "should show 'active' annotation");
    }

    #[tokio::test]
    async fn lcm_grep_tool_no_match() {
        let (_, tool) = test_tool();
        let result = tool
            .execute(json!({ "pattern": "zzznomatch" }))
            .await
            .unwrap();
        assert!(result.success);
        assert!(result.output.contains("No messages matched"));
    }

    #[tokio::test]
    async fn lcm_grep_tool_invalid_regex() {
        let (_, tool) = test_tool();
        let result = tool
            .execute(json!({ "pattern": "[unclosed" }))
            .await
            .unwrap();
        assert!(!result.success);
        assert!(result.error.is_some());
        assert!(result.error.unwrap().contains("invalid regex"));
    }

    #[tokio::test]
    async fn lcm_grep_tool_limit_respected() {
        let (tmp, tool) = test_tool();
        let sqlite = SqliteMemory::new(tmp.path()).unwrap();
        for i in 0..30 {
            insert_msg(&sqlite, &format!("m{i}"), "sess", "user", &format!("needle {i}"));
        }
        let result = tool
            .execute(json!({ "pattern": "needle", "limit": 5 }))
            .await
            .unwrap();
        assert!(result.success);
        // Count occurrences of "active" lines to verify limit
        let active_count = result.output.matches("active").count();
        assert!(active_count <= 5, "should respect limit=5, got {active_count}");
    }

    #[tokio::test]
    async fn lcm_grep_tool_missing_pattern() {
        let (_, tool) = test_tool();
        let result = tool.execute(json!({})).await.unwrap();
        assert!(!result.success);
        assert!(result.error.unwrap().contains("missing required parameter"));
    }

    #[test]
    fn name_and_schema() {
        let tmp = TempDir::new().unwrap();
        let sqlite = Arc::new(SqliteMemory::new(tmp.path()).unwrap());
        let tool = LcmGrepTool::new(sqlite);
        assert_eq!(tool.name(), "lcm_grep");
        let schema = tool.parameters_schema();
        assert!(schema["properties"]["pattern"].is_object());
        assert!(schema["required"].as_array().unwrap().contains(&json!("pattern")));
    }
}
