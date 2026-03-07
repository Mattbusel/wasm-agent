// SPDX-License-Identifier: MIT
//! Core types for the ReAct agent loop.

use serde::{Deserialize, Serialize};

/// A single step in the ReAct loop.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReActStep {
    /// The agent's internal reasoning trace.
    Thought(String),
    /// A tool invocation with its name and JSON input.
    Action { tool: String, input: String },
    /// The result of a tool invocation.
    Observation(String),
    /// The agent's final answer, terminating the loop.
    FinalAnswer(String),
}

impl ReActStep {
    /// Returns a short label for the step kind.
    pub fn kind(&self) -> &'static str {
        match self {
            ReActStep::Thought(_) => "Thought",
            ReActStep::Action { .. } => "Action",
            ReActStep::Observation(_) => "Observation",
            ReActStep::FinalAnswer(_) => "FinalAnswer",
        }
    }

    /// Returns `true` if this step terminates the loop.
    pub fn is_final(&self) -> bool {
        matches!(self, ReActStep::FinalAnswer(_))
    }

    /// Returns the primary text content of the step.
    pub fn content(&self) -> &str {
        match self {
            ReActStep::Thought(s) | ReActStep::Observation(s) | ReActStep::FinalAnswer(s) => s,
            ReActStep::Action { input, .. } => input,
        }
    }
}

/// Role of a participant in the conversation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Role {
    /// The system prompt that defines the agent's persona and instructions.
    System,
    /// A human user turn.
    User,
    /// An assistant (model) turn.
    Assistant,
    /// A tool result injected into the conversation.
    Tool,
}

/// A single message in the conversation history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    /// Role of the message author.
    pub role: Role,
    /// Text content of the message.
    pub content: String,
    /// Rough token count estimate (4 chars per token heuristic).
    pub token_estimate: usize,
}

impl Message {
    /// Creates a new message, estimating its token count.
    pub fn new(role: Role, content: impl Into<String>) -> Self {
        let content = content.into();
        let token_estimate = content.len() / 4;
        Self { role, content, token_estimate }
    }

    /// Convenience constructor for system messages.
    pub fn system(content: impl Into<String>) -> Self { Self::new(Role::System, content) }
    /// Convenience constructor for user messages.
    pub fn user(content: impl Into<String>) -> Self { Self::new(Role::User, content) }
    /// Convenience constructor for assistant messages.
    pub fn assistant(content: impl Into<String>) -> Self { Self::new(Role::Assistant, content) }
}

/// Agent configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    /// Maximum number of ReAct iterations before giving up.
    pub max_iterations: u32,
    /// Maximum total tokens allowed in the conversation history.
    pub context_token_limit: usize,
    /// Optional wall-clock timeout in milliseconds.
    pub timeout_ms: Option<u64>,
    /// Model identifier string.
    pub model: String,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            max_iterations: 10,
            context_token_limit: 8192,
            timeout_ms: Some(30_000),
            model: "claude-haiku-4-5-20251001".into(),
        }
    }
}

/// The result of executing a tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    /// Name of the tool that produced this result.
    pub tool_name: String,
    /// The tool's textual output.
    pub output: String,
    /// Whether execution succeeded.
    pub success: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_react_step_thought_kind() {
        let s = ReActStep::Thought("think".into());
        assert_eq!(s.kind(), "Thought");
        assert!(!s.is_final());
    }

    #[test]
    fn test_react_step_final_answer_is_final() {
        let s = ReActStep::FinalAnswer("done".into());
        assert!(s.is_final());
    }

    #[test]
    fn test_react_step_action_kind() {
        let s = ReActStep::Action { tool: "search".into(), input: "query".into() };
        assert_eq!(s.kind(), "Action");
    }

    #[test]
    fn test_react_step_observation_kind() {
        let s = ReActStep::Observation("result".into());
        assert_eq!(s.kind(), "Observation");
        assert!(!s.is_final());
    }

    #[test]
    fn test_react_step_content_returns_text() {
        let s = ReActStep::Thought("my thought".into());
        assert_eq!(s.content(), "my thought");
    }

    #[test]
    fn test_react_step_action_content_is_input() {
        let s = ReActStep::Action { tool: "t".into(), input: "i".into() };
        assert_eq!(s.content(), "i");
    }

    #[test]
    fn test_message_token_estimate_nonzero_for_nonempty() {
        let m = Message::user("hello world this is a test");
        assert!(m.token_estimate > 0);
    }

    #[test]
    fn test_message_empty_content_zero_tokens() {
        let m = Message::user("");
        assert_eq!(m.token_estimate, 0);
    }

    #[test]
    fn test_agent_config_default_has_reasonable_limits() {
        let c = AgentConfig::default();
        assert!(c.max_iterations > 0);
        assert!(c.context_token_limit > 0);
    }

    #[test]
    fn test_message_system_has_system_role() {
        let m = Message::system("sys");
        assert_eq!(m.role, Role::System);
    }

    #[test]
    fn test_message_assistant_has_assistant_role() {
        let m = Message::assistant("asst");
        assert_eq!(m.role, Role::Assistant);
    }
}
