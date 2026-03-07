// SPDX-License-Identifier: MIT
use thiserror::Error;

/// All errors that can occur in the wasm-agent ReAct loop.
#[derive(Debug, Error, Clone)]
pub enum AgentError {
    /// The agent ran the maximum number of iterations without producing a final answer.
    #[error("Max iterations ({0}) reached without final answer")]
    MaxIterationsExceeded(u32),
    /// A tool was requested that does not exist in the registry.
    #[error("Tool '{name}' not found in registry")]
    ToolNotFound { name: String },
    /// A tool was found but its execution failed.
    #[error("Tool '{name}' execution failed: {reason}")]
    ToolFailed { name: String, reason: String },
    /// The LLM response could not be parsed as a valid ReAct step.
    #[error("Parse error in ReAct step: {0}")]
    ParseError(String),
    /// The agent exceeded its wall-clock time budget.
    #[error("Timeout: agent exceeded {timeout_ms}ms budget")]
    Timeout { timeout_ms: u64 },
    /// The conversation history exceeded its token budget.
    #[error("History overflow: {size} tokens exceeds context limit {limit}")]
    HistoryOverflow { size: usize, limit: usize },
    /// A value could not be serialized.
    #[error("Serialization error: {0}")]
    Serialization(String),
    /// A tool was registered with an invalid signature.
    #[error("Invalid tool signature: {0}")]
    InvalidToolSignature(String),
}
