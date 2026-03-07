// SPDX-License-Identifier: MIT
//! # wasm-agent
//!
//! ReAct agent loop for WASM/edge environments.
//!
//! Implements the Thought-Action-Observation cycle with a synchronous,
//! WASM-compatible tool dispatch system. All logic compiles and tests on
//! the host target; WASM-specific bindings are gated behind
//! `#[cfg(target_arch = "wasm32")]`.

pub mod error;
pub mod history;
pub mod loop_runner;
pub mod tools;
pub mod types;

pub use error::AgentError;
pub use loop_runner::{parse_react_step, LoopRunner};
pub use tools::{ToolRegistry, ToolSpec};
pub use types::{AgentConfig, Message, ReActStep, Role};
