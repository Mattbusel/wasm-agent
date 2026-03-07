// SPDX-License-Identifier: MIT
//! Tool registry and dispatch for WASM-safe function signatures.

use std::collections::HashMap;
use crate::error::AgentError;
use crate::types::ToolResult;

/// A tool handler: takes a JSON string input, returns a [`ToolResult`].
///
/// Uses `Box<dyn Fn>` for WASM-compatible dispatch — no async, no `Send`
/// requirement in single-threaded WASM environments.
pub type ToolFn = Box<dyn Fn(&str) -> ToolResult>;

/// Metadata describing a tool for system-prompt generation and introspection.
#[derive(Debug, Clone)]
pub struct ToolSpec {
    /// Unique name used to invoke the tool from the ReAct loop.
    pub name: String,
    /// Human-readable description included in the agent's system prompt.
    pub description: String,
    /// JSON Schema string describing the tool's input format.
    pub input_schema: String,
}

impl ToolSpec {
    /// Creates a new [`ToolSpec`].
    ///
    /// # Arguments
    /// * `name` — Unique tool name (must be non-empty).
    /// * `description` — Human-readable description.
    /// * `schema` — JSON Schema for the tool's input.
    pub fn new(name: impl Into<String>, description: impl Into<String>, schema: impl Into<String>) -> Self {
        Self { name: name.into(), description: description.into(), input_schema: schema.into() }
    }
}

/// Registry of available tools, keyed by name.
pub struct ToolRegistry {
    specs: HashMap<String, ToolSpec>,
    handlers: HashMap<String, ToolFn>,
}

impl ToolRegistry {
    /// Creates an empty registry.
    pub fn new() -> Self {
        Self { specs: HashMap::new(), handlers: HashMap::new() }
    }

    /// Registers a tool with its specification and handler.
    ///
    /// # Errors
    /// Returns [`AgentError::InvalidToolSignature`] if the tool name is empty.
    pub fn register(&mut self, spec: ToolSpec, handler: ToolFn) -> Result<(), AgentError> {
        if spec.name.is_empty() {
            return Err(AgentError::InvalidToolSignature("tool name cannot be empty".into()));
        }
        self.specs.insert(spec.name.clone(), spec.clone());
        self.handlers.insert(spec.name, handler);
        Ok(())
    }

    /// Dispatches a tool call by name with the given JSON input string.
    ///
    /// # Errors
    /// Returns [`AgentError::ToolNotFound`] if no tool with that name is registered.
    pub fn dispatch(&self, tool_name: &str, input: &str) -> Result<ToolResult, AgentError> {
        let handler = self.handlers.get(tool_name)
            .ok_or_else(|| AgentError::ToolNotFound { name: tool_name.to_string() })?;
        Ok(handler(input))
    }

    /// Returns the spec for a registered tool, or `None` if not found.
    pub fn spec(&self, name: &str) -> Option<&ToolSpec> { self.specs.get(name) }

    /// Returns the number of registered tools.
    pub fn tool_count(&self) -> usize { self.specs.len() }

    /// Returns the names of all registered tools.
    pub fn tool_names(&self) -> Vec<&str> { self.specs.keys().map(|s| s.as_str()).collect() }

    /// Generates a system-prompt snippet listing all registered tools.
    pub fn tools_prompt(&self) -> String {
        let mut lines = vec!["Available tools:".to_string()];
        for spec in self.specs.values() {
            lines.push(format!("- {}: {}", spec.name, spec.description));
        }
        lines.join("\n")
    }
}

impl Default for ToolRegistry {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn echo_tool() -> (ToolSpec, ToolFn) {
        let spec = ToolSpec::new("echo", "Echoes input back", r#"{"type":"string"}"#);
        let handler: ToolFn = Box::new(|input: &str| ToolResult {
            tool_name: "echo".into(),
            output: format!("echo: {input}"),
            success: true,
        });
        (spec, handler)
    }

    #[test]
    fn test_registry_register_and_dispatch_ok() {
        let mut reg = ToolRegistry::new();
        let (spec, handler) = echo_tool();
        reg.register(spec, handler).unwrap();
        let result = reg.dispatch("echo", "hello").unwrap();
        assert!(result.success);
        assert_eq!(result.output, "echo: hello");
    }

    #[test]
    fn test_registry_dispatch_unknown_tool_returns_error() {
        let reg = ToolRegistry::new();
        let err = reg.dispatch("nonexistent", "").unwrap_err();
        assert!(matches!(err, AgentError::ToolNotFound { .. }));
    }

    #[test]
    fn test_registry_register_empty_name_returns_error() {
        let mut reg = ToolRegistry::new();
        let spec = ToolSpec::new("", "bad", "{}");
        let err = reg.register(spec, Box::new(|_| ToolResult {
            tool_name: "".into(), output: "".into(), success: false,
        })).unwrap_err();
        assert!(matches!(err, AgentError::InvalidToolSignature(_)));
    }

    #[test]
    fn test_registry_tool_count_increments() {
        let mut reg = ToolRegistry::new();
        assert_eq!(reg.tool_count(), 0);
        let (spec, handler) = echo_tool();
        reg.register(spec, handler).unwrap();
        assert_eq!(reg.tool_count(), 1);
    }

    #[test]
    fn test_registry_tools_prompt_contains_tool_name() {
        let mut reg = ToolRegistry::new();
        let (spec, handler) = echo_tool();
        reg.register(spec, handler).unwrap();
        assert!(reg.tools_prompt().contains("echo"));
    }

    #[test]
    fn test_registry_spec_retrieval_present_and_absent() {
        let mut reg = ToolRegistry::new();
        let (spec, handler) = echo_tool();
        reg.register(spec, handler).unwrap();
        assert!(reg.spec("echo").is_some());
        assert!(reg.spec("missing").is_none());
    }

    #[test]
    fn test_registry_tool_names_lists_all() {
        let mut reg = ToolRegistry::new();
        let (spec, handler) = echo_tool();
        reg.register(spec, handler).unwrap();
        let names = reg.tool_names();
        assert!(names.contains(&"echo"));
    }

    #[test]
    fn test_registry_default_is_empty() {
        let reg = ToolRegistry::default();
        assert_eq!(reg.tool_count(), 0);
    }
}
