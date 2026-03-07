// SPDX-License-Identifier: MIT
//! ReAct loop runner: Thought -> Action -> Observation -> FinalAnswer.

use crate::error::AgentError;
use crate::history::ConversationHistory;
use crate::tools::ToolRegistry;
use crate::types::{AgentConfig, Message, ReActStep, Role};

/// Parses a raw LLM response string into a [`ReActStep`].
///
/// Supported formats:
/// - `Thought: <text>`
/// - `Action: <tool_name>(<input>)`
/// - `Action: <tool_name>: <input>`
/// - `Final Answer: <text>`
///
/// # Errors
/// Returns [`AgentError::ParseError`] if the response does not match any known prefix.
pub fn parse_react_step(response: &str) -> Result<ReActStep, AgentError> {
    let trimmed = response.trim();
    if let Some(rest) = trimmed.strip_prefix("Final Answer:") {
        return Ok(ReActStep::FinalAnswer(rest.trim().to_string()));
    }
    if let Some(rest) = trimmed.strip_prefix("Thought:") {
        return Ok(ReActStep::Thought(rest.trim().to_string()));
    }
    if let Some(rest) = trimmed.strip_prefix("Action:") {
        let rest = rest.trim();
        if let Some(paren) = rest.find('(') {
            let tool = rest[..paren].trim().to_string();
            let input = rest[paren + 1..].trim_end_matches(')').trim().to_string();
            return Ok(ReActStep::Action { tool, input });
        }
        if let Some(colon) = rest.find(':') {
            let tool = rest[..colon].trim().to_string();
            let input = rest[colon + 1..].trim().to_string();
            return Ok(ReActStep::Action { tool, input });
        }
        return Ok(ReActStep::Action { tool: rest.to_string(), input: String::new() });
    }
    Err(AgentError::ParseError(format!("Could not parse ReAct step from: {trimmed}")))
}

/// The ReAct loop execution context.
///
/// Holds references to the agent configuration and tool registry.
/// Drives the Thought -> Action -> Observation -> FinalAnswer cycle.
pub struct LoopRunner<'a> {
    config: &'a AgentConfig,
    registry: &'a ToolRegistry,
    history: ConversationHistory,
}

impl<'a> LoopRunner<'a> {
    /// Creates a new runner with its own conversation history.
    pub fn new(config: &'a AgentConfig, registry: &'a ToolRegistry) -> Self {
        let history = ConversationHistory::new(config.context_token_limit);
        Self { config, registry, history }
    }

    /// Processes one LLM response, dispatching tool calls when the response is an Action.
    ///
    /// In production `llm_response` comes from an LLM API call.
    /// Accepting it as a parameter keeps the runner pure and fully testable without I/O.
    ///
    /// # Errors
    /// - [`AgentError::ParseError`] — response does not match any ReAct format.
    /// - [`AgentError::ToolNotFound`] — the requested tool is not in the registry.
    pub fn step(&mut self, llm_response: &str) -> Result<ReActStep, AgentError> {
        let step = parse_react_step(llm_response)?;
        let obs_msg = match &step {
            ReActStep::Action { tool, input } => {
                let result = self.registry.dispatch(tool, input)?;
                Some(Message::new(Role::Tool, result.output))
            }
            _ => None,
        };
        self.history.push_with_eviction(Message::assistant(llm_response));
        if let Some(obs) = obs_msg {
            self.history.push_with_eviction(obs);
        }
        Ok(step)
    }

    /// Runs a pre-scripted sequence of LLM responses, returning the final answer.
    ///
    /// Useful for offline simulation and deterministic testing.
    ///
    /// # Errors
    /// - [`AgentError::MaxIterationsExceeded`] — sequence exhausted before a `FinalAnswer`.
    /// - Any error from [`step`](Self::step).
    pub fn run_scripted(&mut self, responses: &[&str]) -> Result<String, AgentError> {
        for (i, response) in responses.iter().enumerate() {
            if i as u32 >= self.config.max_iterations {
                return Err(AgentError::MaxIterationsExceeded(self.config.max_iterations));
            }
            let step = self.step(response)?;
            if let ReActStep::FinalAnswer(answer) = step {
                return Ok(answer);
            }
        }
        Err(AgentError::MaxIterationsExceeded(self.config.max_iterations))
    }

    /// Returns a reference to the accumulated conversation history.
    pub fn history(&self) -> &ConversationHistory { &self.history }

    /// Returns the number of messages currently in the history.
    pub fn iteration_count(&self) -> usize { self.history.len() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::{ToolRegistry, ToolSpec};
    use crate::types::ToolResult;

    fn registry_with_echo() -> ToolRegistry {
        let mut reg = ToolRegistry::new();
        reg.register(
            ToolSpec::new("echo", "Echoes input", "{}"),
            Box::new(|input: &str| ToolResult {
                tool_name: "echo".into(),
                output: format!("echoed: {input}"),
                success: true,
            }),
        ).unwrap();
        reg
    }

    #[test]
    fn test_parse_thought() {
        let step = parse_react_step("Thought: I need to search for information").unwrap();
        assert!(matches!(step, ReActStep::Thought(_)));
    }

    #[test]
    fn test_parse_action_paren_syntax() {
        let step = parse_react_step("Action: echo(hello world)").unwrap();
        assert!(matches!(&step, ReActStep::Action { tool, input }
            if tool == "echo" && input == "hello world"));
    }

    #[test]
    fn test_parse_action_colon_syntax() {
        let step = parse_react_step("Action: echo: hello").unwrap();
        assert!(matches!(&step, ReActStep::Action { tool, .. } if tool == "echo"));
    }

    #[test]
    fn test_parse_final_answer() {
        let step = parse_react_step("Final Answer: 42").unwrap();
        assert!(matches!(&step, ReActStep::FinalAnswer(s) if s == "42"));
    }

    #[test]
    fn test_parse_unknown_format_returns_parse_error() {
        let err = parse_react_step("This is just random text without a prefix").unwrap_err();
        assert!(matches!(err, AgentError::ParseError(_)));
    }

    #[test]
    fn test_parse_action_no_args() {
        let step = parse_react_step("Action: mytool").unwrap();
        assert!(matches!(&step, ReActStep::Action { tool, .. } if tool == "mytool"));
    }

    #[test]
    fn test_loop_runner_scripted_final_answer() {
        let config = AgentConfig::default();
        let reg = registry_with_echo();
        let mut runner = LoopRunner::new(&config, &reg);
        let answer = runner.run_scripted(&[
            "Thought: Let me think",
            "Action: echo(test)",
            "Final Answer: done",
        ]).unwrap();
        assert_eq!(answer, "done");
    }

    #[test]
    fn test_loop_runner_max_iterations_returns_error() {
        let config = AgentConfig { max_iterations: 2, ..Default::default() };
        let reg = ToolRegistry::new();
        let mut runner = LoopRunner::new(&config, &reg);
        let err = runner.run_scripted(&[
            "Thought: step 1",
            "Thought: step 2",
            "Thought: step 3",
        ]).unwrap_err();
        assert!(matches!(err, AgentError::MaxIterationsExceeded(2)));
    }

    #[test]
    fn test_loop_runner_tool_not_found_returns_error() {
        let config = AgentConfig::default();
        let reg = ToolRegistry::new();
        let mut runner = LoopRunner::new(&config, &reg);
        let err = runner.step("Action: missing_tool(input)").unwrap_err();
        assert!(matches!(err, AgentError::ToolNotFound { .. }));
    }

    #[test]
    fn test_loop_runner_history_grows_with_steps() {
        let config = AgentConfig::default();
        let reg = registry_with_echo();
        let mut runner = LoopRunner::new(&config, &reg);
        runner.step("Thought: thinking").unwrap();
        runner.step("Thought: still thinking").unwrap();
        assert!(runner.history().len() >= 2);
    }

    #[test]
    fn test_loop_runner_action_adds_observation_to_history() {
        let config = AgentConfig::default();
        let reg = registry_with_echo();
        let mut runner = LoopRunner::new(&config, &reg);
        runner.step("Action: echo(hi)").unwrap();
        // Two messages: the Action response + the Observation
        assert!(runner.history().len() >= 2);
    }

    #[test]
    fn test_loop_runner_iteration_count_matches_history_len() {
        let config = AgentConfig::default();
        let reg = ToolRegistry::new();
        let mut runner = LoopRunner::new(&config, &reg);
        runner.step("Thought: one").unwrap();
        assert_eq!(runner.iteration_count(), runner.history().len());
    }
}
