// SPDX-License-Identifier: MIT
use wasm_agent::error::AgentError;
use wasm_agent::types::ToolResult;
use wasm_agent::{AgentConfig, LoopRunner, ReActStep, ToolRegistry, ToolSpec};

fn build_registry() -> ToolRegistry {
    let mut reg = ToolRegistry::new();
    reg.register(
        ToolSpec::new("calculator", "Evaluates simple expressions", r#"{"type":"string"}"#),
        Box::new(|input: &str| {
            let output = format!("Result of '{input}': 42");
            ToolResult { tool_name: "calculator".into(), output, success: true }
        }),
    )
    .unwrap();
    reg.register(
        ToolSpec::new("lookup", "Looks up information", "{}"),
        Box::new(|input: &str| ToolResult {
            tool_name: "lookup".into(),
            output: format!("Info about: {input}"),
            success: true,
        }),
    )
    .unwrap();
    reg
}

#[test]
fn test_full_react_loop_two_tools() {
    let config = AgentConfig::default();
    let reg = build_registry();
    let mut runner = LoopRunner::new(&config, &reg);
    let answer = runner
        .run_scripted(&[
            "Thought: I need to look something up first",
            "Action: lookup(France)",
            "Thought: Now I'll calculate",
            "Action: calculator(2 + 2)",
            "Final Answer: The answer is 42",
        ])
        .unwrap();
    assert_eq!(answer, "The answer is 42");
}

#[test]
fn test_react_loop_with_parse_errors_propagates() {
    let config = AgentConfig::default();
    let reg = ToolRegistry::new();
    let mut runner = LoopRunner::new(&config, &reg);
    let err = runner.step("random gibberish no prefix").unwrap_err();
    assert!(matches!(err, AgentError::ParseError(_)));
}

#[test]
fn test_react_loop_observation_recorded_after_action() {
    let config = AgentConfig::default();
    let reg = build_registry();
    let mut runner = LoopRunner::new(&config, &reg);
    runner.step("Action: calculator(1+1)").unwrap();
    // History must contain the action message and the tool observation
    assert!(runner.history().len() >= 2);
}

#[test]
fn test_tool_registry_tools_prompt_lists_all_tools() {
    let reg = build_registry();
    let prompt = reg.tools_prompt();
    assert!(prompt.contains("calculator"));
    assert!(prompt.contains("lookup"));
}

#[test]
fn test_max_iterations_respected_in_scripted_run() {
    let config = AgentConfig { max_iterations: 1, ..Default::default() };
    let reg = ToolRegistry::new();
    let mut runner = LoopRunner::new(&config, &reg);
    let err = runner
        .run_scripted(&["Thought: one", "Thought: two"])
        .unwrap_err();
    assert!(matches!(err, AgentError::MaxIterationsExceeded(1)));
}

#[test]
fn test_react_step_serialization_roundtrip() {
    let step = ReActStep::Action { tool: "search".into(), input: "query".into() };
    let json = serde_json::to_string(&step).unwrap();
    let back: ReActStep = serde_json::from_str(&json).unwrap();
    assert_eq!(step, back);
}
