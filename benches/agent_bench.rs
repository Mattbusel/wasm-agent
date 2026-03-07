// SPDX-License-Identifier: MIT
use criterion::{criterion_group, criterion_main, Criterion};
use wasm_agent::loop_runner::parse_react_step;
use wasm_agent::tools::{ToolRegistry, ToolSpec};
use wasm_agent::types::ToolResult;

fn bench_parse_react_step(c: &mut Criterion) {
    c.bench_function("parse_thought", |b| b.iter(|| {
        let _ = parse_react_step("Thought: I need to analyze this carefully");
    }));
    c.bench_function("parse_action", |b| b.iter(|| {
        let _ = parse_react_step("Action: search(what is the capital of France)");
    }));
    c.bench_function("parse_final_answer", |b| b.iter(|| {
        let _ = parse_react_step("Final Answer: The capital of France is Paris");
    }));
}

fn bench_tool_dispatch(c: &mut Criterion) {
    let mut reg = ToolRegistry::new();
    reg.register(
        ToolSpec::new("noop", "Does nothing", "{}"),
        Box::new(|_| ToolResult { tool_name: "noop".into(), output: "ok".into(), success: true }),
    ).unwrap();
    c.bench_function("tool_dispatch", |b| b.iter(|| {
        let _ = reg.dispatch("noop", "");
    }));
}

criterion_group!(benches, bench_parse_react_step, bench_tool_dispatch);
criterion_main!(benches);
