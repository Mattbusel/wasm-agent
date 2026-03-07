# wasm-agent

ReAct agent loop for WASM and edge environments — Thought-Action-Observation with tool dispatch.

A minimal, dependency-light agent loop that runs in WebAssembly, Cloudflare Workers, Fastly Compute, and any environment where you can't use OS threads or a full Tokio runtime.

## What's inside

- **ReActLoop** — Thought → Action → Observation cycle with configurable max iterations
- **ToolRegistry** — register sync or async tools as closures; dispatch by name
- **AgentContext** — scratchpad memory for the current reasoning chain
- **StopCondition** — pluggable termination: goal reached, max steps, timeout, error threshold
- **Trace** — structured log of every thought/action/observation for debugging

## Features

- **WASM-compatible** — no OS threads, no `std::time`, no filesystem
- **Tool dispatch** — tools are plain functions; no macros or derive required
- **Traceable** — full reasoning trace returned alongside the final answer
- **Resumable** — serialize agent state mid-loop and resume later

## Quick start

```rust
use wasm_agent::{ReActLoop, ToolRegistry};

let mut tools = ToolRegistry::new();
tools.register("search", |query: &str| {
    format!("Results for: {}", query)
});

let agent = ReActLoop::new(tools)
    .max_steps(10);

let result = agent.run("What is the capital of France?").await?;
println!("Answer: {}", result.answer);
println!("Steps taken: {}", result.trace.len());
```

## Add to your project

```toml
[dependencies]
wasm-agent = { git = "https://github.com/Mattbusel/wasm-agent" }
```

## Test coverage

```bash
cargo test
```

---

> Used inside [tokio-prompt-orchestrator](https://github.com/Mattbusel/tokio-prompt-orchestrator) -- a production Rust orchestration layer for LLM pipelines. See the full [primitive library collection](https://github.com/Mattbusel/rust-crates).