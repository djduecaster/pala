# PALA Future Ideas

Last updated: 2026-03-01

## Long-Context Cosmos Experiments (Post-Demo)
- Keep demo path on short prompts for predictable latency and reliability.
- If time permits after demo stabilization, evaluate longer context windows on H100 with cache-aware request design.
- Benchmark three traffic modes before adopting:
  - Cold cache (first request / no shared prefix reuse)
  - Warm cache (stable shared prefix)
  - Mixed traffic (realistic request churn)
- Compare latency + quality at input sizes such as 1k, 5k, 10k, 20k, and 50k tokens.
- Only adopt longer-context mode if it preserves behavior-loop responsiveness and does not reduce demo stability.
