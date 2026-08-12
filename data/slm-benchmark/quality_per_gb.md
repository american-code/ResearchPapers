# SLM Coding Efficiency — Phase 1 Benchmark

*Run: 2026-08-06 | Hardware: Apple Silicon (lab-02, 32 GB) | Quant: 4-bit MLX*

## Results

| Model | Params | Mem GB | HumanEval | MBPP | Real/10 | Tok/s | HE/GB | MBPP/GB |
|-------|--------|--------|-----------|------|---------|-------|-------|---------|
| Qwen2.5-Coder-1.5B-Instruct | 1.5B | 1.3 | 0.579 | 0.047 | 0.40 | 162 | 0.4561 | 0.0368 |
| Qwen2.5-Coder-3B-Instruct | 3.0B | 2.2 | 0.756 | 0.062 | 0.60 | 92 | 0.3360 | 0.0277 |
| Qwen2.5-Coder-7B-Instruct | 7.0B | 4.7 | 0.555 | 0.070 | 0.80 | 51 | 0.1183 | 0.0149 |
| Phi-4-Mini-Instruct | 3.8B | — | ERROR | — | — | — | — | — |
| DeepSeek-Coder-V2-Lite-Instruct | 16.0B | — | ERROR | — | — | — | — | — |
| Granite-Code-8B-Instruct | 8.0B | 9.2 | 0.579 | 0.039 | 0.70 | 28 | 0.0630 | 0.0042 |

## Notes
- **HumanEval**: pass@1 greedy, 164 problems (`openai/openai_humaneval`)
- **MBPP**: pass@1 greedy, MBPP-sanitized test split (257 problems)
- **Real/10**: 10 research-codebase tasks (l2_normalize, cosine_sim, bootstrap CI,
  top-k, IOI logit diff, Procrustes, JSON merge, code extraction, log parsing, power analysis)
- **Tok/s**: mean generation throughput across all suites
- **HE/GB / MBPP/GB**: pass@1 ÷ peak model memory (quality-per-GB efficiency)
- HumanEval results for 5 models from v1 run (2026-08-05); MBPP + real tasks from v2 run (2026-08-06)
