# Fast-dLLM on Apple M4 Max

## Environment

- Apple M4 Max, 40-core GPU
- 64 GB unified memory
- PyTorch MPS, BF16
- LLaDA-8B-Instruct
- GSM8K 5-shot
- batch size: 1
- generation length: 256
- block length: 32
- threshold: 0.75

## Rotated-K v2

Same 150 GSM8K samples:

- Average latency: 17.895 s/sample
- Throughput: 12.824 token/s
- NFE: 59.69/sample
- Flexible EM: 75.33%
- Strict EM: 34.00%
- Speedup over original Fast-dLLM: 1.251x
- No aggregate accuracy or NFE regression

## Request-local v2b

Five-run A/B:

- v2: 20.025 s
- v2b: 20.016 s
- Difference: 0.05%, within measurement noise

v2b is an engineering refactor rather than a performance optimization:

- removes global POSITION_CONTEXT
- uses request-local ContextVar state
- passes explicit block bounds
- removes nonzero/max position recovery
- uses contiguous KV-cache replacement

## Cross-block cache refresh

Refresh interval 2, same 150 GSM8K samples:

- Total time: 1814.266 s
- Average latency: 12.095 s/sample
- NFE: 65.11/sample
- Flexible EM: 78.00%
- Strict EM: 38.00%
- Speedup over Rotated-K v2: 1.480x
- Latency reduction over Rotated-K v2: 32.41%
- Speedup over original Fast-dLLM: approximately 1.851x

This is an approximate inference optimization because KV states between
full refreshes may be stale. On this 150-sample subset, no accuracy
regression was observed, but the apparent accuracy increase should not
be interpreted as statistically significant.
