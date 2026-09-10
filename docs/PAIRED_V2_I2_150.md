# Paired Analysis: Rotated-K v2 vs Interval-2 Cache Refresh

## Setting

- LLaDA-8B-Instruct
- Apple M4 Max / PyTorch MPS / BF16
- GSM8K 5-shot
- Same paired 150 samples
- gen_length=256
- block_length=32
- threshold=0.75

## Performance

- Rotated-K v2: 17.895 s/sample
- Interval=2: 12.095 s/sample
- Speedup: 1.480x
- Latency reduction: 32.41%

## Flexible exact match

- Both correct: 103
- v2 correct / interval=2 wrong: 10
- v2 wrong / interval=2 correct: 14
- Both wrong: 23
- v2 EM: 75.33%
- interval=2 EM: 78.00%
- McNemar exact two-sided p: 0.541256

## Strict exact match

- Both correct: 25
- v2 correct / interval=2 wrong: 26
- v2 wrong / interval=2 correct: 32
- Both wrong: 67
- v2 EM: 34.00%
- interval=2 EM: 38.00%
- McNemar exact two-sided p: 0.511842

## Trajectory divergence

- Flexible final-answer string agreement: 65.33%
- Flexible discordant pairs: 24
- Formatting-only numeric-equivalent flips: 2
- True numeric-answer changes: 22
- Mean paired NFE delta: +5.413
- Median paired NFE delta: +4
- Interval=2 had higher NFE on 111/150 samples

## Degradation gates

Both configurations had:

- 0 residual-mask samples
- 0 empty responses
- 0 replacement-character samples

## Interpretation

Interval=2 is an approximate inference optimization, not a mathematically
equivalent transformation. It materially changes individual decoding
trajectories and numerical answers.

On this paired 150-sample GSM8K subset, no statistically detectable
aggregate accuracy degradation was observed. The apparent EM increase
is not statistically significant and must not be described as an
accuracy improvement.

Validated scope:

- GSM8K
- threshold=0.75
- gen_length=256
- block_length=32
