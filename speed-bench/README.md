## Benchmarking

Here we collect prefill and generation speed obtained with the BC-250.

Run `q36-bench` as:

```
./q36-bench \
  -m gguf/Qwen3.6-35B-A3B-AntirezExperts-IQ2XXS-gateup-Q2K-down-Q8rest.gguf \
  --prompt-file tests/long_context_story_prompt.txt \
  --ctx-start 2048 \
  --ctx-max 32768 \
  --step-incr 2048 \
  --gen-tokens 128 \
  -ctk q8_0 -ctv q4_0 \
  --csv speed-bench/bc250.csv
```

The current tokenizer produces 34392 tokens from the story fixture, making
32768 the last complete 2048-token frontier.

To generate an SVG graph from a CSV file:

```
python3 speed-bench/plot_speed.py speed-bench/bc250.csv --title "BC-250 Q2 t/s"
```

## Concurrent sessions

Build and run one concurrency/context cell per process:

```sh
make speed-bench/session_concurrency_bench
./speed-bench/session_concurrency_bench \
  -m gguf/Qwen3.6-35B-A3B-IQ2XXS-Q2K-Q8.gguf \
  --prompt-file tests/long_context_story_prompt.txt \
  --concurrency 2 --ctx 128 --gen-tokens 64 --prefill-chunk 128 --verify
```

The Vulkan harness staggers each stream's position in the corpus, warms the decode
pipelines with sixteen tokens, then rebuilds the prompts before timing. It
reports aggregate and per-stream throughput, step latency percentiles, and GPU
memory accounting. Repeat cells at 128, 2048, and 4096 prompt tokens and widths
1, 2, and 4; serialize all runs as required by [SPEEDUP.md](../SPEEDUP.md).
Use `--cold` to measure the first decode without the warmup.

`--append-tokens 128` resumes another prefilled session during decoding and
reports the combined step's delay. `--mode serial-spec --mtp` measures the
existing per-session speculative cycles. Verification runs outside the timing
window and compares every output token with ordinary single-stream greedy
decode.

Incomplete work and verification failures return an error. Exit status 77
means the cell exceeds the conservative memory budget and was skipped before
allocating sessions.

The existing resident batch decoder can disagree with single-stream greedy
output at longer contexts. Two streams with 4096 story tokens, 32 output
tokens, and prefill chunk 1024 reproduce a mismatch at stream 1, output index
13, with both warm and `--cold` runs. Keep `--verify` enabled when evaluating
batching changes.
