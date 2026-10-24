# Performance and Benchmarking

[README](../README.md#speed)

## BC-250 context sweeps

The [README](../README.md#speed) reports the two resident model sweeps recorded
on 2026-10-24 at commit `da0af84`, using Vulkan/RADV (Mesa 26.2.5). Raw CSVs
and reproduction commands are in [the benchmark directory](../speed-bench/bc250-20261024/README.md).

Each process allocates 65,665 tokens and measures frontiers from 2K through
64K, doubling each time. Prefill uses 1024-token chunks and measures only the
new interval. Decode generates 128 greedy tokens, excluding EOS. Snapshot
restore replays the prefix outside both timing windows. Models are warmed
before measurement, and only one model process runs at a time.

MTP uses the embedded head, draft ceiling eight, and adaptive confidence
(`--mtp-margin 0`). Its throughput depends on the continuation and acceptance
rate. Plain decode and MTP use separate sweeps over the same prompt. The
README's prefill column comes from the plain sweep.

The original story fixture has 34,392 tokens. These sweeps concatenate two
copies with two newlines between them to reach 64K. Results describe this
fixed workload and build.

## Metal measurements

The following single-process measurements used the default q2 GGUF fully
resident and the mixed q2-q4 GGUF with SSD streaming on a 16 GB M2 Pro with
macOS 26.5. The SSD run used the automatic 4240-expert (6.99 GiB) cache. Both
runs used [the story fixture](../tests/long_context_story_prompt.txt),
incremental prefill at doubling context frontiers, greedy decoding with 128 generated tokens per frontier, and
Q8_0 K / Q4_0 V cache:

| Runtime | Quant | Expert residency | Prompt | Prefill | Generation |
| --- | ---: | --- | ---: | ---: | ---: |
| Metal | q2 | Full model resident | 2048 ctx | 448.75 t/s | 37.78 t/s |
| Metal | q2 | Full model resident | 4096 ctx | 366.19 t/s | 34.93 t/s |
| Metal | q2 | Full model resident | 8192 ctx | 270.02 t/s | 31.08 t/s |
| Metal | q2 | Full model resident | 16384 ctx | 177.21 t/s | 25.64 t/s |
| Metal SSD | q2-q4 | 4240 expert slots / 6.99 GiB | 4096 ctx | 58.74 t/s | 8.97 t/s |
| Metal SSD | q2-q4 | 4240 expert slots / 6.99 GiB | 8192 ctx | 55.32 t/s | 9.20 t/s |

![M2 Pro Metal Q2 t/s](../speed-bench/metal_q2_ts.svg)

These numbers show the expected capacity tradeoff, not a cross-machine
performance promise. Metal generation, prefill, and SSD behavior should be
remeasured on the oldest and smallest supported Mac before a release.

## Historical BC-250 measurements

These are the earlier Qwen3.6 measurements retained from the README.
For the current models through 64K, see the [Speed section](../README.md#speed).

Benchmarks on a **BC-250** (40 CUs unlocked, Cyan-Skillfish Governor performance
mode at 500-2000 MHz, 85 °C thermal ceiling — a hot summer in Italy). Runs
use greedy decoding, `--gen-tokens 128`, the extended long-context story
prompt, and Q8_0 K / Q4_0 V cache. The q2 model is fully resident. The mixed
q2-q4 model uses SSD streaming with the automatic 5724-expert (4.72 GiB)
cache. Each row was measured in a separate process, with only one benchmark
process running at a time.

| Machine | Quant | Mode | Prompt | Prefill | Generation |
| --- | ---: | --- | ---: | ---: | ---: |
| BC-250 (40 CU) | q2 | Resident | 2048 ctx | 639.85 t/s | 81.85 t/s |
| BC-250 (40 CU) | q2 | Resident | 4096 ctx | 597.10 t/s | 79.74 t/s |
| BC-250 (40 CU) | q2 | Resident | 8192 ctx | 501.50 t/s | 74.72 t/s |
| BC-250 (40 CU) | q2 | Resident | 16384 ctx | 373.32 t/s | 65.04 t/s |
| BC-250 (40 CU) | q2 | Resident | 24576 ctx | 287.24 t/s | 56.17 t/s |
| BC-250 (40 CU) | q2 | Resident | 32768 ctx | 244.06 t/s | 51.26 t/s |
| BC-250 (40 CU) | q2-q4 | SSD streaming | 4096 ctx | 45.43 t/s | 14.52 t/s |
| BC-250 (40 CU) | q2-q4 | SSD streaming | 8192 ctx | 48.78 t/s | 13.98 t/s |

![BC-250 Q2 t/s](../speed-bench/bc250_ts.svg)

Use `q36-bench` for reproducible prefill and decode measurements. Release
builds also have a conservative BC-250 performance gate under
`make benchmark-gate`; record results on the same board and power state when
comparing changes.
