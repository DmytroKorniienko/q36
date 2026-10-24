# BC-250, 2026-10-24

Resident Vulkan measurements at commit `da0af84`, using RADV, Mesa 26.2.5,
and a BC-250 with 16 GB unified GDDR6. The benchmark was rebuilt with
`make -j4 q36-bench`. All runs are serialized and use warmed weights,
Q8_0 keys, Q4_0 values, 1024-token prefill chunks, and 128 greedy output
tokens per frontier.

| Model | Plain decode and prefill | MTP decode and prefill |
| --- | --- | --- |
| Qwen3.6-35B-A3B IQ2XXS/Q2K/Q8 | [CSV](qwen36-plain.csv) | [CSV](qwen36-mtp.csv) |
| Qwen3.8-27B GSQ-RCO IQ3_XXS | [CSV](qwen38-plain.csv) | [CSV](qwen38-mtp.csv) |

Prefill throughput measures the interval since the previous frontier, rather
than a whole-prefix average. Snapshot restore replays the prefix outside the
timing windows. MTP uses the embedded head, draft limit eight and margin zero;
its benefit depends on the generated continuation.

Reproduce from the repository root:

```sh
make -j4 q36-bench
python3 - <<'PYTHON'
from pathlib import Path
story = Path('tests/long_context_story_prompt.txt').read_text()
Path('/tmp/q36-story-64k.txt').write_text(story + '\n\n' + story)
PYTHON

for name in qwen36 qwen38; do
    case "$name" in
        qwen36) model=gguf/Qwen3.6-35B-A3B-IQ2XXS-Q2K-Q8.gguf ;;
        qwen38) model=gguf/Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf ;;
    esac
    for mode in plain mtp; do
        set --
        if [ "$mode" = mtp ]; then
            set -- --mtp "$model" --mtp-draft 8 --mtp-margin 0
        fi
        ./q36-bench -m "$model" --vulkan \
            --prompt-file /tmp/q36-story-64k.txt \
            --ctx-start 2048 --ctx-max 65536 --step-mul 2 \
            --gen-tokens 128 --warm-weights -ctk q8_0 -ctv q4_0 \
            "$@" --csv "speed-bench/bc250-20261024/$name-$mode.csv"
    done
done
```

The original fixture has 34,392 tokens; two copies extend it beyond 64K.
The recorded order was Qwen3.6 MTP, Qwen3.6 plain, Qwen3.8 plain,
Qwen3.8 MTP. The commands above group plain and MTP runs for each model.

Generate the README graphs from the recorded CSVs:

```sh
python3 speed-bench/plot_speed.py speed-bench/bc250-20261024/qwen36-plain.csv \
    --mtp-csv speed-bench/bc250-20261024/qwen36-mtp.csv \
    --title "BC-250 · Qwen3.6-35B-A3B" \
    -o speed-bench/bc250-20261024/qwen36_ts.svg
python3 speed-bench/plot_speed.py speed-bench/bc250-20261024/qwen38-plain.csv \
    --mtp-csv speed-bench/bc250-20261024/qwen38-mtp.csv \
    --title "BC-250 · Qwen3.8-27B" \
    -o speed-bench/bc250-20261024/qwen38_ts.svg
```

Prefill uses the left axis. Plain and MTP decode share the right axis.
