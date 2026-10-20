#!/usr/bin/env python3
"""Run a small steering scale sweep through q36.

This is intentionally thin: it exercises the same public CLI options users
will use in production and leaves all inference behavior inside q36.
"""

import argparse
import math
import subprocess
from pathlib import Path


def read_prompts(path: Path) -> list[str]:
    prompts = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            prompts.append(line)
    if not prompts:
        raise SystemExit(f"{path}: no prompts found")
    return prompts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--q36", default="./q36")
    ap.add_argument("--model", default="gguf/Qwen3.8-27B-UD-IQ3_S.gguf")
    ap.add_argument("--direction", required=True,
                    help="flat f32 vector file produced by build_direction.py")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--scales", default="-2,-1,-0.5,0,0.5,1,2")
    ap.add_argument("--tokens", type=int, default=160)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--attn-scale", type=float, default=0.0)
    ap.add_argument("--nothink", action="store_true")
    args = ap.parse_args()

    prompts = read_prompts(Path(args.prompts))
    scales = [float(x) for x in args.scales.split(",") if x.strip()]

    if not scales or not all(math.isfinite(x) and -100 <= x <= 100
                             for x in [*scales, args.attn_scale]):
        ap.error("scales must be finite and between -100 and 100")
    q36 = str(Path(args.q36).resolve())

    for prompt in prompts:
        print("=" * 80)
        print(f"PROMPT: {prompt}")
        for scale in scales:
            print("-" * 80)
            print(f"FFN scale: {scale:g}", flush=True)
            cmd = [
                q36,
                "-m", args.model,
                "--ctx", str(args.ctx),
                "-n", str(args.tokens),
                "--temp", "0",
                "--dir-steering-file", args.direction,
                "--dir-steering-ffn", str(scale),
                "--dir-steering-attn", str(args.attn_scale),
                "-p", prompt,
            ]
            if args.nothink:
                cmd.append("--nothink")
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
