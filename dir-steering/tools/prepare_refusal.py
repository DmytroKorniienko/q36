#!/usr/bin/env python3
"""Prepare fixed, disjoint prompt splits from the paper authors' public corpus."""

import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from urllib.request import urlopen

COMMIT = "9d852fae1a9121c78b29142de733cb1340770cc3"
SOURCE = f"https://raw.githubusercontent.com/andyrdt/refusal_direction/{COMMIT}/dataset/splits"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-dir", help="read q36-{split}.json files here instead of downloading")
    ap.add_argument("--out", default="dir-steering/out/refusal/data")
    ap.add_argument("--test-offset", type=int, default=0,
                    help="skip previously evaluated test pairs while retaining train/validation splits")
    ap.add_argument("--validation-from",
                    help="use a previous prepared dataset's test split for calibration")
    args = ap.parse_args()
    if args.test_offset < 0:
        ap.error("--test-offset must be nonnegative")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)
    used = set()
    manifest = {"source": SOURCE, "commit": COMMIT, "seed": 42,
                "test_offset": args.test_offset, "splits": {}}
    groups = {"train": 128, "val": 16, "test": 32}
    for split, count in groups.items():
        selected = {}
        for group in ("harmful", "harmless"):
            name = f"{group}_{split}"
            if args.source_dir:
                raw = (Path(args.source_dir) / f"q36-{name}.json").read_bytes()
            else:
                with urlopen(f"{SOURCE}/{name}.json", timeout=60) as response:
                    raw = response.read()
            source = json.loads(raw)
            order = list(range(len(source)))
            rng.shuffle(order)
            rows = []
            skip = args.test_offset if split == "test" else 0
            for index in order:
                item = source[index]
                text = " ".join(item["instruction"].split())
                category = str(item.get("category") or "")
                if not text or text.startswith("#") or len(text) > 640 or text in used:
                    continue
                if group == "harmful" and re.search(
                        r"sexual|porn|underage|minor|child|rape|chemical_biological|bioweapon|anthrax|ricin|sarin|nuclear",
                        text + " " + category, re.I):
                    continue
                used.add(text)
                if skip:
                    skip -= 1
                    continue
                rows.append({"instruction": text, "source_index": index,
                             "category": item.get("category"), "group": group})
                if len(rows) == count:
                    break
            if len(rows) != count:
                raise SystemExit(f"{name}: only {len(rows)} eligible prompts")
            manifest["splits"][name] = {"count": count,
                "raw_sha256": hashlib.sha256(raw).hexdigest(),
                "source_indices": [row["source_index"] for row in rows]}
            selected[group] = rows
            (out / f"{name}.txt").write_text(
                "\n".join(row["instruction"] for row in rows) + "\n", encoding="utf-8")
        combined = selected["harmful"] + selected["harmless"]
        (out / f"{split}.txt").write_text(
            "\n".join(row["instruction"] for row in combined) + "\n", encoding="utf-8")
        (out / f"{split}.json").write_text(json.dumps(combined, indent=2) + "\n")
        print(f"{split}: {count} target + {count} control prompts")
    if args.validation_from:
        previous = Path(args.validation_from)
        prior = json.loads((previous / "manifest.json").read_text())
        if prior["source"] != SOURCE:
            raise SystemExit("validation corpus must use the same source commit")
        for suffix in ("txt", "json"):
            (out / f"val.{suffix}").write_bytes((previous / f"test.{suffix}").read_bytes())
        for group in ("harmful", "harmless"):
            (out / f"{group}_val.txt").write_bytes((previous / f"{group}_test.txt").read_bytes())
            manifest["splits"][f"{group}_val"] = {
                **prior["splits"][f"{group}_test"], "source_split": f"{group}_test"}
        manifest["validation_source"] = str(previous / "test.json")
        print("validation: using the previous test split for calibration")
    seen = set()
    for split in groups:
        prompts = (out / f"{split}.txt").read_text().splitlines()
        if len(prompts) != len(set(prompts)) or seen.intersection(prompts):
            raise SystemExit("splits overlap; reserve a later --test-offset")
        seen.update(prompts)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
