from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from ase.io import iread, write
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add a random per-structure force_train_mask to an extxyz file."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--key", default="force_train_mask")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.fraction <= 1.0:
        raise ValueError("--fraction must be in (0, 1].")
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if args.output.exists() and not args.overwrite:
        print(f"Force-subset file already exists: {args.output}")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = args.output.with_suffix(args.output.suffix + f".tmp.{os.getpid()}")
    tmp_output.unlink(missing_ok=True)

    rng = np.random.default_rng(args.seed)
    n_frames = 0
    n_force = 0
    try:
        for atoms in tqdm(iread(args.input, index=":"), desc="mark force subset"):
            use_forces = bool(rng.random() < args.fraction)
            atoms.info[args.key] = int(use_forces)
            n_frames += 1
            n_force += int(use_forces)
            write(tmp_output, atoms, format="extxyz", append=True)
        tmp_output.replace(args.output)
    except Exception:
        tmp_output.unlink(missing_ok=True)
        raise

    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "key": args.key,
        "fraction": args.fraction,
        "seed": args.seed,
        "frames": n_frames,
        "force_frames": n_force,
        "actual_fraction": n_force / n_frames if n_frames else 0.0,
    }
    with args.output.with_suffix(args.output.suffix + ".summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
