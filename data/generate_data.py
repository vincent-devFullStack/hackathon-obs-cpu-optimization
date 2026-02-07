#!/usr/bin/env python3
import argparse
import array
import hashlib
import json
import os
import random
import sys
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_float64_file(path: Path, total_values: int, rng: random.Random, chunk_size: int = 1 << 15) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        remaining = total_values
        while remaining > 0:
            n = min(chunk_size, remaining)
            buf = array.array("d", (rng.random() for _ in range(n)))
            if sys.byteorder != "little":
                buf.byteswap()
            f.write(buf.tobytes())
            remaining -= n


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate deterministic cosine benchmark dataset.")
    parser.add_argument("--output-dir", default="data", help="Directory where dataset files are written.")
    parser.add_argument("--metadata", default="metadata.json", help="Metadata JSON filename.")
    parser.add_argument("--e-file", default="E.f64", help="Embeddings binary filename.")
    parser.add_argument("--a-file", default="A.f64", help="Axis binary filename.")
    parser.add_argument("--N", type=int, default=2000, help="Number of embedding vectors.")
    parser.add_argument("--M", type=int, default=15, help="Number of axis vectors.")
    parser.add_argument("--D", type=int, default=96, help="Vector dimension.")
    parser.add_argument("--seed", type=int, default=42, help="PRNG seed.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files.")
    args = parser.parse_args()

    if args.N <= 0 or args.M <= 0 or args.D <= 0:
        raise SystemExit("N, M, and D must be positive integers.")

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = out_dir / args.metadata
    e_path = out_dir / args.e_file
    a_path = out_dir / args.a_file

    if not args.force:
        for p in (metadata_path, e_path, a_path):
            if p.exists():
                raise SystemExit(f"File already exists: {p}. Use --force to overwrite.")

    rng = random.Random(args.seed)

    write_float64_file(e_path, args.N * args.D, rng)
    write_float64_file(a_path, args.M * args.D, rng)

    e_sha = sha256_file(e_path)
    a_sha = sha256_file(a_path)

    metadata = {
        "format": "cosine-benchmark-v1",
        "dtype": "float64-le",
        "N": args.N,
        "M": args.M,
        "D": args.D,
        "seed": args.seed,
        "E_file": args.e_file,
        "A_file": args.a_file,
        "sha256": {
            "E_f64": e_sha,
            "A_f64": a_sha,
        },
    }

    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Wrote metadata: {metadata_path}")
    print(f"Wrote E: {e_path} ({e_path.stat().st_size} bytes)")
    print(f"Wrote A: {a_path} ({a_path.stat().st_size} bytes)")
    print(f"sha256(E)={e_sha}")
    print(f"sha256(A)={a_sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
