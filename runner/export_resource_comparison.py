#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

def mem_total_kb() -> float:
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return float(parts[1])
    except Exception:
        return 0.0
    return 0.0



def load_summary(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"summary not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def ns_to_ms(value: float) -> float:
    return float(value) / 1e6


def kb_to_mb(value: float) -> float:
    return float(value) / 1024.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Export resource usage comparison from summary.json")
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    summary = load_summary(Path(args.summary_json))
    impls = summary.get("implementations", [])

    total_kb = mem_total_kb()
    entries = []
    for impl in impls:
        median_wall_ns = impl.get("median_wall_ns")
        median_cpu_ns = impl.get("median_cpu_ns")
        proc = impl.get("process_metrics") or {}

        max_rss_kb = proc.get("max_rss_kb")
        max_threads = proc.get("max_threads")

        cpu_usage_pct = None
        if median_wall_ns and median_cpu_ns and float(median_wall_ns) > 0:
            cpu_usage_pct = (float(median_cpu_ns) / float(median_wall_ns)) * 100.0

        ram_usage_pct = None
        if max_rss_kb is not None and total_kb > 0:
            ram_usage_pct = (float(max_rss_kb) / float(total_kb)) * 100.0

        entries.append(
            {
                "code_language": impl.get("language"),
                "implementation": impl.get("impl_id"),
                "speedup_vs_baseline": impl.get("speedup_vs_baseline_wall"),
                "cpu_processing_time_ms": ns_to_ms(median_cpu_ns) if median_cpu_ns is not None else None,
                "max_ram_mb": kb_to_mb(max_rss_kb) if max_rss_kb is not None else None,
                "ram_usage_percent": ram_usage_pct,
                "cpu_usage_percent": cpu_usage_pct,
                "cores_used": int(max_threads) if max_threads is not None else None,
                "_sort_wall_ns": float(median_wall_ns) if median_wall_ns is not None else float("inf"),
            }
        )

    entries.sort(key=lambda x: x["_sort_wall_ns"])
    for entry in entries:
        entry.pop("_sort_wall_ns", None)

    output = {
        "source_summary": str(Path(args.summary_json).resolve()),
        "entries": entries,
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
