#!/usr/bin/env python3
import argparse
import csv
import gc
import hashlib
import json
import math
import os
import platform
import shlex
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from metrics_linux import (
    THREAD_ENV_VARS,
    perf_available,
    run_perf_stat,
    run_with_metrics,
    wait_for_stable_system,
)

ABS_TOL = 1e-9
REL_TOL = 1e-9

JAVA_PROFILES: Dict[str, List[str]] = {
    "strict_single_core": [
        "-Xms2g",
        "-Xmx2g",
        "-XX:ActiveProcessorCount=1",
        "-XX:+UseSerialGC",
        "-XX:ParallelGCThreads=1",
    ],
    "relaxed_single_core": [
        "-Xms2g",
        "-Xmx2g",
        "-XX:ActiveProcessorCount=1",
        "-XX:+UseSerialGC",
    ],
}

# Disallowed JVM flags:
# -XX:CICompilerCount=1 is known to be invalid on some JVM builds and can fail startup.
BLOCKED_JAVA_OPTS = {"-XX:CICompilerCount=1"}


@dataclass
class Implementation:
    impl_id: str
    language: str
    requested_version: str
    command: List[str]


def warn(msg: str, warnings: List[str]) -> None:
    print(f"[WARN] {msg}", file=sys.stderr)
    warnings.append(msg)


def tail_lines(text: str, n: int = 20) -> str:
    lines = text.splitlines()
    return "\n".join(lines[-n:])


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_concat(paths: Sequence[Path]) -> str:
    h = hashlib.sha256()
    for p in paths:
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
    return h.hexdigest()


def median(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    return float(statistics.median(values))


def collect_tool_versions() -> Dict[str, Optional[str]]:
    def cmd_line(cmd: List[str], stderr: bool = False) -> Optional[str]:
        try:
            p = subprocess.run(cmd, capture_output=True, text=True)
        except Exception:
            return None
        if p.returncode != 0:
            return None
        data = p.stderr if stderr else p.stdout
        lines = [x.strip() for x in data.splitlines() if x.strip()]
        return lines[0] if lines else None

    return {
        "python": sys.version.replace("\n", " "),
        "kernel": platform.release(),
        "gcc": cmd_line(["gcc", "--version"]),
        "g++": cmd_line(["g++", "--version"]),
        "java": cmd_line(["java", "-version"], stderr=True),
        "go": cmd_line(["go", "version"]),
        "cargo": cmd_line(["cargo", "--version"]),
    }


def load_metadata(metadata_path: Path) -> Tuple[Dict[str, object], Path, Path, str, str, str]:
    if not metadata_path.exists():
        raise RuntimeError(f"metadata file not found: {metadata_path}")

    meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    required = ["format", "dtype", "N", "M", "D", "E_file", "A_file"]
    for key in required:
        if key not in meta:
            raise RuntimeError(f"metadata missing key: {key}")

    if meta["format"] != "cosine-benchmark-v1":
        raise RuntimeError(f"unsupported metadata format: {meta['format']}")
    if meta["dtype"] != "float64-le":
        raise RuntimeError(f"unsupported metadata dtype: {meta['dtype']}")

    data_dir = metadata_path.parent
    e_path = data_dir / str(meta["E_file"])
    a_path = data_dir / str(meta["A_file"])
    if not e_path.exists() or not a_path.exists():
        raise RuntimeError("dataset files referenced in metadata are missing")

    e_sha = sha256_file(e_path)
    a_sha = sha256_file(a_path)
    dataset_sha = sha256_concat([e_path, a_path])

    if isinstance(meta.get("sha256"), dict):
        expected_e = meta["sha256"].get("E_f64")
        expected_a = meta["sha256"].get("A_f64")
        if expected_e and expected_e != e_sha:
            raise RuntimeError(f"metadata sha mismatch for E: expected {expected_e} got {e_sha}")
        if expected_a and expected_a != a_sha:
            raise RuntimeError(f"metadata sha mismatch for A: expected {expected_a} got {a_sha}")

    return meta, e_path, a_path, e_sha, a_sha, dataset_sha


def check_or_create_dataset_lock(lock_path: Path, dataset_sha: str) -> Dict[str, object]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if lock_path.exists():
        current = lock_path.read_text(encoding="utf-8").strip()
        if current != dataset_sha:
            raise RuntimeError(
                f"dataset lock mismatch in {lock_path}: lock={current}, current={dataset_sha}. "
                "Refuse to run to prevent incomparable results."
            )
        return {"path": str(lock_path), "exists": True, "created": False, "matched": True}

    lock_path.write_text(dataset_sha + "\n", encoding="utf-8")
    return {"path": str(lock_path), "exists": False, "created": True, "matched": True}


def python_numpy_available(python_exec: str) -> Tuple[bool, Optional[str]]:
    cmd = [python_exec, "-c", "import numpy as np; print(np.__version__)"]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True)
    except Exception:
        return False, None
    if p.returncode != 0:
        return False, None
    version = p.stdout.strip().splitlines()[0] if p.stdout.strip() else None
    return True, version


def java_version_probe(java_opts: Sequence[str], env: Dict[str, str]) -> Dict[str, object]:
    cmd = ["java", *java_opts, "-version"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    except Exception as exc:
        return {
            "ok": False,
            "returncode": -1,
            "stdout": "",
            "stderr": str(exc),
            "command": cmd,
        }
    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "command": cmd,
    }


def parse_java_runtime_info(version_text: str) -> Dict[str, Optional[str]]:
    lines = [ln.strip() for ln in version_text.splitlines() if ln.strip()]
    raw = "\n".join(lines)
    line1 = lines[0] if lines else None
    vm_line = lines[1] if len(lines) > 1 else None

    kind = "Unknown"
    text = raw.lower()
    if "openj9" in text:
        kind = "OpenJ9"
    elif "graalvm" in text:
        kind = "GraalVM"
    elif "hotspot" in text or "server vm" in text:
        kind = "HotSpot"

    return {
        "jvm_kind": kind,
        "line1": line1,
        "vm_line": vm_line,
        "raw": raw,
    }


def java_probe_has_unknown_option(stderr_text: str) -> bool:
    hay = stderr_text.lower()
    patterns = [
        "unrecognized vm option",
        "unrecognized option",
        "could not create the java virtual machine",
        "invalid -xx option",
    ]
    return any(p in hay for p in patterns)


def resolve_java_opts(
    requested_opts: Sequence[str],
    env: Dict[str, str],
    warnings: List[str],
) -> Dict[str, object]:
    if shutil.which("java") is None:
        raise RuntimeError("java command not found in PATH")

    base_probe = java_version_probe([], env)
    if not bool(base_probe.get("ok")):
        raise RuntimeError(f"java -version failed: {tail_lines(str(base_probe.get('stderr', '')), 20)}")

    runtime_info = parse_java_runtime_info(str(base_probe.get("stderr", "")) or str(base_probe.get("stdout", "")))
    requested = list(requested_opts)
    blocked = [opt for opt in requested if opt in BLOCKED_JAVA_OPTS]
    attempts: List[Dict[str, object]] = []
    fallback_reason: Optional[str] = None
    profile_used: Optional[str] = None
    effective: Optional[List[str]] = None

    if blocked:
        fallback_reason = f"blocked option(s) detected: {', '.join(blocked)}"
        warn(
            f"Java JVM opts rejected, falling back to strict_single_core ({fallback_reason})",
            warnings,
        )
        requested = []

    if requested:
        probe = java_version_probe(requested, env)
        attempts.append(
            {
                "name": "requested",
                "opts": requested,
                "ok": bool(probe.get("ok")),
                "returncode": int(probe.get("returncode", -1)),
                "stderr_tail": tail_lines(str(probe.get("stderr", "")), 20),
            }
        )
        if bool(probe.get("ok")):
            effective = requested
            profile_used = "requested"
        else:
            if java_probe_has_unknown_option(str(probe.get("stderr", ""))):
                fallback_reason = "requested JVM options include unrecognized/unsupported option(s)"
            else:
                fallback_reason = f"requested JVM options failed (exit={probe.get('returncode')})"
            warn(
                "Java JVM opts rejected, falling back to strict_single_core",
                warnings,
            )

    if effective is None:
        strict_opts = JAVA_PROFILES["strict_single_core"]
        strict_probe = java_version_probe(strict_opts, env)
        attempts.append(
            {
                "name": "strict_single_core",
                "opts": strict_opts,
                "ok": bool(strict_probe.get("ok")),
                "returncode": int(strict_probe.get("returncode", -1)),
                "stderr_tail": tail_lines(str(strict_probe.get("stderr", "")), 20),
            }
        )
        if bool(strict_probe.get("ok")):
            effective = list(strict_opts)
            profile_used = "strict_single_core"
        else:
            warn(
                "Java strict_single_core profile rejected, falling back to relaxed_single_core",
                warnings,
            )
            relaxed_opts = JAVA_PROFILES["relaxed_single_core"]
            relaxed_probe = java_version_probe(relaxed_opts, env)
            attempts.append(
                {
                    "name": "relaxed_single_core",
                    "opts": relaxed_opts,
                    "ok": bool(relaxed_probe.get("ok")),
                    "returncode": int(relaxed_probe.get("returncode", -1)),
                    "stderr_tail": tail_lines(str(relaxed_probe.get("stderr", "")), 20),
                }
            )
            if bool(relaxed_probe.get("ok")):
                effective = list(relaxed_opts)
                profile_used = "relaxed_single_core"
                if fallback_reason is None:
                    fallback_reason = "strict_single_core not supported on this JVM"
            else:
                raise RuntimeError(
                    "unable to find a valid JVM option set (requested/strict/relaxed all failed)"
                )

    return {
        "requested_opts": list(requested_opts),
        "effective_opts": effective or [],
        "fallback_reason": fallback_reason,
        "profile_used": profile_used,
        "jvm_runtime": runtime_info,
        "blocked_options": blocked,
        "attempts": attempts,
    }


def make_implementations(
    repo_root: Path,
    metadata_path: Path,
    warmup: int,
    runs: int,
    repeat: int,
    python_exec: str,
    include_python: bool,
    include_python_numpy: bool,
    java_opts: List[str],
) -> List[Implementation]:
    common = [
        "--metadata",
        str(metadata_path),
        "--warmup",
        str(warmup),
        "--runs",
        str(runs),
        "--repeat",
        str(repeat),
    ]

    impls: List[Implementation] = [
        Implementation(
            impl_id="c-naive",
            language="c",
            requested_version="naive",
            command=[str(repo_root / "c" / "benchmark_c"), *common],
        ),
        Implementation(
            impl_id="cpp-naive",
            language="cpp",
            requested_version="naive",
            command=[str(repo_root / "cpp" / "benchmark_cpp"), *common],
        ),
        Implementation(
            impl_id="rust-naive",
            language="rust",
            requested_version="naive",
            command=[str(repo_root / "rust" / "target" / "release" / "cosine_benchmark_rust"), *common],
        ),
        Implementation(
            impl_id="go-naive",
            language="go",
            requested_version="naive",
            command=[str(repo_root / "go" / "benchmark_go"), *common],
        ),
        Implementation(
            impl_id="java-naive",
            language="java",
            requested_version="naive",
            command=["java", *java_opts, "-cp", str(repo_root / "java"), "CosineBenchmark", *common],
        ),
    ]

    if include_python:
        impls.insert(
            0,
            Implementation(
                impl_id="python-naive",
                language="python",
                requested_version="naive",
                command=[python_exec, str(repo_root / "python" / "benchmark_python.py"), *common, "--version", "naive"],
            ),
        )

    if include_python and include_python_numpy:
        impls.insert(
            1,
            Implementation(
                impl_id="python-numpy",
                language="python",
                requested_version="numpy",
                command=[python_exec, str(repo_root / "python" / "benchmark_python.py"), *common, "--version", "numpy"],
            ),
        )

    return impls


def build_for_impls(repo_root: Path, env: Dict[str, str], impls: List[Implementation]) -> Tuple[List[Implementation], List[Dict[str, str]]]:
    build_steps = {
        "c": ("make", ["make", "-C", str(repo_root / "c")]),
        "cpp": ("make", ["make", "-C", str(repo_root / "cpp")]),
        "rust": ("cargo", ["cargo", "build", "--release", "--manifest-path", str(repo_root / "rust" / "Cargo.toml")]),
        "go": ("go", ["go", "build", "-o", str(repo_root / "go" / "benchmark_go"), str(repo_root / "go" / "main.go")]),
        "java": ("javac", ["javac", str(repo_root / "java" / "CosineBenchmark.java")]),
    }

    langs = sorted({i.language for i in impls if i.language != "python"})
    skipped: List[Dict[str, str]] = []
    skipped_langs = set()

    for lang in langs:
        if lang not in build_steps:
            continue
        tool, cmd = build_steps[lang]
        if shutil.which(tool) is None:
            skipped_langs.add(lang)
            skipped.append({"impl_id": f"{lang}-*", "error": f"skipped: required build tool not found ({tool})"})
            continue
        proc = subprocess.run(cmd, cwd=str(repo_root), env=env)
        if proc.returncode != 0:
            raise RuntimeError(f"build failed for {lang}: {' '.join(cmd)}")

    kept = [i for i in impls if i.language not in skipped_langs]
    return kept, skipped


def filter_runnable(impls: List[Implementation]) -> Tuple[List[Implementation], List[Dict[str, str]]]:
    kept: List[Implementation] = []
    skipped: List[Dict[str, str]] = []
    for impl in impls:
        if not impl.command:
            skipped.append({"impl_id": impl.impl_id, "error": "skipped: empty command"})
            continue
        cmd0 = impl.command[0]
        has_sep = ("/" in cmd0) or ("\\" in cmd0)
        exists = Path(cmd0).exists() if has_sep else (shutil.which(cmd0) is not None)
        if not exists:
            skipped.append({"impl_id": impl.impl_id, "error": f"skipped: command not found ({cmd0})"})
            continue
        kept.append(impl)
    return kept, skipped


def parse_output_strict(stdout: str, impl_id: str) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    lines = [ln.strip() for ln in stdout.splitlines() if ln.strip()]
    if not lines:
        raise ValueError("empty stdout")

    objs: List[Dict[str, object]] = []
    for idx, line in enumerate(lines):
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"non-JSON output at line {idx + 1}: {exc}") from exc
        if not isinstance(obj, dict):
            raise ValueError(f"JSON line {idx + 1} is not an object")
        objs.append(obj)

    meta = objs[0]
    if meta.get("type") != "meta":
        raise ValueError("first line must be a meta JSON object")
    if str(meta.get("impl", "")) != impl_id:
        raise ValueError(f"meta.impl mismatch: expected {impl_id}, got {meta.get('impl')}")

    runs: List[Dict[str, object]] = []
    for idx, obj in enumerate(objs[1:], start=2):
        if obj.get("type") != "run":
            raise ValueError(f"line {idx} is not a run object")
        if str(obj.get("impl", "")) != impl_id:
            raise ValueError(f"run.impl mismatch at line {idx}")
        runs.append(obj)

    if not runs:
        raise ValueError("no run objects found")
    return meta, runs


def require_keys(obj: Dict[str, object], keys: Sequence[str], where: str) -> None:
    for key in keys:
        if key not in obj:
            raise ValueError(f"missing key '{key}' in {where}")


def f64(v: object, name: str) -> float:
    try:
        return float(v)
    except Exception as exc:
        raise ValueError(f"invalid float for {name}: {v}") from exc


def i64(v: object, name: str) -> int:
    try:
        return int(v)
    except Exception as exc:
        raise ValueError(f"invalid integer for {name}: {v}") from exc


def stability_brief(report: Dict[str, object]) -> str:
    last = report.get("last", {}) if isinstance(report, dict) else {}
    metrics = last.get("metrics", {}) if isinstance(last, dict) else {}
    stable = bool(report.get("stable", False))
    waited = float(report.get("waited_sec", 0.0))
    attempts = int(report.get("attempts", 0))
    cpu = metrics.get("cpu_util_avg")
    load1 = metrics.get("load1_avg")
    io = metrics.get("disk_io_mbps_avg")
    mem = metrics.get("mem_available_mb_min")
    swap = metrics.get("swap_activity")
    return (
        f"stable={stable} waited={waited:.1f}s attempts={attempts} "
        f"cpu_avg={cpu} load1_avg={load1} io_avg_mb_s={io} mem_min_mb={mem} swap={swap}"
    )


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(description="Run all cosine benchmark implementations with strict verifications")
    parser.add_argument("--metadata", default=str(repo_root / "data" / "metadata.json"))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--repeat", type=int, default=50)

    parser.add_argument("--enforce-single-thread", action="store_true")
    parser.add_argument("--cpu-affinity", default=None, help="CPU affinity list, e.g. '2' or '2,3'")
    parser.add_argument("--pin-affinity", default=None, help="Alias of --cpu-affinity")
    parser.add_argument("--cpu-set", default=None, help="Deprecated alias for --cpu-affinity")
    parser.add_argument("--nice", type=int, default=None)

    parser.add_argument("--stability-enable", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stability-window-sec", type=float, default=6.0)
    parser.add_argument("--stability-sample-interval-sec", type=float, default=0.5)
    parser.add_argument("--stability-timeout-sec", type=float, default=60.0)
    parser.add_argument("--stability-backoff-sec", type=float, default=2.0)
    parser.add_argument("--stability-mode", choices=["wait", "skip", "fail"], default="wait")
    parser.add_argument("--cpu-util-max", type=float, default=20.0)
    parser.add_argument("--load1-max-factor", type=float, default=0.30)
    parser.add_argument("--run-queue-max-factor", type=float, default=0.30)
    parser.add_argument("--disk-io-mbps-max", type=float, default=5.0)
    parser.add_argument("--mem-available-min-mb", type=float, default=2048.0)
    parser.add_argument("--mem-available-min-percent", type=float, default=15.0)
    parser.add_argument("--swap-activity-max", type=float, default=0.0)
    parser.add_argument("--cpu-util-variance-max", type=float, default=50.0)
    parser.add_argument("--print-stability", action="store_true")

    parser.add_argument("--cooldown-sec", type=float, default=0.0)
    parser.add_argument("--gc-between", action="store_true")

    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--no-perf", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--timeout-sec", type=int, default=None)
    parser.add_argument("--baseline", default="python-naive")
    parser.add_argument("--impls", default="")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--disable-python", action="store_true", help="Disable python implementations")
    parser.add_argument("--no-python-numpy", action="store_true")
    parser.add_argument("--java-opts", default="")
    parser.add_argument("--dry-run", action="store_true", help="Validate preflight/build/runtime and exit without running benchmarks")

    parser.add_argument("--dataset-lock", default=str(repo_root / "results" / "dataset.lock"))
    parser.add_argument("--results-csv", default=str(repo_root / "results" / "results.csv"))
    parser.add_argument("--summary-json", default=str(repo_root / "results" / "summary.json"))

    args = parser.parse_args()

    cpu_affinity = args.cpu_affinity or args.pin_affinity or args.cpu_set

    if args.warmup < 0 or args.runs <= 0 or args.repeat <= 0:
        raise SystemExit("warmup must be >= 0, runs and repeat must be > 0")

    warnings: List[str] = []
    critical_errors: List[Dict[str, object]] = []
    stability_reports: Dict[str, Dict[str, object]] = {}

    metadata_path = Path(args.metadata).resolve()
    try:
        meta, e_path, a_path, e_sha, a_sha, dataset_sha = load_metadata(metadata_path)
    except Exception as exc:
        print(f"[FAIL] dataset metadata verification failed: {exc}", file=sys.stderr)
        return 1

    try:
        lock_info = check_or_create_dataset_lock(Path(args.dataset_lock).resolve(), dataset_sha)
    except Exception as exc:
        print(f"[FAIL] dataset lock verification failed: {exc}", file=sys.stderr)
        return 1

    env = os.environ.copy()
    thread_env_before = {k: env.get(k) for k in THREAD_ENV_VARS}

    if args.enforce_single_thread:
        for k in THREAD_ENV_VARS:
            env[k] = "1"
        # Go runtime may otherwise schedule work on multiple OS threads.
        env["GOMAXPROCS"] = "1"
    else:
        missing = [k for k in THREAD_ENV_VARS if env.get(k) is None]
        if missing:
            warn(
                "thread env vars missing (set to 1 for stable comparison): " + ", ".join(missing),
                warnings,
            )
        for k in THREAD_ENV_VARS:
            v = env.get(k)
            if v is not None and v != "1":
                warn(f"{k}={v} (multi-thread runtime may bias comparison)", warnings)

    thread_env_effective = {k: env.get(k) for k in THREAD_ENV_VARS}

    if cpu_affinity is None:
        warn("no CPU affinity set; measurements may be less stable", warnings)

    requested_java_opts = shlex.split(args.java_opts) if args.java_opts.strip() else []
    java_resolution: Dict[str, object] = {
        "requested_opts": requested_java_opts,
        "effective_opts": [],
        "fallback_reason": None,
        "profile_used": None,
        "jvm_runtime": None,
        "blocked_options": [],
        "attempts": [],
    }

    include_python = not args.disable_python
    include_python_numpy = include_python and (not args.no_python_numpy)
    if include_python_numpy:
        has_np, np_ver = python_numpy_available(args.python)
        if not has_np:
            include_python_numpy = False
            warn("numpy not available for selected python; python-numpy implementation skipped", warnings)
        elif np_ver:
            print(f"[INFO] numpy detected: {np_ver}")
    elif args.disable_python:
        warn("python implementations disabled by --disable-python", warnings)

    impls = make_implementations(
        repo_root=repo_root,
        metadata_path=metadata_path,
        warmup=args.warmup,
        runs=args.runs,
        repeat=args.repeat,
        python_exec=args.python,
        include_python=include_python,
        include_python_numpy=include_python_numpy,
        java_opts=[],
    )

    if args.impls.strip():
        wanted = {x.strip() for x in args.impls.split(",") if x.strip()}
        impls = [i for i in impls if i.impl_id in wanted]

    if not impls:
        raise SystemExit("no implementation selected")

    java_selected = any(i.language == "java" for i in impls)
    if java_selected:
        try:
            java_resolution = resolve_java_opts(requested_java_opts, env, warnings)
        except Exception as exc:
            print(f"[FAIL] JVM options validation failed: {exc}", file=sys.stderr)
            return 1
        effective_java_opts = list(java_resolution.get("effective_opts", []))
        for impl in impls:
            if impl.language == "java":
                impl.command = [
                    "java",
                    *effective_java_opts,
                    "-cp",
                    str(repo_root / "java"),
                    "CosineBenchmark",
                    "--metadata",
                    str(metadata_path),
                    "--warmup",
                    str(args.warmup),
                    "--runs",
                    str(args.runs),
                    "--repeat",
                    str(args.repeat),
                ]
    elif requested_java_opts:
        warn("--java-opts provided but java implementation not selected", warnings)

    if not args.skip_build:
        impls, skipped_build = build_for_impls(repo_root, env, impls)
        for s in skipped_build:
            warn(f"{s['impl_id']}: {s['error']}", warnings)

    impls, skipped_runtime = filter_runnable(impls)
    for s in skipped_runtime:
        warn(f"{s['impl_id']}: {s['error']}", warnings)

    if not impls:
        raise SystemExit("no runnable implementation available after preflight checks")

    if args.dry_run:
        print("[DRY-RUN] preflight checks passed")
        print("[DRY-RUN] runnable implementations: " + ", ".join(i.impl_id for i in impls))
        if java_selected:
            print(
                "[DRY-RUN] java opts requested="
                + json.dumps(java_resolution.get("requested_opts", []))
                + " effective="
                + json.dumps(java_resolution.get("effective_opts", []))
            )
        return 0

    if args.warmup < 5 and args.runs <= 5:
        warn("warmup < 5 with very low runs may increase noise (notably Java/JIT)", warnings)

    perf_enabled = (not args.no_perf) and perf_available()

    n_meta = int(meta["N"])
    m_meta = int(meta["M"])
    d_meta = int(meta["D"])

    results_rows: List[Dict[str, object]] = []
    impl_summaries: Dict[str, Dict[str, object]] = {}

    for idx, impl in enumerate(impls):
        if args.gc_between:
            gc.collect()

        stability_report: Optional[Dict[str, object]] = None
        if args.stability_enable:
            gate_config = {
                "mode": args.stability_mode,
                "window_sec": args.stability_window_sec,
                "sample_interval_sec": args.stability_sample_interval_sec,
                "timeout_sec": args.stability_timeout_sec,
                "backoff_sec": args.stability_backoff_sec,
                "cpu_util_max": args.cpu_util_max,
                "load1_max_factor": args.load1_max_factor,
                "run_queue_max_factor": args.run_queue_max_factor,
                "disk_io_mbps_max": args.disk_io_mbps_max,
                "mem_available_min_mb": args.mem_available_min_mb,
                "mem_available_min_percent": args.mem_available_min_percent,
                "swap_activity_max": args.swap_activity_max,
                "cpu_util_variance_max": args.cpu_util_variance_max,
            }

            stability_report = wait_for_stable_system(gate_config)
            stability_reports[impl.impl_id] = stability_report

            if args.print_stability or not bool(stability_report.get("stable", False)):
                print(f"[STABILITY] {impl.impl_id} {stability_brief(stability_report)}")
                last = stability_report.get("last", {}) if isinstance(stability_report, dict) else {}
                reasons = last.get("blocking_reasons", []) if isinstance(last, dict) else []
                for reason in reasons:
                    print(f"[STABILITY] block: {reason}")

            last = stability_report.get("last", {}) if isinstance(stability_report, dict) else {}
            unavailable = last.get("unavailable_metrics", []) if isinstance(last, dict) else []
            if unavailable:
                warn(
                    f"stability metrics unavailable for {impl.impl_id}: {', '.join(str(x) for x in unavailable)}",
                    warnings,
                )

            if not bool(stability_report.get("stable", False)):
                reasons = last.get("blocking_reasons", []) if isinstance(last, dict) else []
                reason_txt = "; ".join(str(x) for x in reasons) if reasons else "unknown reason"
                timed_out = bool(stability_report.get("timed_out", False))
                if args.stability_mode == "fail":
                    critical_errors.append(
                        {
                            "impl_id": impl.impl_id,
                            "error": f"resource gating failed ({'timeout' if timed_out else 'unstable'}): {reason_txt}",
                            "stability": stability_report,
                        }
                    )
                    break
                warn(
                    f"{impl.impl_id}: skipped due to unstable resources ({'timeout' if timed_out else args.stability_mode}): {reason_txt}",
                    warnings,
                )
                if idx < len(impls) - 1 and args.cooldown_sec > 0:
                    time.sleep(args.cooldown_sec)
                continue

        print(f"[RUN] {impl.impl_id}")
        cmd = [*impl.command, "--expected-dataset-sha", dataset_sha]

        try:
            timed = run_with_metrics(
                command=cmd,
                cwd=str(repo_root),
                env=env,
                cpu_affinity=cpu_affinity,
                nice=args.nice,
                timeout_s=args.timeout_sec,
            )
        except Exception as exc:
            critical_errors.append({"impl_id": impl.impl_id, "error": f"execution error: {exc}"})
            if not args.keep_going:
                break
            if idx < len(impls) - 1 and args.cooldown_sec > 0:
                time.sleep(args.cooldown_sec)
            continue

        if timed.get("timed_out"):
            critical_errors.append({"impl_id": impl.impl_id, "error": "timeout"})
            if not args.keep_going:
                break
            if idx < len(impls) - 1 and args.cooldown_sec > 0:
                time.sleep(args.cooldown_sec)
            continue

        if int(timed.get("returncode", -1)) != 0:
            critical_errors.append(
                {
                    "impl_id": impl.impl_id,
                    "error": f"non-zero exit code: {timed.get('returncode')}",
                    "stdout_tail": tail_lines(str(timed.get("stdout", "")), 20),
                    "stderr_tail": tail_lines(str(timed.get("stderr", "")), 20),
                }
            )
            if not args.keep_going:
                break
            if idx < len(impls) - 1 and args.cooldown_sec > 0:
                time.sleep(args.cooldown_sec)
            continue

        try:
            meta_line, run_lines = parse_output_strict(str(timed.get("stdout", "")), impl.impl_id)
        except Exception as exc:
            critical_errors.append(
                {
                    "impl_id": impl.impl_id,
                    "error": f"strict output parse failed: {exc}",
                    "stdout_tail": tail_lines(str(timed.get("stdout", "")), 20),
                    "stderr_tail": tail_lines(str(timed.get("stderr", "")), 20),
                }
            )
            if not args.keep_going:
                break
            if idx < len(impls) - 1 and args.cooldown_sec > 0:
                time.sleep(args.cooldown_sec)
            continue

        try:
            require_keys(
                meta_line,
                ["type", "impl", "N", "M", "D", "repeat", "warmup", "runs", "dataset_sha256", "build_flags", "runtime"],
                f"meta[{impl.impl_id}]",
            )
            if i64(meta_line["N"], "meta.N") != n_meta or i64(meta_line["M"], "meta.M") != m_meta or i64(meta_line["D"], "meta.D") != d_meta:
                raise ValueError("dataset dimensions mismatch")
            if i64(meta_line["warmup"], "meta.warmup") != args.warmup or i64(meta_line["runs"], "meta.runs") != args.runs or i64(meta_line["repeat"], "meta.repeat") != args.repeat:
                raise ValueError("benchmark params mismatch")
            if str(meta_line["dataset_sha256"]) != dataset_sha:
                raise ValueError("dataset SHA mismatch in meta")

            if impl.language in ("c", "cpp"):
                flags = str(meta_line.get("build_flags", "")).replace(",", " ")
                if ("-O2" not in flags) and ("-O3" not in flags):
                    raise ValueError(f"build flags for {impl.impl_id} missing -O2/-O3: {flags}")
                if "-march=native" not in flags:
                    warn(f"{impl.impl_id}: -march=native absent in build flags ({flags})", warnings)

            if impl.language == "java" and args.warmup < 5 and args.runs <= 5:
                warn("java warmup/runs low; JIT/GC stability may be poor", warnings)

        except Exception as exc:
            critical_errors.append({"impl_id": impl.impl_id, "error": f"meta verification failed: {exc}"})
            if not args.keep_going:
                break
            if idx < len(impls) - 1 and args.cooldown_sec > 0:
                time.sleep(args.cooldown_sec)
            continue

        run_ids = set()
        wall_vals: List[float] = []
        cpu_vals: List[float] = []
        checksum_vals: List[float] = []

        st_metrics = {}
        if stability_report:
            st_metrics = (stability_report.get("last", {}) if isinstance(stability_report, dict) else {}).get("metrics", {}) or {}

        for run_obj in run_lines:
            try:
                require_keys(
                    run_obj,
                    [
                        "type",
                        "impl",
                        "run_id",
                        "wall_ns",
                        "cpu_ns",
                        "checksum",
                        "max_rss_kb",
                        "ctx_voluntary",
                        "ctx_involuntary",
                        "minor_faults",
                        "major_faults",
                    ],
                    f"run[{impl.impl_id}]",
                )

                run_id = i64(run_obj["run_id"], "run_id")
                wall_ns = i64(run_obj["wall_ns"], "wall_ns")
                cpu_ns = i64(run_obj["cpu_ns"], "cpu_ns")
                checksum = f64(run_obj["checksum"], "checksum")
                max_rss_kb = i64(run_obj["max_rss_kb"], "max_rss_kb")
                ctx_vol = i64(run_obj["ctx_voluntary"], "ctx_voluntary")
                ctx_invol = i64(run_obj["ctx_involuntary"], "ctx_involuntary")
                minor_faults = i64(run_obj["minor_faults"], "minor_faults")
                major_faults = i64(run_obj["major_faults"], "major_faults")
                max_threads_run = run_obj.get("max_threads")
                max_threads_run_int = i64(max_threads_run, "max_threads") if max_threads_run is not None else None
            except Exception as exc:
                critical_errors.append({"impl_id": impl.impl_id, "error": f"run verification failed: {exc}"})
                if not args.keep_going:
                    break
                continue

            run_ids.add(run_id)
            wall_vals.append(float(wall_ns))
            cpu_vals.append(float(cpu_ns))
            checksum_vals.append(checksum)

            row = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "impl_id": impl.impl_id,
                "run_id": run_id,
                "wall_ns": wall_ns,
                "cpu_ns": cpu_ns,
                "checksum": checksum,
                "dataset_sha256": dataset_sha,
                "max_rss_kb": max_rss_kb,
                "ctx_voluntary": ctx_vol,
                "ctx_involuntary": ctx_invol,
                "minor_faults": minor_faults,
                "major_faults": major_faults,
                "cpu_migrations": timed.get("cpu_migrations"),
                "max_threads": max_threads_run_int if max_threads_run_int is not None else timed.get("max_threads"),
                "perf_cycles": None,
                "perf_instructions": None,
                "perf_cache_misses": None,
                "perf_branch_misses": None,
                "warmup": args.warmup,
                "runs": args.runs,
                "repeat": args.repeat,
                "language": impl.language,
                "requested_version": impl.requested_version,
                "affinity_observed": timed.get("affinity_observed"),
                "runner_wall_ns": timed.get("runner_wall_ns"),
                "runner_cpu_ns": timed.get("runner_cpu_ns"),
                "process_returncode": timed.get("returncode"),
                "stability_cpu_util_avg": st_metrics.get("cpu_util_avg"),
                "stability_load1_avg": st_metrics.get("load1_avg"),
                "stability_io_mbps_avg": st_metrics.get("disk_io_mbps_avg"),
                "stability_mem_available_mb": st_metrics.get("mem_available_mb_min"),
                "stability_swap_in": st_metrics.get("swap_in"),
                "stability_swap_out": st_metrics.get("swap_out"),
                "stability_swap_activity": st_metrics.get("swap_activity"),
                "stability_wait_sec": stability_report.get("waited_sec") if stability_report else None,
                "stability_attempts": stability_report.get("attempts") if stability_report else None,
                "stability_pass": bool(stability_report.get("stable")) if stability_report else None,
            }
            results_rows.append(row)

        if len(run_ids) != args.runs:
            critical_errors.append(
                {
                    "impl_id": impl.impl_id,
                    "error": f"run count mismatch: expected {args.runs}, got {len(run_ids)}",
                }
            )
            if not args.keep_going:
                break
            if idx < len(impls) - 1 and args.cooldown_sec > 0:
                time.sleep(args.cooldown_sec)
            continue

        if args.enforce_single_thread and timed.get("max_threads") and int(timed["max_threads"]) > 1:
            # Go and Java runtimes can keep helper threads alive (GC/JIT/runtime) even when
            # compute parallelism is constrained; keep this visible but non-fatal.
            if impl.language in ("go", "java"):
                warn(
                    f"{impl.impl_id}: runtime helper threads observed under single-thread mode (max_threads={timed.get('max_threads')})",
                    warnings,
                )
            else:
                critical_errors.append(
                    {
                        "impl_id": impl.impl_id,
                        "error": f"single-thread enforcement violated: observed max_threads={timed.get('max_threads')}",
                    }
                )
                if not args.keep_going:
                    warn(
                        f"{impl.impl_id}: single-thread enforcement violated; continuing remaining impls and failing at end",
                        warnings,
                    )
        elif timed.get("max_threads") and int(timed["max_threads"]) > 1:
            warn(f"{impl.impl_id}: observed max_threads={timed.get('max_threads')}", warnings)

        if cpu_affinity and timed.get("affinity_observed") and str(timed.get("affinity_observed")) != str(cpu_affinity):
            warn(
                f"{impl.impl_id}: requested CPU affinity '{cpu_affinity}' but observed '{timed.get('affinity_observed')}'",
                warnings,
            )

        perf_metrics_all: Dict[str, Optional[float]] = {}
        if perf_enabled:
            perf = run_perf_stat(
                command=cmd,
                cwd=str(repo_root),
                env=env,
                cpu_affinity=cpu_affinity,
                nice=args.nice,
                timeout_s=args.timeout_sec,
            )
            if int(perf.get("returncode", -1)) == 0:
                perf_metrics_all = dict(perf.get("metrics", {}))
                for row in results_rows:
                    if row["impl_id"] == impl.impl_id:
                        row["perf_cycles"] = perf_metrics_all.get("cycles")
                        row["perf_instructions"] = perf_metrics_all.get("instructions")
                        row["perf_cache_misses"] = perf_metrics_all.get("cache-misses")
                        row["perf_branch_misses"] = perf_metrics_all.get("branch-misses")
                        row["cpu_migrations"] = (
                            row["cpu_migrations"] if row["cpu_migrations"] is not None else perf_metrics_all.get("cpu-migrations")
                        )
            else:
                warn(f"perf stat failed for {impl.impl_id}", warnings)

        impl_summaries[impl.impl_id] = {
            "impl_id": impl.impl_id,
            "language": impl.language,
            "requested_version": impl.requested_version,
            "meta": meta_line,
            "runs": len(run_ids),
            "median_wall_ns": median(wall_vals),
            "median_cpu_ns": median(cpu_vals),
            "median_checksum": median(checksum_vals),
            "checksum_min": min(checksum_vals),
            "checksum_max": max(checksum_vals),
            "runner_wall_ns": timed.get("runner_wall_ns"),
            "runner_cpu_ns": timed.get("runner_cpu_ns"),
            "preflight_stability": stability_report,
            "process_metrics": {
                "max_rss_kb": timed.get("max_rss_kb"),
                "ctx_voluntary": timed.get("ctx_switches_voluntary"),
                "ctx_involuntary": timed.get("ctx_switches_involuntary"),
                "minor_faults": timed.get("page_faults_minor"),
                "major_faults": timed.get("page_faults_major"),
                "cpu_migrations": timed.get("cpu_migrations"),
                "max_threads": timed.get("max_threads"),
                "affinity_observed": timed.get("affinity_observed"),
            },
            "perf": perf_metrics_all,
        }

        if idx < len(impls) - 1 and args.cooldown_sec > 0:
            time.sleep(args.cooldown_sec)

    if not results_rows:
        critical_errors.append({"impl_id": "runner", "error": "no successful benchmark result"})

    if impl_summaries:
        baseline_id = args.baseline if args.baseline in impl_summaries else sorted(impl_summaries.keys())[0]
        baseline_wall = float(impl_summaries[baseline_id]["median_wall_ns"])
        baseline_checksum = float(impl_summaries[baseline_id]["median_checksum"])

        for impl_id, summary in impl_summaries.items():
            wall = float(summary["median_wall_ns"])
            summary["speedup_vs_baseline_wall"] = baseline_wall / wall if wall > 0 else None
            summary["baseline"] = impl_id == baseline_id

            chk = float(summary["median_checksum"])
            if not math.isclose(chk, baseline_checksum, rel_tol=REL_TOL, abs_tol=ABS_TOL):
                critical_errors.append(
                    {
                        "impl_id": impl_id,
                        "error": (
                            "checksum mismatch vs baseline "
                            f"({baseline_id}={baseline_checksum:.17g}, {impl_id}={chk:.17g}, "
                            f"rtol={REL_TOL}, atol={ABS_TOL})"
                        ),
                    }
                )
    else:
        baseline_id = None

    csv_path = Path(args.results_csv).resolve()
    summary_path = Path(args.summary_json).resolve()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "timestamp",
        "impl_id",
        "run_id",
        "wall_ns",
        "cpu_ns",
        "checksum",
        "dataset_sha256",
        "max_rss_kb",
        "ctx_voluntary",
        "ctx_involuntary",
        "minor_faults",
        "major_faults",
        "cpu_migrations",
        "max_threads",
        "perf_cycles",
        "perf_instructions",
        "perf_cache_misses",
        "perf_branch_misses",
        "stability_cpu_util_avg",
        "stability_load1_avg",
        "stability_io_mbps_avg",
        "stability_mem_available_mb",
        "stability_swap_in",
        "stability_swap_out",
        "stability_swap_activity",
        "stability_wait_sec",
        "stability_attempts",
        "stability_pass",
        "warmup",
        "runs",
        "repeat",
        "language",
        "requested_version",
        "affinity_observed",
        "runner_wall_ns",
        "runner_cpu_ns",
        "process_returncode",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results_rows:
            writer.writerow(row)

    tool_versions = collect_tool_versions()

    verifications = {
        "dataset_lock": {"status": "pass", **lock_info},
        "dataset_sha": {"status": "pass", "sha256": dataset_sha, "E_sha256": e_sha, "A_sha256": a_sha},
        "parameter_match": {"status": "fail" if any("meta verification failed" in e["error"] for e in critical_errors) else "pass"},
        "output_format": {"status": "fail" if any("strict output parse failed" in e["error"] for e in critical_errors) else "pass"},
        "checksum_consistency": {
            "status": "fail" if any("checksum mismatch" in e["error"] for e in critical_errors) else "pass",
            "abs_tol": ABS_TOL,
            "rel_tol": REL_TOL,
        },
        "single_thread": {
            "status": "fail" if any("single-thread enforcement violated" in e["error"] for e in critical_errors) else "pass",
            "enforced": args.enforce_single_thread,
        },
        "compilation_flags": {
            "status": "fail" if any("build flags" in e["error"] for e in critical_errors) else "pass",
        },
        "resource_gating": {
            "status": "fail" if any("resource gating failed" in e["error"] for e in critical_errors) else "pass",
            "enabled": bool(args.stability_enable),
            "mode": args.stability_mode,
        },
    }

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "metadata_path": str(metadata_path),
        "dataset": {
            "N": n_meta,
            "M": m_meta,
            "D": d_meta,
            "format": meta["format"],
            "dtype": meta["dtype"],
            "dataset_sha256": dataset_sha,
            "E_sha256": e_sha,
            "A_sha256": a_sha,
            "lock": lock_info,
        },
        "params": {
            "warmup": args.warmup,
            "runs": args.runs,
            "repeat": args.repeat,
            "cpu_affinity": cpu_affinity,
            "nice": args.nice,
            "disable_python": bool(args.disable_python),
            "enforce_single_thread": args.enforce_single_thread,
            "java_opts_requested": java_resolution.get("requested_opts", []),
            "java_opts_effective": java_resolution.get("effective_opts", []),
            "perf_enabled": perf_enabled,
            "cooldown_sec": args.cooldown_sec,
            "gc_between": bool(args.gc_between),
        },
        "java_control": {
            "requested_opts": java_resolution.get("requested_opts", []),
            "effective_opts": java_resolution.get("effective_opts", []),
            "profile_used": java_resolution.get("profile_used"),
            "fallback_reason": java_resolution.get("fallback_reason"),
            "blocked_options": java_resolution.get("blocked_options", []),
            "attempts": java_resolution.get("attempts", []),
            "jvm_runtime": java_resolution.get("jvm_runtime"),
        },
        "stability": {
            "enabled": bool(args.stability_enable),
            "mode": args.stability_mode,
            "window_sec": args.stability_window_sec,
            "sample_interval_sec": args.stability_sample_interval_sec,
            "timeout_sec": args.stability_timeout_sec,
            "backoff_sec": args.stability_backoff_sec,
            "thresholds": {
                "cpu_util_max": args.cpu_util_max,
                "load1_max_factor": args.load1_max_factor,
                "run_queue_max_factor": args.run_queue_max_factor,
                "disk_io_mbps_max": args.disk_io_mbps_max,
                "mem_available_min_mb": args.mem_available_min_mb,
                "mem_available_min_percent": args.mem_available_min_percent,
                "swap_activity_max": args.swap_activity_max,
                "cpu_util_variance_max": args.cpu_util_variance_max,
            },
            "reports_by_impl": stability_reports,
        },
        "thread_environment": {
            "before": thread_env_before,
            "effective": thread_env_effective,
        },
        "versions": tool_versions,
        "baseline_impl_id": baseline_id,
        "implementations": [impl_summaries[k] for k in sorted(impl_summaries.keys())],
        "warnings": warnings,
        "critical_errors": critical_errors,
        "verifications": verifications,
        "results_csv": str(csv_path),
    }

    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("\n=== Summary (median wall, speedup vs baseline) ===")
    if impl_summaries:
        print(f"Baseline: {baseline_id}")
        print(f"{'impl_id':16} {'wall_ms':>12} {'cpu_ms':>12} {'speedup':>10} {'checksum':>20}")
        for s in sorted(impl_summaries.values(), key=lambda x: float(x["median_wall_ns"])):
            wall_ms = float(s["median_wall_ns"]) / 1e6
            cpu_ms = float(s["median_cpu_ns"]) / 1e6
            speed = s.get("speedup_vs_baseline_wall")
            speed_txt = f"x{speed:.2f}" if speed is not None else "n/a"
            print(f"{s['impl_id']:16} {wall_ms:12.3f} {cpu_ms:12.3f} {speed_txt:>10} {float(s['median_checksum']):20.12g}")

    print(f"\nWrote CSV: {csv_path}")
    print(f"Wrote summary: {summary_path}")

    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print(f"- {w}")

    if critical_errors:
        print("\nCritical errors:")
        for e in critical_errors:
            print(f"- {e.get('impl_id')}: {e.get('error')}")
            if e.get("stdout_tail"):
                print("  stdout tail:")
                print(e["stdout_tail"])
            if e.get("stderr_tail"):
                print("  stderr tail:")
                print(e["stderr_tail"])
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
