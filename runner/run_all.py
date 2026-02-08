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
import tomllib
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

WARNINGS_ENABLED = True

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
JAVA_STABLE_REQUIRED_OPTS = {
    "-Xms2g",
    "-Xmx2g",
    "-XX:ActiveProcessorCount=1",
    "-XX:+UseSerialGC",
}
RUNNER_ENV_KEYS = [*THREAD_ENV_VARS, "GOMAXPROCS", "RAYON_NUM_THREADS"]

KNOWN_SINGLE_THREAD_IMPLS = {
    "c-naive",
    "c-simd-portable",
    "c-simd-native",
    "cpp-naive",
    "cpp-simd-portable",
    "cpp-simd-native",
    "rust-naive",
    "rust-simd",
}


@dataclass
class Implementation:
    impl_id: str
    language: str
    requested_version: str
    variant: str
    command: List[str]


def warn(msg: str, warnings: List[str]) -> None:
    if not WARNINGS_ENABLED:
        return
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
        "javac": cmd_line(["javac", "-version"], stderr=True),
        "java": cmd_line(["java", "-version"], stderr=True),
        "go": cmd_line(["go", "version"]),
        "cargo": cmd_line(["cargo", "--version"]),
        "rustc": cmd_line(["rustc", "--version"]),
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


def shell_join(cmd: Sequence[str]) -> str:
    return " ".join(shlex.quote(x) for x in cmd)


def injected_env_view(env_before: Dict[str, str], env_after: Dict[str, str]) -> Dict[str, str]:
    view: Dict[str, str] = {}
    for key in RUNNER_ENV_KEYS:
        before = env_before.get(key)
        after = env_after.get(key)
        if after is not None and after != before:
            view[key] = after
    return view


def java_opts_are_stable(opts: Sequence[str]) -> bool:
    return JAVA_STABLE_REQUIRED_OPTS.issubset(set(opts))


def parse_make_vars(text: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def get_make_build_info(repo_root: Path, env: Dict[str, str], lang: str, profile: str) -> Dict[str, str]:
    if lang == "c":
        cmd = ["make", "-s", "-C", str(repo_root / "c"), "print-flags", f"PROFILE={profile}"]
    elif lang == "cpp":
        cmd = ["make", "-s", "-C", str(repo_root / "cpp"), "print-flags", f"PROFILE={profile}"]
    else:
        return {}
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"failed to read {lang} make flags: {tail_lines(proc.stderr, 20)}")
    return parse_make_vars(proc.stdout)


def read_rust_release_profile(cargo_toml_path: Path) -> Dict[str, object]:
    result = {
        "available": False,
        "opt_level": None,
        "lto": None,
        "codegen_units": None,
        "panic": None,
        "ok": False,
        "issues": [],
    }
    raw = tomllib.loads(cargo_toml_path.read_text(encoding="utf-8"))
    prof = (((raw.get("profile") or {}).get("release")) or {})
    if not prof:
        result["issues"] = ["[profile.release] section missing"]
        return result

    result["available"] = True
    result["opt_level"] = prof.get("opt-level")
    result["lto"] = prof.get("lto")
    result["codegen_units"] = prof.get("codegen-units")
    result["panic"] = prof.get("panic")

    issues: List[str] = []
    if int(prof.get("opt-level", -1)) != 3:
        issues.append("opt-level must be 3")
    if not bool(prof.get("lto")):
        issues.append("lto must be enabled")
    if int(prof.get("codegen-units", -1)) != 1:
        issues.append("codegen-units must be 1")
    if str(prof.get("panic", "")) != "abort":
        issues.append("panic must be 'abort'")
    result["issues"] = issues
    result["ok"] = len(issues) == 0
    return result


def parse_self_check(stdout: str, impl_id: str) -> Dict[str, object]:
    lines = [ln.strip() for ln in stdout.splitlines() if ln.strip()]
    if not lines:
        raise ValueError("empty self-check stdout")
    obj = json.loads(lines[0])
    if not isinstance(obj, dict):
        raise ValueError("self-check output is not JSON object")
    if obj.get("type") != "self_check":
        raise ValueError("self-check output type mismatch")
    if str(obj.get("impl")) != impl_id:
        raise ValueError(f"self-check impl mismatch: expected {impl_id}, got {obj.get('impl')}")
    if not bool(obj.get("ok", False)):
        raise ValueError("self-check reported ok=false")
    return obj


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
            if java_opts_are_stable(requested):
                effective = requested
                profile_used = "requested"
            else:
                fallback_reason = "requested JVM options are valid but non-stable for benchmarking"
                warn(
                    "Java JVM opts accepted by JVM but non-stable for benchmark; falling back to strict_single_core",
                    warnings,
                )
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


def resolve_java_opts_multicore(
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
    profile_used = "default"
    effective: List[str] = []

    if blocked:
        raise RuntimeError(f"blocked JVM option(s) in multi-core mode: {', '.join(blocked)}")

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
            fallback_reason = f"requested JVM options failed (exit={probe.get('returncode')}), using JVM defaults"
            warn("Java JVM opts rejected in multi-core mode, falling back to JVM defaults", warnings)

    return {
        "requested_opts": list(requested_opts),
        "effective_opts": effective,
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
    include_python_vectorized: bool,
    build_profile: str,
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

    c_simd_impl_id = f"c-simd-{build_profile}"
    cpp_simd_impl_id = f"cpp-simd-{build_profile}"

    impls: List[Implementation] = [
        Implementation(
            impl_id="c-naive",
            language="c",
            requested_version="naive",
            variant="naive",
            command=[str(repo_root / "c" / "benchmark_c"), *common],
        ),
        Implementation(
            impl_id=c_simd_impl_id,
            language="c",
            requested_version="simd",
            variant="simd",
            command=[str(repo_root / "c" / "benchmark_c_simd"), *common],
        ),
        Implementation(
            impl_id="cpp-naive",
            language="cpp",
            requested_version="naive",
            variant="naive",
            command=[str(repo_root / "cpp" / "benchmark_cpp"), *common],
        ),
        Implementation(
            impl_id=cpp_simd_impl_id,
            language="cpp",
            requested_version="simd",
            variant="simd",
            command=[str(repo_root / "cpp" / "benchmark_cpp_simd"), *common],
        ),
        Implementation(
            impl_id="rust-naive",
            language="rust",
            requested_version="naive",
            variant="naive",
            command=[str(repo_root / "rust" / "target" / "release" / "cosine_benchmark_rust"), *common],
        ),
        Implementation(
            impl_id="rust-simd",
            language="rust",
            requested_version="simd",
            variant="simd",
            command=[str(repo_root / "rust" / "target" / "release" / "rust_simd"), *common],
        ),
        Implementation(
            impl_id="rust-par",
            language="rust",
            requested_version="par",
            variant="par",
            command=[str(repo_root / "rust" / "target" / "release" / "rust_par"), *common],
        ),
        Implementation(
            impl_id="go-naive",
            language="go",
            requested_version="naive",
            variant="naive",
            command=[str(repo_root / "go" / "benchmark_go"), *common],
        ),
        Implementation(
            impl_id="go-opt",
            language="go",
            requested_version="opt",
            variant="opt",
            command=[str(repo_root / "go" / "benchmark_go_opt"), *common],
        ),
        Implementation(
            impl_id="java-naive",
            language="java",
            requested_version="naive",
            variant="naive",
            command=["java", "-cp", str(repo_root / "java"), "CosineBenchmark", *common],
        ),
        Implementation(
            impl_id="java-opt",
            language="java",
            requested_version="opt",
            variant="opt",
            command=["java", "-cp", str(repo_root / "java"), "CosineBenchmarkOpt", *common],
        ),
    ]

    if include_python:
        impls.insert(
            0,
            Implementation(
                impl_id="python-naive",
                language="python",
                requested_version="naive",
                variant="naive",
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
                variant="blas",
                command=[python_exec, str(repo_root / "python" / "benchmark_python.py"), *common, "--version", "numpy"],
            ),
        )

    if include_python and include_python_vectorized:
        impls.insert(
            2,
            Implementation(
                impl_id="python-vectorized",
                language="python",
                requested_version="vectorized",
                variant="simd",
                command=[python_exec, str(repo_root / "python" / "benchmark_python_vectorized.py"), *common],
            ),
        )

    return impls


def build_for_impls(
    repo_root: Path,
    env: Dict[str, str],
    impls: List[Implementation],
    build_profile: str,
) -> Tuple[List[Implementation], List[Dict[str, str]], Dict[str, Dict[str, object]]]:
    build_steps = {
        "c": (
            "make",
            [["make", "-B", "-C", str(repo_root / "c"), f"PROFILE={build_profile}"]],
        ),
        "cpp": (
            "make",
            [["make", "-B", "-C", str(repo_root / "cpp"), f"PROFILE={build_profile}"]],
        ),
        "rust": (
            "cargo",
            [["cargo", "build", "--release", "--manifest-path", str(repo_root / "rust" / "Cargo.toml")]],
        ),
        "go": (
            "go",
            [
                ["go", "build", "-o", str(repo_root / "go" / "benchmark_go"), str(repo_root / "go" / "main.go")],
                ["go", "build", "-o", str(repo_root / "go" / "benchmark_go_opt"), str(repo_root / "go" / "main_opt.go")],
            ],
        ),
        "java": (
            "javac",
            [
                [
                    "javac",
                    str(repo_root / "java" / "CosineBenchmark.java"),
                    str(repo_root / "java" / "CosineBenchmarkOpt.java"),
                ]
            ],
        ),
    }

    langs = sorted({i.language for i in impls if i.language != "python"})
    skipped: List[Dict[str, str]] = []
    skipped_langs = set()
    build_info: Dict[str, Dict[str, object]] = {}

    for lang in langs:
        if lang not in build_steps:
            continue
        tool, cmd_list = build_steps[lang]
        if shutil.which(tool) is None:
            skipped_langs.add(lang)
            skipped.append({"impl_id": f"{lang}-*", "error": f"skipped: required build tool not found ({tool})"})
            continue
        for cmd in cmd_list:
            proc = subprocess.run(cmd, cwd=str(repo_root), env=env)
            if proc.returncode != 0:
                raise RuntimeError(f"build failed for {lang}: {' '.join(cmd)}")
        info: Dict[str, object] = {"tool": tool, "build_commands": cmd_list}
        if lang in ("c", "cpp"):
            info.update(get_make_build_info(repo_root, env, lang, build_profile))
            info["build_profile"] = build_profile
        build_info[lang] = info

    kept = [i for i in impls if i.language not in skipped_langs]
    return kept, skipped, build_info


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


def affinity_cpu_count(spec: Optional[str]) -> Optional[int]:
    if spec is None:
        return None
    cpus = set()
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            parts = chunk.split("-", 1)
            if len(parts) != 2:
                return None
            try:
                start = int(parts[0])
                end = int(parts[1])
            except ValueError:
                return None
            if end < start:
                return None
            for cpu in range(start, end + 1):
                cpus.add(cpu)
        else:
            try:
                cpus.add(int(chunk))
            except ValueError:
                return None
    return len(cpus)


def mode_banner(run_mode: str, multicore_scope: str = "all") -> str:
    if run_mode == "single-core":
        return "MODE: single-core (comparative baseline)"
    if run_mode == "multi-core":
        if multicore_scope == "scalable-only":
            return "MODE: multi-core scalable-only (throughput), excluding single-thread impls"
        return "MODE: multi-core (throughput / scalability)"
    return "MODE: custom"


def replace_or_append_cli_arg(cmd: Sequence[str], flag: str, value: str) -> List[str]:
    out = list(cmd)
    if flag in out:
        idx = out.index(flag)
        if idx + 1 < len(out):
            out[idx + 1] = value
        else:
            out.append(value)
        return out
    out.extend([flag, value])
    return out


def make_probe_command(cmd: Sequence[str], dataset_sha: str) -> List[str]:
    out = list(cmd)
    out = replace_or_append_cli_arg(out, "--warmup", "0")
    out = replace_or_append_cli_arg(out, "--runs", "1")
    out = replace_or_append_cli_arg(out, "--repeat", "1")
    out = replace_or_append_cli_arg(out, "--expected-dataset-sha", dataset_sha)
    return out


def probe_impl_threads(
    repo_root: Path,
    env: Dict[str, str],
    impl: Implementation,
    dataset_sha: str,
    cpu_affinity: Optional[str],
    nice: Optional[int],
    timeout_s: Optional[int],
) -> Dict[str, object]:
    cmd = make_probe_command(impl.command, dataset_sha)
    try:
        timed = run_with_metrics(
            command=cmd,
            cwd=str(repo_root),
            env=env,
            cpu_affinity=cpu_affinity,
            nice=nice,
            timeout_s=timeout_s,
        )
    except Exception as exc:
        return {
            "ok": False,
            "impl_id": impl.impl_id,
            "reason": f"probe execution error: {exc}",
            "max_threads": None,
        }

    if bool(timed.get("timed_out", False)):
        return {
            "ok": False,
            "impl_id": impl.impl_id,
            "reason": "probe timeout",
            "max_threads": timed.get("max_threads"),
        }

    if int(timed.get("returncode", -1)) != 0:
        return {
            "ok": False,
            "impl_id": impl.impl_id,
            "reason": f"probe non-zero exit ({timed.get('returncode')})",
            "stdout_tail": tail_lines(str(timed.get("stdout", "")), 20),
            "stderr_tail": tail_lines(str(timed.get("stderr", "")), 20),
            "max_threads": timed.get("max_threads"),
        }

    try:
        _, run_lines = parse_output_strict(str(timed.get("stdout", "")), impl.impl_id)
    except Exception as exc:
        return {
            "ok": False,
            "impl_id": impl.impl_id,
            "reason": f"probe output parse failed: {exc}",
            "stdout_tail": tail_lines(str(timed.get("stdout", "")), 20),
            "stderr_tail": tail_lines(str(timed.get("stderr", "")), 20),
            "max_threads": timed.get("max_threads"),
        }

    observed = []
    if timed.get("max_threads") is not None:
        try:
            observed.append(int(timed.get("max_threads")))
        except Exception:
            pass

    for run_obj in run_lines:
        v = run_obj.get("max_threads")
        if v is None:
            continue
        try:
            observed.append(int(v))
        except Exception:
            continue

    max_threads = max(observed) if observed else None
    scalable = bool(max_threads is not None and max_threads >= 2)
    return {
        "ok": True,
        "impl_id": impl.impl_id,
        "max_threads": max_threads,
        "scalable": scalable,
        "command": shell_join(cmd),
    }


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(description="Run all cosine benchmark implementations with strict verifications")
    parser.add_argument("--metadata", default=str(repo_root / "data" / "metadata.json"))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--run-mode", choices=["single-core", "multi-core"], default=None)
    parser.add_argument("--threads", type=int, default=None, help="Target runtime thread count for --run-mode multi-core")
    parser.add_argument("--multicore-scope", choices=["all", "scalable-only"], default="all")

    parser.add_argument("--enforce-single-thread", action="store_true")
    parser.add_argument("--cpu-affinity", default=None, help="CPU affinity list, e.g. '2' or '2,3'")
    parser.add_argument("--pin-affinity", default=None, help="Alias of --cpu-affinity")
    parser.add_argument("--cpu-set", default=None, help="Alias of --cpu-affinity, e.g. '0-7'")
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
    parser.add_argument("--profile", choices=["portable", "native"], default="portable", help="Compilation profile for C/C++")
    parser.add_argument("--no-perf", action="store_true")
    parser.add_argument("--quiet-warnings", action="store_true", help="Suppress warning output and warning list")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--timeout-sec", type=int, default=None)
    parser.add_argument("--baseline", default="python-naive")
    parser.add_argument("--impls", default="")
    parser.add_argument("--variants", default="", help="Comma-separated variants filter: naive,simd,par,blas,opt")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--disable-python", action="store_true", help="Disable python implementations")
    parser.add_argument("--no-python-numpy", action="store_true")
    parser.add_argument("--java-opts", default="")
    parser.add_argument("--dry-run", action="store_true", help="Validate preflight/build/runtime and exit without running benchmarks")

    parser.add_argument("--dataset-lock", default=str(repo_root / "results" / "dataset.lock"))
    parser.add_argument("--results-csv", default=str(repo_root / "results" / "results.csv"))
    parser.add_argument("--summary-json", default=str(repo_root / "results" / "summary.json"))

    args = parser.parse_args()

    global WARNINGS_ENABLED
    WARNINGS_ENABLED = not args.quiet_warnings

    cpu_affinity = args.cpu_affinity or args.pin_affinity or args.cpu_set
    run_mode = args.run_mode if args.run_mode else "custom"
    multicore_scope = args.multicore_scope

    if args.warmup < 0 or args.runs <= 0 or args.repeat <= 0:
        raise SystemExit("warmup must be >= 0, runs and repeat must be > 0")

    affinity_count = affinity_cpu_count(cpu_affinity)
    if cpu_affinity is not None and affinity_count is None:
        print(f"[WARN] unable to parse CPU affinity spec for validation: '{cpu_affinity}'", file=sys.stderr)

    if run_mode == "single-core":
        if not args.enforce_single_thread:
            raise SystemExit("[FAIL] run-mode single-core requires --enforce-single-thread")
        if cpu_affinity is None:
            raise SystemExit("[FAIL] run-mode single-core requires --cpu-affinity with a single core")
        if affinity_count is not None and affinity_count != 1:
            raise SystemExit(
                f"[FAIL] run-mode single-core requires mono-core affinity; got '{cpu_affinity}' ({affinity_count} cores)"
            )
        if args.profile != "portable":
            raise SystemExit("[FAIL] run-mode single-core requires --profile portable")

    if run_mode == "multi-core":
        if args.enforce_single_thread:
            raise SystemExit("[FAIL] run-mode multi-core forbids --enforce-single-thread")
        if cpu_affinity is not None and affinity_count == 1:
            raise SystemExit(
                f"[FAIL] run-mode multi-core forbids mono-core affinity; got '{cpu_affinity}'"
            )
        if args.threads is None:
            raise SystemExit("[FAIL] run-mode multi-core requires --threads T")
        if args.threads < 2:
            raise SystemExit("[FAIL] run-mode multi-core requires --threads >= 2")
        if args.profile != "native":
            raise SystemExit("[FAIL] run-mode multi-core requires --profile native")

    if run_mode != "multi-core" and args.threads is not None:
        print("[WARN] --threads is only used with --run-mode multi-core", file=sys.stderr)
    if run_mode != "multi-core" and multicore_scope != "all":
        print("[WARN] --multicore-scope is only used with --run-mode multi-core", file=sys.stderr)

    warnings: List[str] = []
    critical_errors: List[Dict[str, object]] = []
    stability_reports: Dict[str, Dict[str, object]] = {}
    build_info_by_lang: Dict[str, Dict[str, object]] = {}
    multicore_scope_exclusions: List[Dict[str, object]] = []
    multicore_scope_probes: Dict[str, Dict[str, object]] = {}
    expected_scalable_impls: set[str] = set()

    print(mode_banner(run_mode, multicore_scope))

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
    thread_env_before = {k: env.get(k) for k in RUNNER_ENV_KEYS}

    configured_threads: Optional[int] = None

    if run_mode == "multi-core":
        configured_threads = int(args.threads)
        thread_value = str(configured_threads)
        for k in THREAD_ENV_VARS:
            env[k] = thread_value
        env["GOMAXPROCS"] = thread_value
        env["RAYON_NUM_THREADS"] = thread_value
    elif args.enforce_single_thread:
        for k in THREAD_ENV_VARS:
            env[k] = "1"
        # Go runtime may otherwise schedule work on multiple OS threads.
        env["GOMAXPROCS"] = "1"
        env["RAYON_NUM_THREADS"] = "1"
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
        rayon_threads = env.get("RAYON_NUM_THREADS")
        if rayon_threads is not None and rayon_threads != "1":
            warn(f"RAYON_NUM_THREADS={rayon_threads} (multi-thread runtime may bias comparison)", warnings)

    thread_env_effective = {k: env.get(k) for k in RUNNER_ENV_KEYS}
    env_injected = injected_env_view(os.environ, env)

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
    has_np = False
    np_ver: Optional[str] = None
    if include_python:
        has_np, np_ver = python_numpy_available(args.python)
        if has_np and np_ver:
            print(f"[INFO] numpy detected: {np_ver}")

    include_python_numpy = include_python and (not args.no_python_numpy) and has_np
    include_python_vectorized = include_python and has_np

    if include_python and not has_np:
        if not args.no_python_numpy:
            warn("numpy not available for selected python; python-numpy and python-vectorized skipped", warnings)
        else:
            warn("numpy not available for selected python; python-vectorized skipped", warnings)
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
        include_python_vectorized=include_python_vectorized,
        build_profile=args.profile,
    )
    impl_catalog = list(impls)

    if args.variants.strip():
        wanted_variants = {x.strip() for x in args.variants.split(",") if x.strip()}
        known_variants = {"naive", "simd", "par", "blas", "opt"}
        unknown_variants = sorted(v for v in wanted_variants if v not in known_variants)
        if unknown_variants:
            raise SystemExit("unknown --variants entries: " + ", ".join(unknown_variants))
        impls = [i for i in impls if i.variant in wanted_variants]

    if args.impls.strip():
        wanted = {x.strip() for x in args.impls.split(",") if x.strip()}
        impls = [i for i in impls if i.impl_id in wanted]

    baseline_impl = next((i for i in impl_catalog if i.impl_id == args.baseline), None)
    baseline_forced_added = False
    if baseline_impl is not None and all(i.impl_id != args.baseline for i in impls):
        impls.append(baseline_impl)
        baseline_forced_added = True

    if not impls:
        raise SystemExit("no implementation selected")

    java_selected = any(i.language == "java" for i in impls)
    if java_selected:
        try:
            if run_mode == "multi-core":
                java_resolution = resolve_java_opts_multicore(requested_java_opts, env, warnings)
            else:
                java_resolution = resolve_java_opts(requested_java_opts, env, warnings)
        except Exception as exc:
            print(f"[FAIL] JVM options validation failed: {exc}", file=sys.stderr)
            return 1
        effective_java_opts = list(java_resolution.get("effective_opts", []))
        for impl in impls:
            if impl.language == "java":
                class_name = impl.command[3] if len(impl.command) >= 4 else "CosineBenchmark"
                impl.command = [
                    "java",
                    *effective_java_opts,
                    "-cp",
                    str(repo_root / "java"),
                    class_name,
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
        impls, skipped_build, build_info_by_lang = build_for_impls(repo_root, env, impls, args.profile)
        for s in skipped_build:
            warn(f"{s['impl_id']}: {s['error']}", warnings)

    impls, skipped_runtime = filter_runnable(impls)
    for s in skipped_runtime:
        warn(f"{s['impl_id']}: {s['error']}", warnings)

    if run_mode == "multi-core" and multicore_scope == "scalable-only":
        filtered_impls: List[Implementation] = []
        scalable_count = 0
        for impl in impls:
            if impl.impl_id == args.baseline:
                filtered_impls.append(impl)
                continue

            if impl.impl_id in KNOWN_SINGLE_THREAD_IMPLS:
                multicore_scope_exclusions.append(
                    {
                        "impl_id": impl.impl_id,
                        "reason": "known_single_thread_impl",
                    }
                )
                continue

            probe = probe_impl_threads(
                repo_root=repo_root,
                env=env,
                impl=impl,
                dataset_sha=dataset_sha,
                cpu_affinity=cpu_affinity,
                nice=args.nice,
                timeout_s=args.timeout_sec,
            )
            multicore_scope_probes[impl.impl_id] = probe
            if not bool(probe.get("ok", False)):
                reason = str(probe.get("reason", "probe_failed"))
                multicore_scope_exclusions.append(
                    {
                        "impl_id": impl.impl_id,
                        "reason": reason,
                        "probe": probe,
                    }
                )
                warn(f"{impl.impl_id}: excluded from scalable-only scope ({reason})", warnings)
                continue

            max_threads_probe = probe.get("max_threads")
            if not bool(probe.get("scalable", False)):
                multicore_scope_exclusions.append(
                    {
                        "impl_id": impl.impl_id,
                        "reason": f"probe_max_threads={max_threads_probe}",
                        "probe": probe,
                    }
                )
                continue

            filtered_impls.append(impl)
            expected_scalable_impls.add(impl.impl_id)
            scalable_count += 1

        impls = filtered_impls
        if multicore_scope_exclusions:
            print(
                "[SCOPE] Excluded impls: "
                + ", ".join(f"{e.get('impl_id')}({e.get('reason')})" for e in multicore_scope_exclusions)
            )
        if scalable_count == 0:
            print(
                "[FAIL] run-mode multi-core with --multicore-scope scalable-only found no scalable implementation",
                file=sys.stderr,
            )
            return 1

    if not impls:
        raise SystemExit("no runnable implementation available after preflight checks")

    if args.dry_run:
        dry_checks: List[Tuple[bool, str, str]] = []
        dry_failed = False

        def add_check(ok: bool, name: str, detail: str) -> None:
            nonlocal dry_failed
            dry_checks.append((ok, name, detail))
            if not ok:
                dry_failed = True

        if args.enforce_single_thread:
            for key in THREAD_ENV_VARS:
                ok = env.get(key) == "1"
                add_check(ok, f"thread-env:{key}", f"value={env.get(key)} expected=1")
            go_selected = any(i.language == "go" for i in impls)
            if go_selected:
                ok = env.get("GOMAXPROCS") == "1"
                add_check(ok, "go:GOMAXPROCS", f"value={env.get('GOMAXPROCS')} expected=1")
            rust_par_selected = any(i.impl_id == "rust-par" for i in impls)
            if rust_par_selected:
                ok = env.get("RAYON_NUM_THREADS") == "1"
                add_check(ok, "rust:RAYON_NUM_THREADS", f"value={env.get('RAYON_NUM_THREADS')} expected=1")

        if java_selected:
            eff = list(java_resolution.get("effective_opts", []))
            if run_mode == "multi-core":
                add_check(
                    "-XX:ActiveProcessorCount=1" not in set(eff),
                    "java:multi-core-opts",
                    "effective=" + json.dumps(eff),
                )
            else:
                add_check(
                    java_opts_are_stable(eff),
                    "java:stable-opts",
                    "effective=" + json.dumps(eff),
                )

        c_info = build_info_by_lang.get("c", {})
        if c_info:
            cflags = str(c_info.get("CFLAGS", ""))
            cprof = str(c_info.get("PROFILE", ""))
            add_check(("-O3" in cflags) or ("-O2" in cflags), "c:opt-flag", cflags)
            add_check(cprof == args.profile, "c:profile", f"make={cprof} expected={args.profile}")
            if args.profile == "native":
                add_check("-march=native" in cflags, "c:native-flag", cflags)
            if args.profile == "portable":
                add_check("-march=native" not in cflags, "c:portable-flag", cflags)

        cpp_info = build_info_by_lang.get("cpp", {})
        if cpp_info:
            cxxflags = str(cpp_info.get("CXXFLAGS", ""))
            cpp_prof = str(cpp_info.get("PROFILE", ""))
            add_check(("-O3" in cxxflags) or ("-O2" in cxxflags), "cpp:opt-flag", cxxflags)
            add_check(cpp_prof == args.profile, "cpp:profile", f"make={cpp_prof} expected={args.profile}")
            if args.profile == "native":
                add_check("-march=native" in cxxflags, "cpp:native-flag", cxxflags)
            if args.profile == "portable":
                add_check("-march=native" not in cxxflags, "cpp:portable-flag", cxxflags)

        rust_selected = any(i.language == "rust" for i in impls)
        if rust_selected:
            cargo_prof = read_rust_release_profile(repo_root / "rust" / "Cargo.toml")
            add_check(bool(cargo_prof.get("ok")), "rust:release-profile", "; ".join(cargo_prof.get("issues", [])) or "ok")
            rust_impl = next((i for i in impls if i.language == "rust"), None)
            if rust_impl is not None:
                add_check("/release/" in rust_impl.command[0], "rust:release-binary-path", rust_impl.command[0])

        dry_self_checks: Dict[str, Dict[str, object]] = {}
        for impl in impls:
            sc_cmd = [*impl.command, "--expected-dataset-sha", dataset_sha, "--self-check"]
            proc = subprocess.run(sc_cmd, cwd=str(repo_root), env=env, capture_output=True, text=True)
            if proc.returncode != 0:
                add_check(False, f"{impl.impl_id}:self-check", f"exit={proc.returncode} stderr={tail_lines(proc.stderr, 5)}")
                continue
            try:
                obj = parse_self_check(proc.stdout, impl.impl_id)
                dry_self_checks[impl.impl_id] = obj
            except Exception as exc:
                add_check(False, f"{impl.impl_id}:self-check-parse", str(exc))
                continue
            ok_sha = str(obj.get("dataset_sha256")) == dataset_sha
            add_check(ok_sha, f"{impl.impl_id}:dataset-sha", f"got={obj.get('dataset_sha256')} expected={dataset_sha}")

            if impl.language == "go" and args.enforce_single_thread:
                rt = obj.get("runtime") if isinstance(obj.get("runtime"), dict) else {}
                gomax = int(rt.get("gomaxprocs", -1)) if rt and rt.get("gomaxprocs") is not None else -1
                add_check(gomax == 1, "go:runtime-gomaxprocs", f"value={gomax} expected=1")

            if impl.impl_id == "rust-par" and args.enforce_single_thread:
                rt = obj.get("runtime") if isinstance(obj.get("runtime"), dict) else {}
                rayon_env = int(rt.get("rayon_num_threads_env", -1)) if rt and rt.get("rayon_num_threads_env") is not None else -1
                add_check(rayon_env == 1, "rust-par:runtime-rayon-num-threads", f"value={rayon_env} expected=1")

            if impl.language == "rust":
                rt = obj.get("runtime") if isinstance(obj.get("runtime"), dict) else {}
                dbg = bool(rt.get("debug_assertions", True))
                abort = bool(rt.get("panic_abort", False))
                add_check(not dbg, "rust:debug-assertions", f"value={dbg} expected=false")
                add_check(abort, "rust:panic-abort", f"value={abort} expected=true")

        checksums = [float(v.get("checksum")) for v in dry_self_checks.values() if v.get("checksum") is not None]
        if checksums:
            base = checksums[0]
            all_equal = all(math.isclose(c, base, rel_tol=REL_TOL, abs_tol=ABS_TOL) for c in checksums[1:])
            add_check(all_equal, "self-check:checksum-consistency", f"count={len(checksums)}")

        print("[DRY-RUN] " + mode_banner(run_mode, multicore_scope))
        print("[DRY-RUN] preflight checks passed")
        print("[DRY-RUN] runnable implementations: " + ", ".join(i.impl_id for i in impls))
        if run_mode == "multi-core":
            print(f"[DRY-RUN] multicore scope: {multicore_scope}")
            if multicore_scope_exclusions:
                print("[DRY-RUN] excluded impls: " + ", ".join(f"{e.get('impl_id')}({e.get('reason')})" for e in multicore_scope_exclusions))
        if java_selected:
            print(
                "[DRY-RUN] java opts requested="
                + json.dumps(java_resolution.get("requested_opts", []))
                + " effective="
                + json.dumps(java_resolution.get("effective_opts", []))
            )
        for ok, name, detail in dry_checks:
            tag = "OK" if ok else "FAIL"
            print(f"[DRY-RUN][{tag}] {name}: {detail}")
        return 1 if dry_failed else 0

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
                runtime_meta = meta_line.get("runtime") if isinstance(meta_line.get("runtime"), dict) else {}
                build_profile_meta = str(runtime_meta.get("build_profile", ""))
                if build_profile_meta and build_profile_meta != args.profile:
                    raise ValueError(
                        f"build profile mismatch for {impl.impl_id}: meta={build_profile_meta}, expected={args.profile}"
                    )
                if args.profile == "native" and "-march=native" not in flags:
                    raise ValueError(f"native profile requested but -march=native missing for {impl.impl_id}: {flags}")
                if args.profile == "portable" and "-march=native" in flags:
                    raise ValueError(f"portable profile requested but -march=native present for {impl.impl_id}: {flags}")

            if impl.language == "rust":
                runtime_meta = meta_line.get("runtime") if isinstance(meta_line.get("runtime"), dict) else {}
                if bool(runtime_meta.get("debug_assertions", True)):
                    raise ValueError("rust build has debug_assertions=true; release build required")
                if not bool(runtime_meta.get("panic_abort", False)):
                    raise ValueError("rust build has panic_abort=false; expected true for benchmark profile")

            if impl.language == "go" and args.enforce_single_thread:
                runtime_meta = meta_line.get("runtime") if isinstance(meta_line.get("runtime"), dict) else {}
                gomax = runtime_meta.get("gomaxprocs")
                if gomax is None or int(gomax) != 1:
                    raise ValueError(f"go runtime gomaxprocs must be 1 under enforce-single-thread (got {gomax})")

            if impl.language == "java" and args.warmup < 5 and args.runs <= 5:
                warn("java warmup/runs low; JIT/GC stability may be poor", warnings)
            if impl.language == "java":
                if run_mode == "multi-core":
                    if "-XX:ActiveProcessorCount=1" in set(java_resolution.get("effective_opts", [])):
                        warn("java effective options include ActiveProcessorCount=1 in multi-core mode", warnings)
                else:
                    if not java_opts_are_stable(java_resolution.get("effective_opts", [])):
                        raise ValueError("java effective options are not stable for benchmark profile")

            if run_mode == "single-core":
                runtime_meta = meta_line.get("runtime") if isinstance(meta_line.get("runtime"), dict) else {}
                if impl.language == "go":
                    gomax = runtime_meta.get("gomaxprocs")
                    if gomax is not None and int(gomax) != 1:
                        warn(
                            f"{impl.impl_id}: runtime gomaxprocs={gomax} but single-core mode expects 1",
                            warnings,
                        )
                if impl.impl_id == "rust-par":
                    rayon_env = runtime_meta.get("rayon_num_threads_env")
                    if rayon_env is not None and int(rayon_env) != 1:
                        warn(
                            f"{impl.impl_id}: rayon_num_threads_env={rayon_env} but single-core mode expects 1",
                            warnings,
                        )

            if run_mode == "multi-core" and configured_threads is not None:
                runtime_meta = meta_line.get("runtime") if isinstance(meta_line.get("runtime"), dict) else {}
                if impl.language == "go":
                    gomax = runtime_meta.get("gomaxprocs")
                    if gomax is not None and int(gomax) != configured_threads:
                        warn(
                            f"{impl.impl_id}: runtime gomaxprocs={gomax} differs from configured T={configured_threads}",
                            warnings,
                        )
                if impl.impl_id == "rust-par":
                    rayon_env = runtime_meta.get("rayon_num_threads_env")
                    if rayon_env is not None and int(rayon_env) != configured_threads:
                        warn(
                            f"{impl.impl_id}: rayon_num_threads_env={rayon_env} differs from configured T={configured_threads}",
                            warnings,
                        )

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
                "variant": impl.variant,
                "run_mode": run_mode,
                "configured_threads": configured_threads,
                "multicore_scope": multicore_scope if run_mode == "multi-core" else "",
                "build_profile": args.profile if impl.language in ("c", "cpp") else "",
                "affinity_observed": timed.get("affinity_observed"),
                "runner_wall_ns": timed.get("runner_wall_ns"),
                "runner_cpu_ns": timed.get("runner_cpu_ns"),
                "process_returncode": timed.get("returncode"),
                "command_executed": shell_join(timed.get("command", [])) if isinstance(timed.get("command"), list) else "",
                "env_injected": json.dumps(env_injected, sort_keys=True),
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
        elif run_mode == "multi-core" and timed.get("max_threads") and int(timed["max_threads"]) <= 1:
            if multicore_scope == "scalable-only":
                if impl.impl_id in expected_scalable_impls:
                    warn(
                        f"{impl.impl_id}: expected scalable in multi-core scope but observed max_threads={timed.get('max_threads')}",
                        warnings,
                    )
            else:
                warn(
                    f"{impl.impl_id}: multi-core mode but observed max_threads={timed.get('max_threads')}",
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
            "variant": impl.variant,
            "run_mode": run_mode,
            "configured_threads": configured_threads,
            "multicore_scope": multicore_scope if run_mode == "multi-core" else "",
            "meta": meta_line,
            "runs": len(run_ids),
            "median_wall_ns": median(wall_vals),
            "median_cpu_ns": median(cpu_vals),
            "median_checksum": median(checksum_vals),
            "checksum_min": min(checksum_vals),
            "checksum_max": max(checksum_vals),
            "command_executed": shell_join(timed.get("command", [])) if isinstance(timed.get("command"), list) else "",
            "env_injected": dict(env_injected),
            "runner_wall_ns": timed.get("runner_wall_ns"),
            "runner_cpu_ns": timed.get("runner_cpu_ns"),
            "preflight_stability": stability_report,
            "build_info": build_info_by_lang.get(impl.language, {}),
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
        "variant",
        "run_mode",
        "configured_threads",
        "multicore_scope",
        "build_profile",
        "affinity_observed",
        "runner_wall_ns",
        "runner_cpu_ns",
        "process_returncode",
        "command_executed",
        "env_injected",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results_rows:
            writer.writerow(row)

    tool_versions = collect_tool_versions()

    verifications = {
        "run_mode_guardrails": {
            "status": "pass",
            "run_mode": run_mode,
            "multicore_scope": multicore_scope if run_mode == "multi-core" else "n/a",
            "cpu_affinity": cpu_affinity,
            "cpu_set_effective": cpu_affinity,
            "affinity_cpu_count": affinity_count,
            "enforce_single_thread": args.enforce_single_thread,
            "configured_threads": configured_threads,
        },
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
        "go_single_thread": {
            "status": "fail" if any("gomaxprocs" in str(e["error"]).lower() for e in critical_errors) else "pass",
            "enforced": args.enforce_single_thread,
        },
        "compilation_flags": {
            "status": "fail" if any("build flags" in e["error"] for e in critical_errors) else "pass",
        },
        "c_cpp_profile": {
            "status": "fail" if any("build profile mismatch" in e["error"] or "profile requested" in e["error"] for e in critical_errors) else "pass",
            "profile": args.profile,
        },
        "rust_release": {
            "status": "fail" if any("rust build has" in str(e["error"]).lower() for e in critical_errors) else "pass",
        },
        "java_stability": {
            "status": "fail" if any("java effective options are not stable" in str(e["error"]).lower() for e in critical_errors) else "pass",
            "effective_opts": java_resolution.get("effective_opts", []),
        },
        "resource_gating": {
            "status": "fail" if any("resource gating failed" in e["error"] for e in critical_errors) else "pass",
            "enabled": bool(args.stability_enable),
            "mode": args.stability_mode,
        },
        "multicore_scope_filter": {
            "status": "pass",
            "run_mode": run_mode,
            "scope": multicore_scope if run_mode == "multi-core" else "n/a",
            "excluded_count": len(multicore_scope_exclusions),
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
            "run_mode": run_mode,
            "multicore_scope": multicore_scope if run_mode == "multi-core" else "n/a",
            "warmup": args.warmup,
            "runs": args.runs,
            "repeat": args.repeat,
            "build_profile": args.profile,
            "cpu_affinity": cpu_affinity,
            "cpu_set_effective": cpu_affinity,
            "configured_threads": configured_threads,
            "nice": args.nice,
            "disable_python": bool(args.disable_python),
            "variants_filter": args.variants,
            "baseline_forced_added": baseline_forced_added,
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
            "injected": env_injected,
        },
        "mode": {
            "id": run_mode,
            "label": mode_banner(run_mode, multicore_scope),
            "configured_threads": configured_threads,
            "multicore_scope": multicore_scope if run_mode == "multi-core" else "n/a",
            "cpu_set_effective": cpu_affinity,
        },
        "multicore_scope": {
            "active": bool(run_mode == "multi-core" and multicore_scope == "scalable-only"),
            "scope": multicore_scope if run_mode == "multi-core" else "n/a",
            "excluded_impls": multicore_scope_exclusions,
            "probe_by_impl": multicore_scope_probes,
            "expected_scalable_impls": sorted(expected_scalable_impls),
        },
        "build_info_by_language": build_info_by_lang,
        "versions": tool_versions,
        "baseline_impl_id": baseline_id,
        "implementations": [impl_summaries[k] for k in sorted(impl_summaries.keys())],
        "warnings": warnings,
        "critical_errors": critical_errors,
        "verifications": verifications,
        "results_csv": str(csv_path),
    }

    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"\n{mode_banner(run_mode, multicore_scope)}")
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
