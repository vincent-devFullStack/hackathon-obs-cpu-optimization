import glob
import os
import shutil
import statistics
import subprocess
import time
from typing import Dict, List, Optional, Tuple


PERF_EVENTS = [
    "cycles",
    "instructions",
    "cache-misses",
    "branches",
    "branch-misses",
    "cpu-migrations",
    "context-switches",
    "page-faults",
    "minor-faults",
    "major-faults",
]

THREAD_ENV_VARS = [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
]


def command_exists(name: str) -> bool:
    return shutil.which(name) is not None


def perf_available() -> bool:
    if not command_exists("perf"):
        return False
    try:
        proc = subprocess.run(["perf", "--version"], capture_output=True, text=True)
    except Exception:
        return False
    return proc.returncode == 0


def apply_cpu_nice_prefix(command: List[str], cpu_affinity: Optional[str], nice: Optional[int]) -> List[str]:
    prefixed = list(command)
    if cpu_affinity:
        prefixed = ["taskset", "-c", cpu_affinity] + prefixed
    if nice is not None:
        prefixed = ["nice", "-n", str(nice)] + prefixed
    return prefixed


def _read_proc_status(pid: int) -> Dict[str, Optional[int]]:
    out: Dict[str, Optional[int]] = {
        "threads": None,
        "vm_hwm_kb": None,
        "ctx_voluntary": None,
        "ctx_involuntary": None,
        "cpus_allowed_list": None,
    }
    path = f"/proc/{pid}/status"
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("Threads:"):
                    out["threads"] = int(line.split(":", 1)[1].strip())
                elif line.startswith("VmHWM:"):
                    out["vm_hwm_kb"] = int(line.split(":", 1)[1].strip().split()[0])
                elif line.startswith("voluntary_ctxt_switches:"):
                    out["ctx_voluntary"] = int(line.split(":", 1)[1].strip())
                elif line.startswith("nonvoluntary_ctxt_switches:"):
                    out["ctx_involuntary"] = int(line.split(":", 1)[1].strip())
                elif line.startswith("Cpus_allowed_list:"):
                    out["cpus_allowed_list"] = line.split(":", 1)[1].strip()
    except Exception:
        pass
    return out


def _read_proc_stat(pid: int) -> Dict[str, Optional[int]]:
    out: Dict[str, Optional[int]] = {
        "minor_faults": None,
        "major_faults": None,
    }
    path = f"/proc/{pid}/stat"
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = f.read().strip()
        rparen = payload.rfind(")")
        if rparen <= 0:
            return out
        tail = payload[rparen + 2 :]
        fields = tail.split()
        out["minor_faults"] = int(fields[7])
        out["major_faults"] = int(fields[9])
    except Exception:
        pass
    return out


def _read_proc_sched(pid: int) -> Dict[str, Optional[int]]:
    out: Dict[str, Optional[int]] = {"cpu_migrations": None}
    path = f"/proc/{pid}/sched"
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip().startswith("nr_migrations"):
                    out["cpu_migrations"] = int(line.split(":", 1)[1].strip())
                    break
    except Exception:
        pass
    return out


def _proc_snapshot(pid: int) -> Dict[str, Optional[int]]:
    snap: Dict[str, Optional[int]] = {}
    snap.update(_read_proc_status(pid))
    snap.update(_read_proc_stat(pid))
    snap.update(_read_proc_sched(pid))
    return snap


def _delta(end_v: Optional[int], start_v: Optional[int]) -> Optional[int]:
    if end_v is None or start_v is None:
        return None
    return int(end_v - start_v)


def run_with_metrics(
    command: List[str],
    cwd: str,
    env: Dict[str, str],
    cpu_affinity: Optional[str] = None,
    nice: Optional[int] = None,
    timeout_s: Optional[int] = None,
    poll_interval_s: float = 0.01,
) -> Dict[str, object]:
    full_cmd = apply_cpu_nice_prefix(command, cpu_affinity=cpu_affinity, nice=nice)

    t0_wall = time.perf_counter_ns()
    t0_times = os.times()
    proc = subprocess.Popen(
        full_cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    start = _proc_snapshot(proc.pid)
    last = dict(start)
    max_threads = start.get("threads") or 0
    max_rss_kb = start.get("vm_hwm_kb") or 0
    affinity_observed = start.get("cpus_allowed_list")
    timed_out = False

    while True:
        if timeout_s is not None and (time.perf_counter_ns() - t0_wall) > (timeout_s * 1_000_000_000):
            proc.kill()
            timed_out = True
            break

        snap = _proc_snapshot(proc.pid)
        if any(v is not None for v in snap.values()):
            last = snap
            if snap.get("threads") is not None:
                max_threads = max(max_threads, int(snap["threads"]))
            if snap.get("vm_hwm_kb") is not None:
                max_rss_kb = max(max_rss_kb, int(snap["vm_hwm_kb"]))
            if snap.get("cpus_allowed_list"):
                affinity_observed = snap.get("cpus_allowed_list")

        if proc.poll() is not None:
            break
        time.sleep(poll_interval_s)

    stdout, stderr = proc.communicate()

    t1_times = os.times()
    t1_wall = time.perf_counter_ns()
    cpu_s = (t1_times.children_user + t1_times.children_system) - (t0_times.children_user + t0_times.children_system)
    cpu_ns = int(cpu_s * 1_000_000_000)

    return {
        "command": full_cmd,
        "pid": proc.pid,
        "returncode": proc.returncode,
        "timed_out": timed_out,
        "stdout": stdout,
        "stderr": stderr,
        "runner_wall_ns": int(t1_wall - t0_wall),
        "runner_cpu_ns": cpu_ns,
        "max_rss_kb": int(max_rss_kb) if max_rss_kb else None,
        "max_threads": int(max_threads) if max_threads else None,
        "ctx_switches_voluntary": _delta(last.get("ctx_voluntary"), start.get("ctx_voluntary")),
        "ctx_switches_involuntary": _delta(last.get("ctx_involuntary"), start.get("ctx_involuntary")),
        "page_faults_minor": _delta(last.get("minor_faults"), start.get("minor_faults")),
        "page_faults_major": _delta(last.get("major_faults"), start.get("major_faults")),
        "cpu_migrations": _delta(last.get("cpu_migrations"), start.get("cpu_migrations")),
        "affinity_observed": affinity_observed,
    }


def parse_perf_stat(stderr_text: str) -> Dict[str, Optional[float]]:
    values: Dict[str, Optional[float]] = {}
    for line in stderr_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(",")
        if len(parts) < 3:
            continue

        value_str = parts[0].strip()
        event = parts[2].strip().replace(":u", "").replace(":k", "")

        if value_str in ("", "<not counted>", "<not supported>"):
            values[event] = None
            continue

        value_str = value_str.replace(" ", "")
        try:
            values[event] = float(value_str)
        except ValueError:
            values[event] = None

    return values


def run_perf_stat(
    command: List[str],
    cwd: str,
    env: Dict[str, str],
    cpu_affinity: Optional[str] = None,
    nice: Optional[int] = None,
    timeout_s: Optional[int] = None,
) -> Dict[str, object]:
    prefixed = apply_cpu_nice_prefix(command, cpu_affinity=cpu_affinity, nice=nice)
    perf_cmd = [
        "perf",
        "stat",
        "-x,",
        "--no-big-num",
        "-e",
        ",".join(PERF_EVENTS),
        "--",
        *prefixed,
    ]

    proc = subprocess.run(
        perf_cmd,
        capture_output=True,
        text=True,
        cwd=cwd,
        env=env,
        timeout=timeout_s,
    )

    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "metrics": parse_perf_stat(proc.stderr),
    }


def _read_proc_stat_global() -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {
        "cpu_total": None,
        "cpu_idle": None,
        "intr_total": None,
    }
    try:
        with open("/proc/stat", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("cpu "):
                    parts = line.split()
                    vals = [int(x) for x in parts[1:]]
                    total = float(sum(vals))
                    idle = float(vals[3] + (vals[4] if len(vals) > 4 else 0))
                    out["cpu_total"] = total
                    out["cpu_idle"] = idle
                elif line.startswith("intr "):
                    parts = line.split()
                    if len(parts) >= 2:
                        out["intr_total"] = float(parts[1])
                if out["cpu_total"] is not None and out["intr_total"] is not None:
                    break
    except Exception:
        pass
    return out


def _read_loadavg() -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {
        "load1": None,
        "run_queue_running": None,
        "run_queue_total": None,
    }
    try:
        with open("/proc/loadavg", "r", encoding="utf-8") as f:
            parts = f.read().strip().split()
        if len(parts) >= 4:
            out["load1"] = float(parts[0])
            rq = parts[3].split("/")
            if len(rq) == 2:
                out["run_queue_running"] = float(rq[0])
                out["run_queue_total"] = float(rq[1])
    except Exception:
        pass
    return out


def _read_meminfo() -> Dict[str, Optional[float]]:
    keys = {"MemTotal": None, "MemAvailable": None, "SwapTotal": None, "SwapFree": None}
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                k, _, rest = line.partition(":")
                if k in keys:
                    v = rest.strip().split()[0]
                    keys[k] = float(v)
    except Exception:
        pass
    return {
        "mem_total_kb": keys["MemTotal"],
        "mem_available_kb": keys["MemAvailable"],
        "swap_total_kb": keys["SwapTotal"],
        "swap_free_kb": keys["SwapFree"],
    }


def _read_vmstat() -> Dict[str, Optional[float]]:
    wanted = {"pswpin": None, "pswpout": None, "pgpgin": None, "pgpgout": None}
    try:
        with open("/proc/vmstat", "r", encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) != 2:
                    continue
                k, v = parts
                if k in wanted:
                    wanted[k] = float(v)
    except Exception:
        pass
    return wanted


def _is_disk_partition(dev: str) -> bool:
    if dev.startswith(("loop", "ram", "fd", "sr", "zram", "dm-", "md")):
        return True
    if dev.startswith("sd") and dev[-1].isdigit():
        return True
    if dev.startswith("nvme") and "p" in dev and dev[-1].isdigit():
        return True
    if dev.startswith("mmcblk") and "p" in dev:
        return True
    return False


def _read_diskstats_total_sectors() -> Optional[float]:
    total = 0.0
    any_ok = False
    try:
        with open("/proc/diskstats", "r", encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 14:
                    continue
                dev = parts[2]
                if _is_disk_partition(dev):
                    continue
                read_sectors = float(parts[5])
                write_sectors = float(parts[9])
                total += read_sectors + write_sectors
                any_ok = True
    except Exception:
        return None
    return total if any_ok else None


def _read_cpu_freq_khz() -> Dict[str, Optional[float]]:
    paths = glob.glob("/sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_cur_freq")
    vals: List[float] = []
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8") as f:
                vals.append(float(f.read().strip()))
        except Exception:
            continue
    if not vals:
        return {"cpu_freq_avg_khz": None, "cpu_freq_min_khz": None, "cpu_freq_max_khz": None}
    return {
        "cpu_freq_avg_khz": float(sum(vals) / len(vals)),
        "cpu_freq_min_khz": float(min(vals)),
        "cpu_freq_max_khz": float(max(vals)),
    }


def _system_snapshot() -> Dict[str, Optional[float]]:
    snap: Dict[str, Optional[float]] = {"ts": time.monotonic(), "cpu_count": float(os.cpu_count() or 1)}
    snap.update(_read_proc_stat_global())
    snap.update(_read_loadavg())
    snap.update(_read_meminfo())
    snap.update(_read_vmstat())
    snap["disk_sectors_total"] = _read_diskstats_total_sectors()
    snap.update(_read_cpu_freq_khz())
    return snap


def _collect_samples(window_sec: float, sample_interval_sec: float) -> List[Dict[str, Optional[float]]]:
    interval = max(0.05, float(sample_interval_sec))
    n = max(2, int(round(max(window_sec, interval) / interval)) + 1)
    samples: List[Dict[str, Optional[float]]] = []
    for i in range(n):
        samples.append(_system_snapshot())
        if i < n - 1:
            time.sleep(interval)
    return samples


def _safe_mean(values: List[float]) -> Optional[float]:
    return float(sum(values) / len(values)) if values else None


def _safe_max(values: List[float]) -> Optional[float]:
    return float(max(values)) if values else None


def _safe_min(values: List[float]) -> Optional[float]:
    return float(min(values)) if values else None


def _safe_var(values: List[float]) -> Optional[float]:
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    return float(statistics.pvariance(values))


def _evaluate_samples(samples: List[Dict[str, Optional[float]]], config: Dict[str, float]) -> Dict[str, object]:
    reasons: List[str] = []
    unavailable: List[str] = []

    cpu_utils: List[float] = []
    load1_vals: List[float] = []
    runq_vals: List[float] = []
    disk_mbps_vals: List[float] = []
    intr_rate_vals: List[float] = []

    mem_avail_mb_vals: List[float] = []
    mem_avail_pct_vals: List[float] = []
    freq_avg_vals: List[float] = []

    for s in samples:
        if s.get("load1") is not None:
            load1_vals.append(float(s["load1"]))
        if s.get("run_queue_running") is not None:
            runq_vals.append(float(s["run_queue_running"]))
        if s.get("mem_available_kb") is not None:
            mem_avail_mb_vals.append(float(s["mem_available_kb"]) / 1024.0)
        if s.get("mem_available_kb") is not None and s.get("mem_total_kb"):
            total = float(s["mem_total_kb"])
            if total > 0:
                mem_avail_pct_vals.append(float(s["mem_available_kb"]) * 100.0 / total)
        if s.get("cpu_freq_avg_khz") is not None:
            freq_avg_vals.append(float(s["cpu_freq_avg_khz"]))

    for i in range(1, len(samples)):
        s0 = samples[i - 1]
        s1 = samples[i]
        dt = float(s1["ts"] - s0["ts"])
        if dt <= 0:
            continue

        if s0.get("cpu_total") is not None and s1.get("cpu_total") is not None and s0.get("cpu_idle") is not None and s1.get("cpu_idle") is not None:
            d_total = float(s1["cpu_total"] - s0["cpu_total"])
            d_idle = float(s1["cpu_idle"] - s0["cpu_idle"])
            if d_total > 0:
                cpu_utils.append(max(0.0, min(100.0, 100.0 * (1.0 - (d_idle / d_total)))))

        disk_rate: Optional[float] = None
        if s0.get("disk_sectors_total") is not None and s1.get("disk_sectors_total") is not None:
            d_sectors = float(s1["disk_sectors_total"] - s0["disk_sectors_total"])
            if d_sectors >= 0:
                disk_rate = (d_sectors * 512.0) / (1024.0 * 1024.0) / dt
        elif s0.get("pgpgin") is not None and s1.get("pgpgin") is not None and s0.get("pgpgout") is not None and s1.get("pgpgout") is not None:
            d_kb = float((s1["pgpgin"] - s0["pgpgin"]) + (s1["pgpgout"] - s0["pgpgout"]))
            if d_kb >= 0:
                disk_rate = (d_kb / 1024.0) / dt
        if disk_rate is not None:
            disk_mbps_vals.append(max(0.0, disk_rate))

        if s0.get("intr_total") is not None and s1.get("intr_total") is not None:
            d_intr = float(s1["intr_total"] - s0["intr_total"])
            if d_intr >= 0:
                intr_rate_vals.append(d_intr / dt)

    cpu_count = int(samples[-1].get("cpu_count") or 1)
    load1_limit = float(config.get("load1_max_factor", 0.30)) * max(1, cpu_count)
    runq_limit = float(config.get("run_queue_max_factor", config.get("load1_max_factor", 0.30))) * max(1, cpu_count)

    cpu_util_avg = _safe_mean(cpu_utils)
    cpu_util_max = _safe_max(cpu_utils)
    cpu_util_var = _safe_var(cpu_utils)

    load1_avg = _safe_mean(load1_vals)
    load1_max = _safe_max(load1_vals)
    runq_avg = _safe_mean(runq_vals)
    runq_max = _safe_max(runq_vals)

    disk_avg = _safe_mean(disk_mbps_vals)
    disk_max = _safe_max(disk_mbps_vals)

    mem_avail_mb_min = _safe_min(mem_avail_mb_vals)
    mem_avail_pct_min = _safe_min(mem_avail_pct_vals)

    intr_rate_avg = _safe_mean(intr_rate_vals)
    freq_avg_khz = _safe_mean(freq_avg_vals)

    swap_activity = None
    swap_in = None
    swap_out = None
    if len(samples) >= 2:
        s0 = samples[0]
        s1 = samples[-1]
        if s0.get("pswpin") is not None and s1.get("pswpin") is not None and s0.get("pswpout") is not None and s1.get("pswpout") is not None:
            swap_in = float(s1["pswpin"] - s0["pswpin"])
            swap_out = float(s1["pswpout"] - s0["pswpout"])
            swap_activity = float(swap_in + swap_out)

    criteria: Dict[str, Dict[str, object]] = {}

    if cpu_util_avg is None:
        unavailable.append("cpu_util")
        criteria["cpu_util"] = {"available": False, "pass": True}
    else:
        ok = cpu_util_avg <= float(config.get("cpu_util_max", 20.0))
        criteria["cpu_util"] = {"available": True, "pass": ok, "value": cpu_util_avg, "limit": float(config.get("cpu_util_max", 20.0))}
        if not ok:
            reasons.append(f"CPU util avg too high: {cpu_util_avg:.2f}% > {float(config.get('cpu_util_max', 20.0)):.2f}%")

    if cpu_util_var is None:
        unavailable.append("cpu_util_variance")
        criteria["cpu_util_variance"] = {"available": False, "pass": True}
    else:
        ok = cpu_util_var <= float(config.get("cpu_util_variance_max", 50.0))
        criteria["cpu_util_variance"] = {"available": True, "pass": ok, "value": cpu_util_var, "limit": float(config.get("cpu_util_variance_max", 50.0))}
        if not ok:
            reasons.append(f"CPU util variance too high: {cpu_util_var:.2f} > {float(config.get('cpu_util_variance_max', 50.0)):.2f}")

    if load1_avg is None:
        unavailable.append("load1")
        criteria["load1"] = {"available": False, "pass": True}
    else:
        ok = load1_avg <= load1_limit
        criteria["load1"] = {"available": True, "pass": ok, "value": load1_avg, "limit": load1_limit}
        if not ok:
            reasons.append(f"Load1 too high: {load1_avg:.2f} > {load1_limit:.2f}")

    if runq_avg is None:
        unavailable.append("run_queue")
        criteria["run_queue"] = {"available": False, "pass": True}
    else:
        ok = runq_avg <= runq_limit
        criteria["run_queue"] = {"available": True, "pass": ok, "value": runq_avg, "limit": runq_limit}
        if not ok:
            reasons.append(f"Run queue too high: {runq_avg:.2f} > {runq_limit:.2f}")

    if disk_avg is None:
        unavailable.append("disk_io")
        criteria["disk_io"] = {"available": False, "pass": True}
    else:
        ok = disk_avg <= float(config.get("disk_io_mbps_max", 5.0))
        criteria["disk_io"] = {"available": True, "pass": ok, "value": disk_avg, "limit": float(config.get("disk_io_mbps_max", 5.0))}
        if not ok:
            reasons.append(f"Disk I/O too high: {disk_avg:.2f} MB/s > {float(config.get('disk_io_mbps_max', 5.0)):.2f} MB/s")

    if mem_avail_mb_min is None:
        unavailable.append("memory")
        criteria["memory"] = {"available": False, "pass": True}
    else:
        mb_limit = float(config.get("mem_available_min_mb", 2048.0))
        pct_limit = float(config.get("mem_available_min_percent", 15.0))
        pct_val = mem_avail_pct_min if mem_avail_pct_min is not None else 0.0
        ok = (mem_avail_mb_min >= mb_limit) or (pct_val >= pct_limit)
        criteria["memory"] = {
            "available": True,
            "pass": ok,
            "value_mb": mem_avail_mb_min,
            "limit_mb": mb_limit,
            "value_percent": pct_val,
            "limit_percent": pct_limit,
        }
        if not ok:
            reasons.append(
                f"Memory available too low: {mem_avail_mb_min:.1f} MB ({pct_val:.1f}%) < {mb_limit:.1f} MB and < {pct_limit:.1f}%"
            )

    if swap_activity is None:
        unavailable.append("swap")
        criteria["swap"] = {"available": False, "pass": True}
    else:
        ok = swap_activity <= float(config.get("swap_activity_max", 0.0))
        criteria["swap"] = {"available": True, "pass": ok, "value": swap_activity, "limit": float(config.get("swap_activity_max", 0.0))}
        if not ok:
            reasons.append(f"Swap activity too high: {swap_activity:.0f} > {float(config.get('swap_activity_max', 0.0)):.0f}")

    stable = len(reasons) == 0

    return {
        "stable": stable,
        "blocking_reasons": reasons,
        "unavailable_metrics": unavailable,
        "criteria": criteria,
        "metrics": {
            "cpu_count": cpu_count,
            "cpu_util_avg": cpu_util_avg,
            "cpu_util_max": cpu_util_max,
            "cpu_util_variance": cpu_util_var,
            "load1_avg": load1_avg,
            "load1_max": load1_max,
            "run_queue_avg": runq_avg,
            "run_queue_max": runq_max,
            "disk_io_mbps_avg": disk_avg,
            "disk_io_mbps_max": disk_max,
            "mem_available_mb_min": mem_avail_mb_min,
            "mem_available_percent_min": mem_avail_pct_min,
            "swap_in": swap_in,
            "swap_out": swap_out,
            "swap_activity": swap_activity,
            "interrupts_per_sec_avg": intr_rate_avg,
            "cpu_freq_avg_khz": freq_avg_khz,
            "sample_count": len(samples),
            "window_sec": float(samples[-1]["ts"] - samples[0]["ts"]) if len(samples) >= 2 else 0.0,
        },
    }


def wait_for_stable_system(config: Dict[str, float]) -> Dict[str, object]:
    mode = str(config.get("mode", "wait"))
    timeout_sec = float(config.get("timeout_sec", 60.0))
    backoff_sec = float(config.get("backoff_sec", 2.0))
    window_sec = float(config.get("window_sec", 6.0))
    sample_interval_sec = float(config.get("sample_interval_sec", 0.5))

    start = time.monotonic()
    attempts: List[Dict[str, object]] = []

    while True:
        samples = _collect_samples(window_sec=window_sec, sample_interval_sec=sample_interval_sec)
        eval_report = _evaluate_samples(samples, config)
        eval_report["attempt_id"] = len(attempts) + 1
        attempts.append(eval_report)

        elapsed = time.monotonic() - start
        stable = bool(eval_report.get("stable", False))

        if stable:
            return {
                "stable": True,
                "mode": mode,
                "timed_out": False,
                "waited_sec": elapsed,
                "attempts": len(attempts),
                "last": eval_report,
                "attempt_history": attempts,
            }

        if mode in ("skip", "fail"):
            return {
                "stable": False,
                "mode": mode,
                "timed_out": False,
                "waited_sec": elapsed,
                "attempts": len(attempts),
                "last": eval_report,
                "attempt_history": attempts,
            }

        if elapsed >= timeout_sec:
            return {
                "stable": False,
                "mode": mode,
                "timed_out": True,
                "waited_sec": elapsed,
                "attempts": len(attempts),
                "last": eval_report,
                "attempt_history": attempts,
            }

        time.sleep(max(0.0, backoff_sec))
