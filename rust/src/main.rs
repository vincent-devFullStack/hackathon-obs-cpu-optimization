use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[repr(C)]
struct Timespec {
    tv_sec: i64,
    tv_nsec: i64,
}

extern "C" {
    fn clock_gettime(clk_id: i32, tp: *mut Timespec) -> i32;
}

const CLOCK_PROCESS_CPUTIME_ID: i32 = 2;

#[derive(Debug)]
struct Dataset {
    n: usize,
    m: usize,
    d: usize,
    e: Vec<f64>,
    a: Vec<f64>,
}

#[derive(Debug, Copy, Clone)]
struct ProcSnapshot {
    threads: i64,
    vm_hwm_kb: i64,
    ctx_voluntary: i64,
    ctx_involuntary: i64,
    minor_faults: i64,
    major_faults: i64,
}

fn parse_arg_int(value: &str, name: &str) -> Result<i32, String> {
    value
        .parse::<i32>()
        .map_err(|_| format!("invalid value for {}: {}", name, value))
}

fn extract_key_pos(json: &str, key: &str) -> Option<usize> {
    json.find(&format!("\"{}\"", key))
}

fn extract_string(json: &str, key: &str) -> Option<String> {
    let key_pos = extract_key_pos(json, key)?;
    let colon = json[key_pos..].find(':')? + key_pos;
    let mut p = colon + 1;
    while p < json.len() && json.as_bytes()[p].is_ascii_whitespace() {
        p += 1;
    }
    if p >= json.len() || json.as_bytes()[p] != b'"' {
        return None;
    }
    p += 1;
    let end = json[p..].find('"')? + p;
    Some(json[p..end].to_string())
}

fn extract_usize(json: &str, key: &str) -> Option<usize> {
    let key_pos = extract_key_pos(json, key)?;
    let colon = json[key_pos..].find(':')? + key_pos;
    let mut p = colon + 1;
    while p < json.len() && json.as_bytes()[p].is_ascii_whitespace() {
        p += 1;
    }
    let mut end = p;
    while end < json.len() && json.as_bytes()[end].is_ascii_digit() {
        end += 1;
    }
    if end == p {
        return None;
    }
    json[p..end].parse::<usize>().ok()
}

fn read_f64_file(path: &Path, expected: usize) -> Result<Vec<f64>, String> {
    let bytes = fs::read(path).map_err(|e| format!("cannot read {}: {}", path.display(), e))?;
    if bytes.len() != expected * 8 {
        return Err(format!(
            "bad size for {}: got {}, expected {}",
            path.display(),
            bytes.len(),
            expected * 8
        ));
    }
    let mut out = Vec::with_capacity(expected);
    for chunk in bytes.chunks_exact(8) {
        let mut b = [0u8; 8];
        b.copy_from_slice(chunk);
        out.push(f64::from_le_bytes(b));
    }
    Ok(out)
}

fn load_dataset(metadata_path: &Path) -> Result<Dataset, String> {
    let json = fs::read_to_string(metadata_path)
        .map_err(|e| format!("cannot read metadata {}: {}", metadata_path.display(), e))?;

    let format = extract_string(&json, "format").ok_or("missing format")?;
    if format != "cosine-benchmark-v1" {
        return Err(format!("unsupported format: {}", format));
    }

    let dtype = extract_string(&json, "dtype").ok_or("missing dtype")?;
    if dtype != "float64-le" {
        return Err(format!("unsupported dtype: {}", dtype));
    }

    let n = extract_usize(&json, "N").ok_or("missing N")?;
    let m = extract_usize(&json, "M").ok_or("missing M")?;
    let d = extract_usize(&json, "D").ok_or("missing D")?;
    let e_file = extract_string(&json, "E_file").ok_or("missing E_file")?;
    let a_file = extract_string(&json, "A_file").ok_or("missing A_file")?;

    let base = metadata_path.parent().unwrap_or_else(|| Path::new("."));
    let e_path = if Path::new(&e_file).is_absolute() {
        PathBuf::from(e_file)
    } else {
        base.join(e_file)
    };
    let a_path = if Path::new(&a_file).is_absolute() {
        PathBuf::from(a_file)
    } else {
        base.join(a_file)
    };

    let e = read_f64_file(&e_path, n * d)?;
    let a = read_f64_file(&a_path, m * d)?;

    Ok(Dataset { n, m, d, e, a })
}

fn cosine_all_pairs_checksum(ds: &Dataset) -> f64 {
    let mut axis_norms = vec![0.0f64; ds.m];
    for (j, axis_norm) in axis_norms.iter_mut().enumerate() {
        let base = j * ds.d;
        let mut s = 0.0;
        for k in 0..ds.d {
            let x = ds.a[base + k];
            s += x * x;
        }
        *axis_norm = s.sqrt();
    }

    let mut checksum = 0.0f64;
    for i in 0..ds.n {
        let e_base = i * ds.d;
        let mut emb_norm_sq = 0.0;
        for k in 0..ds.d {
            let x = ds.e[e_base + k];
            emb_norm_sq += x * x;
        }
        let emb_norm = emb_norm_sq.sqrt();
        if emb_norm == 0.0 {
            continue;
        }

        for (j, axis_norm) in axis_norms.iter().enumerate() {
            let a_base = j * ds.d;
            let mut dot = 0.0;
            for k in 0..ds.d {
                dot += ds.e[e_base + k] * ds.a[a_base + k];
            }
            let denom = emb_norm * *axis_norm;
            if denom != 0.0 {
                checksum += dot / denom;
            }
        }
    }

    checksum
}

fn cpu_time_ns() -> i128 {
    let mut ts = Timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: clock_gettime is called with a valid pointer to a Timespec instance.
    let rc = unsafe { clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &mut ts as *mut Timespec) };
    if rc != 0 {
        return -1;
    }
    (ts.tv_sec as i128) * 1_000_000_000i128 + (ts.tv_nsec as i128)
}

fn parse_i64_suffix(line: &str) -> i64 {
    line.split(':')
        .nth(1)
        .and_then(|s| s.trim().split_whitespace().next())
        .and_then(|s| s.parse::<i64>().ok())
        .unwrap_or(-1)
}

fn parse_proc_status() -> (i64, i64, i64, i64) {
    let content = fs::read_to_string("/proc/self/status").unwrap_or_default();
    let mut threads = -1;
    let mut vm_hwm_kb = -1;
    let mut ctx_voluntary = -1;
    let mut ctx_involuntary = -1;

    for line in content.lines() {
        if line.starts_with("Threads:") {
            threads = parse_i64_suffix(line);
        } else if line.starts_with("VmHWM:") {
            vm_hwm_kb = parse_i64_suffix(line);
        } else if line.starts_with("voluntary_ctxt_switches:") {
            ctx_voluntary = parse_i64_suffix(line);
        } else if line.starts_with("nonvoluntary_ctxt_switches:") {
            ctx_involuntary = parse_i64_suffix(line);
        }
    }

    (threads, vm_hwm_kb, ctx_voluntary, ctx_involuntary)
}

fn parse_proc_faults() -> (i64, i64) {
    let stat = fs::read_to_string("/proc/self/stat").unwrap_or_default();
    let rparen = match stat.rfind(')') {
        Some(v) => v,
        None => return (-1, -1),
    };
    if rparen + 2 >= stat.len() {
        return (-1, -1);
    }

    let tail = &stat[rparen + 2..];
    let f: Vec<&str> = tail.split_whitespace().collect();
    if f.len() <= 9 {
        return (-1, -1);
    }
    let minflt = f[7].parse::<i64>().unwrap_or(-1);
    let majflt = f[9].parse::<i64>().unwrap_or(-1);
    (minflt, majflt)
}

fn proc_snapshot() -> ProcSnapshot {
    let (threads, vm_hwm_kb, ctx_voluntary, ctx_involuntary) = parse_proc_status();
    let (minor_faults, major_faults) = parse_proc_faults();
    ProcSnapshot {
        threads,
        vm_hwm_kb,
        ctx_voluntary,
        ctx_involuntary,
        minor_faults,
        major_faults,
    }
}

fn delta(after: i64, before: i64) -> i64 {
    if after < 0 || before < 0 {
        -1
    } else {
        after - before
    }
}

fn main() {
    let mut metadata = String::from("data/metadata.json");
    let mut expected_dataset_sha = String::new();
    let mut warmup = 5;
    let mut runs = 30;
    let mut repeat = 50;
    let mut self_check = false;

    let args: Vec<String> = env::args().collect();
    let mut i = 1usize;
    while i < args.len() {
        match args[i].as_str() {
            "--metadata" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("missing value for --metadata");
                    std::process::exit(1);
                }
                metadata = args[i].clone();
            }
            "--warmup" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("missing value for --warmup");
                    std::process::exit(1);
                }
                warmup = match parse_arg_int(&args[i], "--warmup") {
                    Ok(v) => v,
                    Err(e) => {
                        eprintln!("{}", e);
                        std::process::exit(1);
                    }
                };
            }
            "--runs" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("missing value for --runs");
                    std::process::exit(1);
                }
                runs = match parse_arg_int(&args[i], "--runs") {
                    Ok(v) => v,
                    Err(e) => {
                        eprintln!("{}", e);
                        std::process::exit(1);
                    }
                };
            }
            "--repeat" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("missing value for --repeat");
                    std::process::exit(1);
                }
                repeat = match parse_arg_int(&args[i], "--repeat") {
                    Ok(v) => v,
                    Err(e) => {
                        eprintln!("{}", e);
                        std::process::exit(1);
                    }
                };
            }
            "--expected-dataset-sha" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("missing value for --expected-dataset-sha");
                    std::process::exit(1);
                }
                expected_dataset_sha = args[i].clone();
            }
            "--self-check" => {
                self_check = true;
            }
            _ => {
                eprintln!("Usage: {} [--metadata PATH] [--warmup N] [--runs N] [--repeat N] [--expected-dataset-sha SHA] [--self-check]", args[0]);
                std::process::exit(1);
            }
        }
        i += 1;
    }

    if warmup < 0 || runs <= 0 || repeat <= 0 {
        eprintln!("warmup must be >= 0, runs and repeat must be > 0");
        std::process::exit(1);
    }

    if expected_dataset_sha.is_empty() {
        eprintln!("missing --expected-dataset-sha");
        std::process::exit(1);
    }

    let dataset = match load_dataset(Path::new(&metadata)) {
        Ok(ds) => ds,
        Err(e) => {
            eprintln!("failed to load dataset: {}", e);
            std::process::exit(1);
        }
    };

    if self_check {
        let checksum = cosine_all_pairs_checksum(&dataset);
        println!(
            "{{\"type\":\"self_check\",\"impl\":\"rust-naive\",\"ok\":true,\"N\":{},\"M\":{},\"D\":{},\"dataset_sha256\":\"{}\",\"checksum\":{}}}",
            dataset.n,
            dataset.m,
            dataset.d,
            expected_dataset_sha,
            checksum
        );
        return;
    }

    println!(
        "{{\"type\":\"meta\",\"impl\":\"rust-naive\",\"N\":{},\"M\":{},\"D\":{},\"repeat\":{},\"warmup\":{},\"runs\":{},\"dataset_sha256\":\"{}\",\"build_flags\":\"\",\"runtime\":{{\"rust_pkg_version\":\"{}\",\"warmup_executed\":{}}}}}",
        dataset.n,
        dataset.m,
        dataset.d,
        repeat,
        warmup,
        runs,
        expected_dataset_sha,
        env!("CARGO_PKG_VERSION"),
        warmup
    );

    for _ in 0..warmup {
        let _ = cosine_all_pairs_checksum(&dataset);
    }

    for run_id in 0..runs {
        let before = proc_snapshot();
        let t0 = Instant::now();
        let c0 = cpu_time_ns();

        let mut checksum_acc = 0.0f64;
        for _ in 0..repeat {
            checksum_acc += cosine_all_pairs_checksum(&dataset);
        }

        let c1 = cpu_time_ns();
        let wall_ns = t0.elapsed().as_nanos() as i128 / (repeat as i128);
        let cpu_ns = if c0 >= 0 && c1 >= 0 {
            (c1 - c0) / (repeat as i128)
        } else {
            -1
        };
        let after = proc_snapshot();

        let max_threads = if before.threads > after.threads {
            before.threads
        } else {
            after.threads
        };

        println!(
            "{{\"type\":\"run\",\"impl\":\"rust-naive\",\"run_id\":{},\"wall_ns\":{},\"cpu_ns\":{},\"checksum\":{},\"max_rss_kb\":{},\"ctx_voluntary\":{},\"ctx_involuntary\":{},\"minor_faults\":{},\"major_faults\":{},\"max_threads\":{}}}",
            run_id,
            wall_ns,
            cpu_ns,
            checksum_acc,
            after.vm_hwm_kb,
            delta(after.ctx_voluntary, before.ctx_voluntary),
            delta(after.ctx_involuntary, before.ctx_involuntary),
            delta(after.minor_faults, before.minor_faults),
            delta(after.major_faults, before.major_faults),
            max_threads
        );
    }
}
