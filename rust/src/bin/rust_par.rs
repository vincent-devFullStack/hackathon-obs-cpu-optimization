#[path = "../bench_common.rs"]
mod bench_common;

use bench_common::{cpu_time_ns, delta, load_dataset, parse_common_args, proc_snapshot};
use rayon::prelude::*;
use std::env;
use std::path::Path;
use std::time::Instant;

fn prepare_axes_normalized(a: &[f64], m: usize, d: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; m * d];
    for j in 0..m {
        let base = j * d;
        let row = &a[base..base + d];
        let mut norm_sq = 0.0f64;
        for &x in row {
            norm_sq += x * x;
        }
        let norm = norm_sq.sqrt();
        let inv = if norm != 0.0 { 1.0 / norm } else { 0.0 };
        for k in 0..d {
            out[base + k] = row[k] * inv;
        }
    }
    out
}

#[inline(always)]
fn dot_unchecked(x: &[f64], y: &[f64]) -> f64 {
    let mut s = 0.0f64;
    let mut i = 0usize;
    let len = x.len();
    while i < len {
        // SAFETY: i is bounded by len, and x/y have same length in callers.
        unsafe {
            s += *x.get_unchecked(i) * *y.get_unchecked(i);
        }
        i += 1;
    }
    s
}

fn checksum_single(e: &[f64], a_normed: &[f64], n: usize, m: usize, d: usize) -> f64 {
    let mut checksum = 0.0f64;
    for i in 0..n {
        let e_base = i * d;
        let e_row = &e[e_base..e_base + d];

        let mut emb_norm_sq = 0.0f64;
        for &x in e_row {
            emb_norm_sq += x * x;
        }
        let emb_norm = emb_norm_sq.sqrt();
        if emb_norm == 0.0 {
            continue;
        }
        let inv_emb_norm = 1.0 / emb_norm;

        for j in 0..m {
            let a_base = j * d;
            let a_row = &a_normed[a_base..a_base + d];
            checksum += dot_unchecked(e_row, a_row) * inv_emb_norm;
        }
    }

    checksum
}

fn checksum_parallel(e: &[f64], a_normed: &[f64], n: usize, m: usize, d: usize) -> f64 {
    (0..n)
        .into_par_iter()
        .map(|i| {
            let e_base = i * d;
            let e_row = &e[e_base..e_base + d];

            let mut emb_norm_sq = 0.0f64;
            for &x in e_row {
                emb_norm_sq += x * x;
            }
            let emb_norm = emb_norm_sq.sqrt();
            if emb_norm == 0.0 {
                return 0.0f64;
            }
            let inv_emb_norm = 1.0 / emb_norm;

            let mut local = 0.0f64;
            for j in 0..m {
                let a_base = j * d;
                let a_row = &a_normed[a_base..a_base + d];
                local += dot_unchecked(e_row, a_row) * inv_emb_norm;
            }
            local
        })
        .sum::<f64>()
}

fn configured_rayon_threads() -> usize {
    match env::var("RAYON_NUM_THREADS") {
        Ok(v) => match v.parse::<usize>() {
            Ok(n) if n > 0 => n,
            _ => 0,
        },
        Err(_) => 0,
    }
}

fn main() {
    let args = parse_common_args();

    let dataset = match load_dataset(Path::new(&args.metadata)) {
        Ok(ds) => ds,
        Err(e) => {
            eprintln!("failed to load dataset: {}", e);
            std::process::exit(1);
        }
    };

    let configured_threads = configured_rayon_threads();
    let use_parallel = configured_threads == 0 || configured_threads > 1;
    let mode = if use_parallel { "parallel" } else { "serial_fallback" };

    let a_normed = prepare_axes_normalized(&dataset.a, dataset.m, dataset.d);

    let run_once = || -> f64 {
        if use_parallel {
            checksum_parallel(&dataset.e, &a_normed, dataset.n, dataset.m, dataset.d)
        } else {
            checksum_single(&dataset.e, &a_normed, dataset.n, dataset.m, dataset.d)
        }
    };

    if args.self_check {
        let checksum = run_once();
        println!(
            "{{\"type\":\"self_check\",\"impl\":\"rust-par\",\"ok\":true,\"N\":{},\"M\":{},\"D\":{},\"dataset_sha256\":\"{}\",\"checksum\":{},\"runtime\":{{\"debug_assertions\":{},\"panic_abort\":{},\"rayon_num_threads_env\":{},\"mode\":\"{}\"}}}}",
            dataset.n,
            dataset.m,
            dataset.d,
            args.expected_dataset_sha,
            checksum,
            cfg!(debug_assertions),
            cfg!(panic = "abort"),
            configured_threads,
            mode
        );
        return;
    }

    println!(
        "{{\"type\":\"meta\",\"impl\":\"rust-par\",\"N\":{},\"M\":{},\"D\":{},\"repeat\":{},\"warmup\":{},\"runs\":{},\"dataset_sha256\":\"{}\",\"build_flags\":\"\",\"runtime\":{{\"rust_pkg_version\":\"{}\",\"warmup_executed\":{},\"debug_assertions\":{},\"panic_abort\":{},\"rayon_num_threads_env\":{},\"mode\":\"{}\"}}}}",
        dataset.n,
        dataset.m,
        dataset.d,
        args.repeat,
        args.warmup,
        args.runs,
        args.expected_dataset_sha,
        env!("CARGO_PKG_VERSION"),
        args.warmup,
        cfg!(debug_assertions),
        cfg!(panic = "abort"),
        configured_threads,
        mode
    );

    for _ in 0..args.warmup {
        let _ = run_once();
    }

    for run_id in 0..args.runs {
        let before = proc_snapshot();
        let t0 = Instant::now();
        let c0 = cpu_time_ns();

        let mut checksum_acc = 0.0f64;
        for _ in 0..args.repeat {
            checksum_acc += run_once();
        }

        let c1 = cpu_time_ns();
        let wall_ns = t0.elapsed().as_nanos() as i128 / (args.repeat as i128);
        let cpu_ns = if c0 >= 0 && c1 >= 0 {
            (c1 - c0) / (args.repeat as i128)
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
            "{{\"type\":\"run\",\"impl\":\"rust-par\",\"run_id\":{},\"wall_ns\":{},\"cpu_ns\":{},\"checksum\":{},\"max_rss_kb\":{},\"ctx_voluntary\":{},\"ctx_involuntary\":{},\"minor_faults\":{},\"major_faults\":{},\"max_threads\":{}}}",
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
