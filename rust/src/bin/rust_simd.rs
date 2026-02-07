#[path = "../bench_common.rs"]
mod bench_common;

use bench_common::{cpu_time_ns, delta, load_dataset, parse_common_args, proc_snapshot};
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

fn cosine_all_pairs_checksum_simd(e: &[f64], a: &[f64], n: usize, m: usize, d: usize) -> f64 {
    let a_normed = prepare_axes_normalized(a, m, d);

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

fn main() {
    let args = parse_common_args();

    let dataset = match load_dataset(Path::new(&args.metadata)) {
        Ok(ds) => ds,
        Err(e) => {
            eprintln!("failed to load dataset: {}", e);
            std::process::exit(1);
        }
    };

    if args.self_check {
        let checksum = cosine_all_pairs_checksum_simd(&dataset.e, &dataset.a, dataset.n, dataset.m, dataset.d);
        println!(
            "{{\"type\":\"self_check\",\"impl\":\"rust-simd\",\"ok\":true,\"N\":{},\"M\":{},\"D\":{},\"dataset_sha256\":\"{}\",\"checksum\":{},\"runtime\":{{\"debug_assertions\":{},\"panic_abort\":{}}}}}",
            dataset.n,
            dataset.m,
            dataset.d,
            args.expected_dataset_sha,
            checksum,
            cfg!(debug_assertions),
            cfg!(panic = "abort")
        );
        return;
    }

    println!(
        "{{\"type\":\"meta\",\"impl\":\"rust-simd\",\"N\":{},\"M\":{},\"D\":{},\"repeat\":{},\"warmup\":{},\"runs\":{},\"dataset_sha256\":\"{}\",\"build_flags\":\"\",\"runtime\":{{\"rust_pkg_version\":\"{}\",\"warmup_executed\":{},\"debug_assertions\":{},\"panic_abort\":{}}}}}",
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
        cfg!(panic = "abort")
    );

    for _ in 0..args.warmup {
        let _ = cosine_all_pairs_checksum_simd(&dataset.e, &dataset.a, dataset.n, dataset.m, dataset.d);
    }

    for run_id in 0..args.runs {
        let before = proc_snapshot();
        let t0 = Instant::now();
        let c0 = cpu_time_ns();

        let mut checksum_acc = 0.0f64;
        for _ in 0..args.repeat {
            checksum_acc += cosine_all_pairs_checksum_simd(&dataset.e, &dataset.a, dataset.n, dataset.m, dataset.d);
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
            "{{\"type\":\"run\",\"impl\":\"rust-simd\",\"run_id\":{},\"wall_ns\":{},\"cpu_ns\":{},\"checksum\":{},\"max_rss_kb\":{},\"ctx_voluntary\":{},\"ctx_involuntary\":{},\"minor_faults\":{},\"major_faults\":{},\"max_threads\":{}}}",
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
