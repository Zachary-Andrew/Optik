use std::collections::HashMap;
use std::io::{BufWriter, Write};
use std::time::Instant;
use crate::{iterate_fasta, extract_kmers, Index, OpenAddr, UnorderedMap,
            dram_latency_ns, estimate_llc_misses_per_1k,
            mean, stddev, dynamic_lookahead, RssWatermark,
            LOOKAHEAD_MIN, LOOKAHEAD_MAX};

// Sentinel values from elastic_hash.hpp.
const EMPTY_KEY: u64 = u64::MAX;
const TOMBSTONE: u64 = u64::MAX - 1;

struct TrialOut {
    build_s:    f64,
    rss_mb:     f64,
    idx_mb:     f64,
    qns:        f64,
    llc:        f64,
    // Same queries but shuffled into random order.
    qns_random: f64,
    llc_random: f64,
}

#[derive(Default)]
struct CellStats {
    build_s:    Vec<f64>,
    rss_mb:     Vec<f64>,
    idx_mb:     Vec<f64>,
    qns:        Vec<f64>,
    llc:        Vec<f64>,
    qns_random: Vec<f64>,
    llc_random: Vec<f64>,
}

impl CellStats {
    fn push(&mut self, r: &TrialOut) {
        self.build_s.push(r.build_s);
        self.rss_mb.push(r.rss_mb);
        self.idx_mb.push(r.idx_mb);
        self.qns.push(r.qns);
        self.llc.push(r.llc);
        self.qns_random.push(r.qns_random);
        self.llc_random.push(r.llc_random);
    }
}

fn shuffle_queries(queries: &[u64]) -> Vec<u64> {
    let mut v = queries.to_vec();
    let mut rng = 0x9e3779b97f4a7c15u64;
    for i in (1..v.len()).rev() {
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
        let j = (rng as usize) % (i + 1);
        v.swap(i, j);
    }
    v
}

// Build exactly as build.rs does: stage then build.
fn timed_build_tpoptoa(counts: &HashMap<u64, u32>, k: usize, hwm: &mut RssWatermark) -> (f64, Index) {
    let n = counts.len();
    let tb = Instant::now();
    let mut idx = Index::new(k, n);
    for (&kmer, &cnt) in counts {
        idx.stage(kmer, cnt);
    }
    idx.build();
    let build_s = tb.elapsed().as_secs_f64();
    hwm.sample();
    (build_s, idx)
}

fn timed_build_stdopen(counts: &HashMap<u64, u32>, hwm: &mut RssWatermark) -> (f64, OpenAddr) {
    let n = counts.len();
    let tb = Instant::now();
    let mut ht = OpenAddr::new(n * 3 / 2 + 64);
    for (&kmer, &cnt) in counts {
        ht.insert(kmer, cnt);
    }
    let build_s = tb.elapsed().as_secs_f64();
    hwm.sample();
    (build_s, ht)
}

fn timed_build_stdunordered(counts: &HashMap<u64, u32>, hwm: &mut RssWatermark) -> (f64, UnorderedMap) {
    let n = counts.len();
    let tb = Instant::now();
    let mut umap = UnorderedMap::new(n * 3 / 2 + 64);
    for (&kmer, &cnt) in counts {
        umap.insert(kmer, cnt);
    }
    let build_s = tb.elapsed().as_secs_f64();
    hwm.sample();
    (build_s, umap)
}

// Query exactly as query.rs does: prefetch pipeline with lookahead.
fn timed_query_tp(idx: &Index, queries: &[u64], dram_ns: f64, lookahead: usize) -> (f64, f64) {
    let l = lookahead;
    let mut sink = 0u64;
    // Warmup: populate branch predictors and TLB.
    for i in 0..queries.len() {
        if i + l < queries.len() { idx.prefetch(queries[i + l]); }
        if let Some(v) = idx.find(queries[i]) { sink ^= v as u64; }
    }
    let t = Instant::now();
    for i in 0..queries.len() {
        if i + l < queries.len() { idx.prefetch(queries[i + l]); }
        if let Some(v) = idx.find(queries[i]) { sink ^= v as u64; }
    }
    std::hint::black_box(sink);
    let qns = t.elapsed().as_nanos() as f64 / queries.len() as f64;
    (qns, estimate_llc_misses_per_1k(qns, dram_ns))
}

fn timed_query_open(ht: &OpenAddr, queries: &[u64], dram_ns: f64, lookahead: usize) -> (f64, f64) {
    let l = lookahead;
    let mut sink = 0u64;
    for i in 0..queries.len() {
        if i + l < queries.len() { ht.prefetch(queries[i + l]); }
        if let Some(v) = ht.find(queries[i]) { sink ^= v as u64; }
    }
    let t = Instant::now();
    for i in 0..queries.len() {
        if i + l < queries.len() { ht.prefetch(queries[i + l]); }
        if let Some(v) = ht.find(queries[i]) { sink ^= v as u64; }
    }
    std::hint::black_box(sink);
    let qns = t.elapsed().as_nanos() as f64 / queries.len() as f64;
    (qns, estimate_llc_misses_per_1k(qns, dram_ns))
}

// std::unordered_map has no prefetch API.
fn timed_query_umap(umap: &UnorderedMap, queries: &[u64], dram_ns: f64) -> (f64, f64) {
    let mut sink = 0u64;
    for &q in queries { if let Some(v) = umap.find(q) { sink ^= v as u64; } }
    let t = Instant::now();
    for &q in queries { if let Some(v) = umap.find(q) { sink ^= v as u64; } }
    std::hint::black_box(sink);
    let qns = t.elapsed().as_nanos() as f64 / queries.len() as f64;
    (qns, estimate_llc_misses_per_1k(qns, dram_ns))
}

fn run_tpoptoa(counts: &HashMap<u64, u32>, queries: &[u64],
               rand_queries: &[u64], k: usize, dram_ns: f64, lookahead: usize) -> TrialOut {
    let mut hwm = RssWatermark::new();
    let (build_s, idx) = timed_build_tpoptoa(counts, k, &mut hwm);
    let (qns, llc)               = timed_query_tp(&idx, queries, dram_ns, lookahead);
    let (qns_random, llc_random) = timed_query_tp(&idx, rand_queries, dram_ns, lookahead);
    TrialOut {
        build_s, rss_mb: hwm.net_mb(),
        idx_mb: idx.bytes() as f64 / (1 << 20) as f64,
        qns, llc, qns_random, llc_random,
    }
}

fn run_stdopen(counts: &HashMap<u64, u32>, queries: &[u64],
               rand_queries: &[u64], dram_ns: f64, lookahead: usize) -> TrialOut {
    let mut hwm = RssWatermark::new();
    let (build_s, ht) = timed_build_stdopen(counts, &mut hwm);
    let (qns, llc)               = timed_query_open(&ht, queries, dram_ns, lookahead);
    let (qns_random, llc_random) = timed_query_open(&ht, rand_queries, dram_ns, lookahead);
    TrialOut {
        build_s, rss_mb: hwm.net_mb(),
        idx_mb: ht.bytes() as f64 / (1 << 20) as f64,
        qns, llc, qns_random, llc_random,
    }
}

fn run_stdunordered(counts: &HashMap<u64, u32>, queries: &[u64],
                    rand_queries: &[u64], dram_ns: f64) -> TrialOut {
    let mut hwm = RssWatermark::new();
    let (build_s, umap) = timed_build_stdunordered(counts, &mut hwm);
    let (qns, llc)               = timed_query_umap(&umap, queries, dram_ns);
    let (qns_random, llc_random) = timed_query_umap(&umap, rand_queries, dram_ns);
    TrialOut {
        build_s, rss_mb: hwm.net_mb(),
        idx_mb: umap.bytes() as f64 / (1 << 20) as f64,
        qns, llc, qns_random, llc_random,
    }
}

fn print_main_table(tp: &CellStats, so: &CellStats, su: &CellStats,
                    trials: usize, n_keys: usize, lookahead: usize) {
    println!("\n{}", "=".repeat(70));
    println!("  RESULTS  ({n_keys} k-mers, {trials} trials)");
    println!("{}\n", "=".repeat(70));
    println!("  {:<22}  {:>16}  {:>16}  {:>16}",
             "", "TP+OptOA", "StdOpen", "StdUnordered");
    println!("  {:<22}  {:>16}  {:>16}  {:>16}",
             "", "avg (sd)", "avg (sd)", "avg (sd)");
    println!("  {}", "-".repeat(74));
    let row = |label: &str, a: &[f64], b: &[f64], c: &[f64], unit: &str| {
        println!("  {:<22}  {:8.3} ({:5.3})  {:8.3} ({:5.3})  {:8.3} ({:5.3})  {}",
                 label, mean(a), stddev(a), mean(b), stddev(b), mean(c), stddev(c), unit);
    };
    row("build time",      &tp.build_s,    &so.build_s,    &su.build_s,    "s");
    row("build RSS",       &tp.rss_mb,     &so.rss_mb,     &su.rss_mb,     "MB");
    row("index size",      &tp.idx_mb,     &so.idx_mb,     &su.idx_mb,     "MB");
    row("query (genomic)", &tp.qns,        &so.qns,        &su.qns,        "ns/query");
    row("query (random)",  &tp.qns_random, &so.qns_random, &su.qns_random, "ns/query");
    println!("\n  {} trials, lookahead [{LOOKAHEAD_MIN},{LOOKAHEAD_MAX}] -> {lookahead} for this dataset",
             trials);
}

fn write_tsv(path: &str, n_keys: usize,
             tp: &CellStats, so: &CellStats, su: &CellStats) -> Result<(), String> {
    let mut f = BufWriter::new(std::fs::File::create(path).map_err(|e| e.to_string())?);
    writeln!(f, "method\tn_keys\ttrial\tbuild_sec\trss_mb\tidx_mb\tquery_ns\tllc_est\tquery_ns_random\tllc_est_random")
        .map_err(|e| e.to_string())?;
    for (name, cell) in [("TP+OptOA", tp), ("StdOpen", so), ("StdUnordered", su)] {
        for t in 0..cell.build_s.len() {
            writeln!(f, "{}\t{}\t{}\t{:.6}\t{:.3}\t{:.3}\t{:.4}\t{:.4}\t{:.4}\t{:.4}",
                     name, n_keys, t + 1,
                     cell.build_s[t], cell.rss_mb[t], cell.idx_mb[t],
                     cell.qns[t], cell.llc[t],
                     cell.qns_random[t], cell.llc_random[t])
                .map_err(|e| e.to_string())?;
        }
    }
    Ok(())
}

pub fn run(input: &str, k: usize, trials: usize, tsv_path: &str) -> Result<(), String> {
    eprintln!("[bench] input={input}  trials={trials}  k={k}  out={tsv_path}");

    eprintln!("[bench] loading k-mers...");
    let mut queries: Vec<u64> = Vec::new();
    let mut counts: HashMap<u64, u32> = HashMap::with_capacity(1 << 22);
    iterate_fasta(input, |_name, seq| {
        extract_kmers(seq, k, |kmer| {
            // Filter sentinels before touching counts or queries.
            if kmer == EMPTY_KEY || kmer == TOMBSTONE { return; }
            queries.push(kmer);
            *counts.entry(kmer).or_insert(0) += 1;
        });
    })?;

    let n = counts.len();
    if n == 0 { return Err("no valid k-mers extracted".into()); }

    let rand_queries = shuffle_queries(&queries);
    let lookahead = {
        let bytes = std::fs::metadata(input).map(|m| m.len()).unwrap_or(0);
        dynamic_lookahead(bytes)
    };
    eprintln!("[bench] {n} distinct k-mers  {} queries  lookahead={lookahead}",
              queries.len());

    // Measure DRAM latency once for LLC estimates.
    let dram_ns = dram_latency_ns(32 * 1024 * 1024);
    eprintln!("[bench] DRAM baseline {dram_ns:.1} ns");

    // Single DRAM flush before the trial loop.
    let _ = dram_latency_ns(32 * 1024 * 1024);

    let (mut tp_stats, mut so_stats, mut su_stats) =
        (CellStats::default(), CellStats::default(), CellStats::default());

    for t in 0..trials {
        // Alternate which backend goes first each trial.
        if t % 2 == 0 {
            let ra = run_tpoptoa(&counts, &queries, &rand_queries, k, dram_ns, lookahead);
            let rb = run_stdopen(&counts, &queries, &rand_queries, dram_ns, lookahead);
            let rc = run_stdunordered(&counts, &queries, &rand_queries, dram_ns);
            eprintln!("[bench]   trial {:2}/{}  TP={:.1}ns  SO={:.1}ns  SU={:.1}ns",
                      t + 1, trials, ra.qns, rb.qns, rc.qns);
            tp_stats.push(&ra); so_stats.push(&rb); su_stats.push(&rc);
        } else {
            let rb = run_stdopen(&counts, &queries, &rand_queries, dram_ns, lookahead);
            let ra = run_tpoptoa(&counts, &queries, &rand_queries, k, dram_ns, lookahead);
            let rc = run_stdunordered(&counts, &queries, &rand_queries, dram_ns);
            eprintln!("[bench]   trial {:2}/{}  TP={:.1}ns  SO={:.1}ns  SU={:.1}ns",
                      t + 1, trials, ra.qns, rb.qns, rc.qns);
            tp_stats.push(&ra); so_stats.push(&rb); su_stats.push(&rc);
        }
    }

    print_main_table(&tp_stats, &so_stats, &su_stats, trials, n, lookahead);

    write_tsv(tsv_path, n, &tp_stats, &so_stats, &su_stats)?;
    eprintln!("[bench] raw data written to {tsv_path}");
    Ok(())
}
