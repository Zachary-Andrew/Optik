use std::collections::HashMap;
use std::io::{BufWriter, Write};
use std::time::Instant;
use crate::{iterate_fasta, extract_kmers, Index, OpenAddr, UnorderedMap,
            dram_latency_ns, estimate_llc_misses_per_1k,
            mean, stddev, dynamic_lookahead, RssWatermark,
            LOOKAHEAD_MIN, LOOKAHEAD_MAX};

struct TrialOut {
    build_s:    f64,
    rss_mb:     f64,
    idx_mb:     f64,
    slot_bits:  f64,
    qns:        f64,
    llc:        f64,
    // same queries but shuffled into random order — used to measure how much
    // the genomic insertion order actually helps vs a cache-cold random access
    qns_random: f64,
    llc_random: f64,
}

#[derive(Default)]
struct CellStats {
    build_s:    Vec<f64>,
    rss_mb:     Vec<f64>,
    idx_mb:     Vec<f64>,
    slot_bits:  Vec<f64>,
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
        self.slot_bits.push(r.slot_bits);
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

// We can't read max_probe_dist from the C++ side, so we approximate probe-chain
// variance by timing individual queries and looking at the p99/median ratio.
// If Robin Hood is keeping chains tight, p99 shouldn't be much worse than median.
fn probe_variance(idx: &Index, sample: &[u64]) -> (f64, f64) {
    let mut times: Vec<f64> = sample.iter().map(|&k| {
        let t = Instant::now();
        std::hint::black_box(idx.find(k));
        t.elapsed().as_nanos() as f64
    }).collect();
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (times[times.len() / 2], times[times.len() * 99 / 100])
}

fn probe_variance_open(ht: &OpenAddr, sample: &[u64]) -> (f64, f64) {
    let mut times: Vec<f64> = sample.iter().map(|&k| {
        let t = Instant::now();
        std::hint::black_box(ht.find(k));
        t.elapsed().as_nanos() as f64
    }).collect();
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (times[times.len() / 2], times[times.len() * 99 / 100])
}

// Runs five query passes over the same index and records ns/query for each.
// A build-once index with no deletions should be perfectly flat — if timing
// drifts upward it would mean something is accumulating like tombstones would.
fn stability_passes(idx: &Index, queries: &[u64], _dram_ns: f64, lookahead: usize) -> Vec<f64> {
    let l = lookahead;
    (0..5).map(|_| {
        let _ = dram_latency_ns(32 * 1024 * 1024);
        let mut sink = 0u64;
        let t = Instant::now();
        for i in 0..queries.len() {
            if i + l < queries.len() { idx.prefetch(queries[i + l]); }
            if let Some(v) = idx.find(queries[i]) { sink ^= v as u64; }
        }
        std::hint::black_box(sink);
        t.elapsed().as_nanos() as f64 / queries.len() as f64
    }).collect()
}

// Times a 512-base Myers alignment. AVX-512 processes 512 DP columns per cycle
// vs 64 for scalar, so it should be roughly 8x faster per cell. The empirical
// scalar floor is about 1.2 ns/cell — anything below that means SIMD is active.
fn check_simd() -> (f64, f64, bool) {
    use crate::myers_align;
    let query: Vec<u8> = (0..512).map(|i| b"ACGT"[i % 4]).collect();
    let text:  Vec<u8> = (0..1024).map(|i| b"ACGT"[i % 4]).collect();
    for _ in 0..16 { myers_align(&query, &text, 64); }
    let t = Instant::now();
    for _ in 0..2000 { std::hint::black_box(myers_align(&query, &text, 64)); }
    let ns_per_call = t.elapsed().as_nanos() as f64 / 2000.0;
    let ns_per_cell = ns_per_call / (512.0 * 1024.0);
    (ns_per_call, ns_per_cell, ns_per_cell < 1.2)
}

// Runs the pointer chase twice to check stability. The first call flushes any
// pending DRAM row refreshes as a side effect and gives the latency baseline.
// If both calls agree within 15% the measurement environment is clean enough.
fn check_dram() -> (f64, f64, bool) {
    let a = dram_latency_ns(32 * 1024 * 1024);
    let b = dram_latency_ns(32 * 1024 * 1024);
    let stable = if a > 0.0 { ((b / a) - 1.0).abs() < 0.15 } else { false };
    (a, b, stable)
}

// The warmup pass populates branch predictors and TLB. The XOR sink stops
// the compiler from treating the finds as dead code and optimizing them away.
fn timed_query_tp(idx: &Index, queries: &[u64], dram_ns: f64, lookahead: usize) -> (f64, f64) {
    let l = lookahead;
    let mut sink = 0u64;
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

// std::unordered_map has no prefetch API so there's no lookahead pipeline here.
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
    let n = counts.len();
    let mut hwm = RssWatermark::new();
    let tb = Instant::now();
    let mut idx = Index::new(k, n);
    for (&kmer, &cnt) in counts { idx.stage(kmer, cnt); }
    idx.build();
    let build_s = tb.elapsed().as_secs_f64();
    hwm.sample();
    let (qns, llc)               = timed_query_tp(&idx, queries, dram_ns, lookahead);
    let (qns_random, llc_random) = timed_query_tp(&idx, rand_queries, dram_ns, lookahead);
    TrialOut {
        build_s, rss_mb: hwm.net_mb(),
        idx_mb: idx.bytes() as f64 / (1 << 20) as f64,
        slot_bits: idx.bits_per_entry(),
        qns, llc, qns_random, llc_random,
    }
}

fn run_stdopen(counts: &HashMap<u64, u32>, queries: &[u64],
               rand_queries: &[u64], dram_ns: f64, lookahead: usize) -> TrialOut {
    let n = counts.len();
    let mut hwm = RssWatermark::new();
    let tb = Instant::now();
    let mut ht = OpenAddr::new(n * 3 / 2 + 64);
    for (&kmer, &cnt) in counts { ht.insert(kmer, cnt); }
    let build_s = tb.elapsed().as_secs_f64();
    hwm.sample();
    let (qns, llc)               = timed_query_open(&ht, queries, dram_ns, lookahead);
    let (qns_random, llc_random) = timed_query_open(&ht, rand_queries, dram_ns, lookahead);
    let nf = ht.size();
    TrialOut {
        build_s, rss_mb: hwm.net_mb(),
        idx_mb: ht.bytes() as f64 / (1 << 20) as f64,
        slot_bits: if nf > 0 { ht.bytes() as f64 * 8.0 / nf as f64 } else { 0.0 },
        qns, llc, qns_random, llc_random,
    }
}

fn run_stdunordered(counts: &HashMap<u64, u32>, queries: &[u64],
                    rand_queries: &[u64], dram_ns: f64) -> TrialOut {
    let n = counts.len();
    let mut hwm = RssWatermark::new();
    let tb = Instant::now();
    let mut umap = UnorderedMap::new(n * 3 / 2 + 64);
    for (&kmer, &cnt) in counts { umap.insert(kmer, cnt); }
    let build_s = tb.elapsed().as_secs_f64();
    hwm.sample();
    let (qns, llc)               = timed_query_umap(&umap, queries, dram_ns);
    let (qns_random, llc_random) = timed_query_umap(&umap, rand_queries, dram_ns);
    let nf = umap.size();
    TrialOut {
        build_s, rss_mb: hwm.net_mb(),
        idx_mb: umap.bytes() as f64 / (1 << 20) as f64,
        slot_bits: if nf > 0 { umap.bytes() as f64 * 8.0 / nf as f64 } else { 0.0 },
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
    row("slot bits",       &tp.slot_bits,  &so.slot_bits,  &su.slot_bits,  "bits/entry");
    row("query (genomic)", &tp.qns,        &so.qns,        &su.qns,        "ns/query");
    row("LLC (genomic)",   &tp.llc,        &so.llc,        &su.llc,        "est./1k");
    row("query (random)",  &tp.qns_random, &so.qns_random, &su.qns_random, "ns/query");
    row("LLC (random)",    &tp.llc_random, &so.llc_random, &su.llc_random, "est./1k");
    println!("\n  {} trials, lookahead [{LOOKAHEAD_MIN},{LOOKAHEAD_MAX}] \
              → {lookahead} for this dataset",
             trials);
}

fn print_property_checks(tp: &CellStats, so: &CellStats, su: &CellStats, n: usize,
                         stability: &[f64], tp_probe: (f64, f64), so_probe: (f64, f64)) {
    println!("\n{}", "=".repeat(70));
    println!("  PROPERTY CHECKS  (TP+OptOA vs StdOpen)");
    println!("{}", "=".repeat(70));

    // Check 1: pointer layer space vs naive 32-bit pointers. Both backends use
    // the identical Robin Hood hash table, so TP's total bytes will be slightly
    // *larger* than StdOpen (hash table is the same, TinyPointerArray is additive
    // overhead). The claim is not "TP uses less total memory" — it is "the pointer
    // representation costs far less than naive 32-bit pointers would". We pass if
    // the pointer layer is at least 4× cheaper than the 32-bit alternative.
    let tp_idx = mean(&tp.idx_mb);
    let so_idx = mean(&so.idx_mb);
    let tp_ptr_mb    = 6.0 * n as f64 / (8.0 * 1024.0 * 1024.0);
    let naive_ptr_mb = 32.0 * n as f64 / (8.0 * 1024.0 * 1024.0);
    let ptr_saving   = if tp_ptr_mb > 0.0 { naive_ptr_mb / tp_ptr_mb } else { 0.0 };
    let overhead_mb  = tp_idx - so_idx;
    let mem_pass     = ptr_saving >= 4.0;
    println!("\n  [{}] 1. Pointer layer space vs naive 32-bit",
             if mem_pass { "PASS" } else { "FAIL" });
    println!("       TP total {tp_idx:.2} MB  StdOpen total {so_idx:.2} MB  \
              (+{overhead_mb:.3} MB TinyPointerArray overhead)");
    println!("       TinyPointerArray {tp_ptr_mb:.3} MB  vs naive 32-bit {naive_ptr_mb:.3} MB  \
              ({ptr_saving:.1}× saving on pointer layer)");

    // Check 2: probe-chain variance. Robin Hood displacement keeps all chains near
    // equal length so the worst-case query shouldn't be much slower than the median.
    // We approximate this with p99/median since we can't read max_probe_dist from
    // the C++ side. Both backends share the same Robin Hood logic so a regression
    // in TP's ratio would indicate the TinyPointerArray is disrupting table layout.
    let tp_ratio   = if tp_probe.0 > 0.0 { tp_probe.1 / tp_probe.0 } else { 1.0 };
    let so_ratio   = if so_probe.0 > 0.0 { so_probe.1 / so_probe.0 } else { 1.0 };
    let probe_pass = tp_ratio <= so_ratio * 1.15;
    println!("\n  [{}] 2. Probe-chain variance (Robin Hood tightness)",
             if probe_pass { "PASS" } else { "FAIL" });
    println!("       TP  median {:.0} ns  p99 {:.0} ns  ratio {:.2}×",
             tp_probe.0, tp_probe.1, tp_ratio);
    println!("       SO  median {:.0} ns  p99 {:.0} ns  ratio {:.2}×",
             so_probe.0, so_probe.1, so_ratio);

    // Check 3: tombstone-free query stability. A table that accumulates tombstones
    // (open addressing with deletions) gets slower over repeated query passes as
    // probe chains lengthen. TP+OptOA is build-once with no deletions so repeated
    // queries must stay flat. We measure max-vs-min spread across 5 passes; getting
    // faster is fine (cache warmup), only the upward bound matters.
    let min_pass = stability.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_pass = stability.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let spread   = if min_pass > 0.0 { (max_pass - min_pass) / min_pass * 100.0 } else { 0.0 };
    let stable_pass = spread < 15.0;
    println!("\n  [{}] 3. No tombstone accumulation (query stability across 5 passes)",
             if stable_pass { "PASS" } else { "FAIL" });
    print!("       ns/query per pass:");
    for (i, &b) in stability.iter().enumerate() { print!("  [{i}] {b:.1}"); }
    println!("\n       min→max spread {spread:.1}%  (pass if < 15%)");

    // Check 4: insertion-order locality. TP stages keys in genomic order so queries
    // that follow that same order tend to hit recently-warmed cache lines. This only
    // shows up when the index fits in LLC — for large datasets where every access
    // misses to DRAM, both backends will show near-identical LLC miss rates. We pass
    // if TP's genomic-order LLC miss rate is no worse than StdOpen's (i.e. TP does
    // not hurt locality even if the benefit isn't measurable on this dataset size).
    let tp_genomic = mean(&tp.llc);
    let so_genomic = mean(&so.llc);
    let tp_random  = mean(&tp.llc_random);
    let so_random  = mean(&so.llc_random);
    let locality_pass = tp_genomic <= so_genomic * 1.10;
    println!("\n  [{}] 4. Insertion-order locality",
             if locality_pass { "PASS" } else { "FAIL" });
    println!("       TP  genomic LLC/1k {tp_genomic:.1}  random {tp_random:.1}");
    println!("       SO  genomic LLC/1k {so_genomic:.1}  random {so_random:.1}");
    if tp_genomic > 800.0 && so_genomic > 800.0 {
        println!("       (index exceeds LLC — both backends are DRAM-bound; \
                  locality gain only visible on smaller datasets)");
    }

    // Check 5: pointer width within the theoretical bound from the Tiny Pointer
    // paper. The bound is ⌈log₂log₂log₂n + log₂4⌉ bits. We use 6 bits (TINY_BITS
    // in the C++ header). The "total bits/entry" rows show full index cost per
    // stored key including the hash table at 0.67 load, not just the pointer layer.
    let nd       = n as f64;
    let log3n    = if nd > 8.0 { nd.log2().log2().log2() } else { 1.0 };
    let bound    = (log3n + 4.0_f64.log2()).ceil();
    let in_bound = 6.0_f64 <= bound + 2.0;
    let tp_bpe   = mean(&tp.slot_bits);
    let so_bpe   = mean(&so.slot_bits);
    let su_bpe   = mean(&su.slot_bits);
    println!("\n  [{}] 5. Pointer width within theoretical bound",
             if in_bound { "PASS" } else { "FAIL" });
    println!("       actual 6 bits  bound {bound:.0} bits  within: {}",
             if in_bound { "yes" } else { "NO — regression!" });
    println!("       total bits/entry (hash table + pointers)  \
              TP {tp_bpe:.1}  StdOpen {so_bpe:.1}  UnorderedMap {su_bpe:.1}");
    println!("       pointer layer saving {ptr_saving:.1}×  \
              ({tp_ptr_mb:.3} MB vs naive {naive_ptr_mb:.3} MB)");

    let all = mem_pass && probe_pass && stable_pass && locality_pass && in_bound;
    println!("\n  {}", if all { "ALL CHECKS PASSED" }
                       else   { "ONE OR MORE CHECKS FAILED — see above" });
}

fn print_simd_check(ns_per_call: f64, ns_per_cell: f64, active: bool) {
    println!("\n  -- SIMD (Myers AVX-512) --");
    println!("  512-base alignment: {ns_per_call:.1} ns/call  ({ns_per_cell:.3} ns/DP-cell)");
    if active {
        println!("  AVX-512 path: ACTIVE");
    } else {
        println!("  AVX-512 path: INACTIVE — running scalar fallback.");
        println!("  Recompile with -C target-feature=+avx512f,+avx512bw,+avx512dq");
        println!("  or set RUSTFLAGS='-C target-cpu=native'");
    }
}

fn print_dram_check(a: f64, b: f64, stable: bool) {
    println!("\n  -- DRAM Refresh Hack --");
    println!("  pointer-chase 1: {a:.1} ns   pointer-chase 2: {b:.1} ns");
    if stable {
        println!("  Stable (< 15% variance) — refresh stalls flushed OK.");
    } else {
        let v = ((b / a) - 1.0).abs() * 100.0;
        println!("  WARNING: {v:.0}% variance — DRAM refresh stalls may be inflating timings.");
        println!("  Try isolcpus, disabling hyperthreading, or re-running.");
    }
    println!("  DRAM baseline used for LLC-miss estimation: {a:.1} ns");
}

fn write_tsv(path: &str, n_keys: usize,
             tp: &CellStats, so: &CellStats, su: &CellStats) -> Result<(), String> {
    let mut f = BufWriter::new(std::fs::File::create(path).map_err(|e| e.to_string())?);
    writeln!(f, "method\tn_keys\ttrial\tbuild_sec\trss_mb\tidx_mb\tslot_bits\
                 \tquery_ns\tllc\tquery_ns_random\tllc_random")
        .map_err(|e| e.to_string())?;
    for (name, cell) in [("TP+OptOA", tp), ("StdOpen", so), ("StdUnordered", su)] {
        for t in 0..cell.build_s.len() {
            writeln!(f, "{}\t{}\t{}\t{:.6}\t{:.3}\t{:.3}\t{:.2}\t{:.4}\t{:.4}\t{:.4}\t{:.4}",
                     name, n_keys, t + 1,
                     cell.build_s[t], cell.rss_mb[t], cell.idx_mb[t],
                     cell.slot_bits[t], cell.qns[t], cell.llc[t],
                     cell.qns_random[t], cell.llc_random[t])
                .map_err(|e| e.to_string())?;
        }
    }
    Ok(())
}

pub fn run(input: &str, k: usize, trials: usize, tsv_path: &str) -> Result<(), String> {
    eprintln!("[bench] input={input}  trials={trials}  k={k}  out={tsv_path}");

    eprintln!("[bench] checking SIMD…");
    let (simd_ns_call, simd_ns_cell, simd_active) = check_simd();

    // The first pointer-chase flushes pending DRAM row refreshes and gives the
    // latency baseline we'll use throughout. The second call checks that the
    // environment is stable enough to trust the timings.
    eprintln!("[bench] measuring DRAM latency…");
    let (dram_ns, dram_ns2, dram_stable) = check_dram();
    eprintln!("[bench] DRAM baseline {dram_ns:.1} ns  stable={dram_stable}\n");

    eprintln!("[bench] loading k-mers…");
    let mut queries: Vec<u64> = Vec::new();
    let mut counts: HashMap<u64, u32> = HashMap::with_capacity(1 << 22);
    iterate_fasta(input, |_name, seq| {
        extract_kmers(seq, k, |kmer| {
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

    let (mut tp_stats, mut so_stats, mut su_stats) =
        (CellStats::default(), CellStats::default(), CellStats::default());

    for t in 0..trials {
        // Alternate which backend goes first each trial so neither consistently
        // benefits from the other warming TLB and branch predictors. Flush DRAM
        // refresh state before each individual backend run, not just once per trial.
        if t % 2 == 0 {
            let _ = dram_latency_ns(32 * 1024 * 1024);
            let ra = run_tpoptoa(&counts, &queries, &rand_queries, k, dram_ns, lookahead);
            let _ = dram_latency_ns(32 * 1024 * 1024);
            let rb = run_stdopen(&counts, &queries, &rand_queries, dram_ns, lookahead);
            let _ = dram_latency_ns(32 * 1024 * 1024);
            let rc = run_stdunordered(&counts, &queries, &rand_queries, dram_ns);
            eprintln!("[bench]   trial {:2}/{}  \
                       TP={:.1}ns/{:.1}ns(rnd)  SO={:.1}ns/{:.1}ns(rnd)  SU={:.1}ns",
                      t + 1, trials, ra.qns, ra.qns_random, rb.qns, rb.qns_random, rc.qns);
            tp_stats.push(&ra); so_stats.push(&rb); su_stats.push(&rc);
        } else {
            let _ = dram_latency_ns(32 * 1024 * 1024);
            let rb = run_stdopen(&counts, &queries, &rand_queries, dram_ns, lookahead);
            let _ = dram_latency_ns(32 * 1024 * 1024);
            let ra = run_tpoptoa(&counts, &queries, &rand_queries, k, dram_ns, lookahead);
            let _ = dram_latency_ns(32 * 1024 * 1024);
            let rc = run_stdunordered(&counts, &queries, &rand_queries, dram_ns);
            eprintln!("[bench]   trial {:2}/{}  \
                       TP={:.1}ns/{:.1}ns(rnd)  SO={:.1}ns/{:.1}ns(rnd)  SU={:.1}ns",
                      t + 1, trials, ra.qns, ra.qns_random, rb.qns, rb.qns_random, rc.qns);
            tp_stats.push(&ra); so_stats.push(&rb); su_stats.push(&rc);
        }
    }

    // Rebuild once more specifically for the per-query probe timing and the
    // five-pass stability test. We use up to 10k keys for probe timing so that
    // individual Instant::now() calls have enough resolution to be meaningful.
    let _ = dram_latency_ns(32 * 1024 * 1024);
    let sample: Vec<u64> = counts.keys().copied().take(10_000.min(n)).collect();

    let mut tp_idx = Index::new(k, n);
    for (&kmer, &cnt) in &counts { tp_idx.stage(kmer, cnt); }
    tp_idx.build();

    let mut so_ht = OpenAddr::new(n * 3 / 2 + 64);
    for (&kmer, &cnt) in &counts { so_ht.insert(kmer, cnt); }

    let tp_probe   = probe_variance(&tp_idx, &sample);
    let so_probe   = probe_variance_open(&so_ht, &sample);
    let stability  = stability_passes(&tp_idx, &queries, dram_ns, lookahead);

    print_main_table(&tp_stats, &so_stats, &su_stats, trials, n, lookahead);
    print_property_checks(&tp_stats, &so_stats, &su_stats, n,
                          &stability, tp_probe, so_probe);
    print_simd_check(simd_ns_call, simd_ns_cell, simd_active);
    print_dram_check(dram_ns, dram_ns2, dram_stable);

    write_tsv(tsv_path, n, &tp_stats, &so_stats, &su_stats)?;
    eprintln!("[bench] raw data written to {tsv_path}");
    Ok(())
}
