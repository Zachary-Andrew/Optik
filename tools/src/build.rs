use std::collections::HashMap;
use std::time::Instant;
use crate::{iterate_fasta, extract_kmers, Index, write_index};

// Sentinel values from elastic_hash.hpp:
//   EMPTY_KEY = std::numeric_limits<uint64_t>::max()     = u64::MAX
//   TOMBSTONE = std::numeric_limits<uint64_t>::max() - 1 = u64::MAX - 1
//
// extract_kmers() calls ok_canonical() = min(fwd, rc). For k < 32 the mask
// is (1 << 2k) - 1, so the maximum canonical value is well below both
// sentinels and no collision is possible. For k = 32 the mask is ~0u64, so
// a pathological sequence can produce a canonical k-mer that equals EMPTY_KEY
// or TOMBSTONE. Staging or querying these values asserts in debug builds of
// ElasticHashTable and silently corrupts the table in release builds.
//
// The C++ benchmark's load_dataset() filters them explicitly:
//   if (kmer == tpoptoa::EMPTY_KEY || kmer == tpoptoa::TOMBSTONE) return;
// We apply the same guard at every k-mer ingestion point.
const EMPTY_KEY: u64 = u64::MAX;
const TOMBSTONE: u64 = u64::MAX - 1;

pub fn run(input: &str, k: usize, output: &str) -> Result<(), String> {
    eprintln!("[build] pass 1: counting k-mers (k={k}) from {input}");
    let t0 = Instant::now();

    let mut counts: HashMap<u64, u32> = HashMap::with_capacity(1 << 22);
    iterate_fasta(input, |_name, seq| {
        extract_kmers(seq, k, |kmer| {
            if kmer == EMPTY_KEY || kmer == TOMBSTONE { return; }
            *counts.entry(kmer).or_insert(0) += 1;
        });
    })?;

    let n = counts.len();
    eprintln!("[build] {n} distinct k-mers in {:.2}s", t0.elapsed().as_secs_f64());
    if n == 0 { return Err("no k-mers found".into()); }

    // Collect entries before the index consumes counts. We sort for
    // deterministic output order — matches what the old sort+dedup produced,
    // and gives the query tool a file with sequential key access.
    let mut entries: Vec<(u64, u32)> = counts.iter().map(|(&k, &v)| (k, v)).collect();
    entries.sort_unstable_by_key(|e| e.0);

    eprintln!("[build] pass 2: building tpoptoa index…");
    let mut idx = Index::new(k, n);
    for (&kmer, &cnt) in &counts { idx.stage(kmer, cnt); }
    // Drop counts before build() so we never hold the staging buffer and the
    // built table simultaneously — matches the memory ordering in C++ build.cpp.
    drop(counts);
    idx.build();

    let mb = idx.bytes() as f64 / (1 << 20) as f64;
    eprintln!("[build] index ready in {:.2}s — {mb:.2} MB  ({:.1} bits/entry)",
              t0.elapsed().as_secs_f64(), idx.bits_per_entry());

    // Write directly from `entries` — no second FASTA read, no idx.find(),
    // no sort of genomic-order occurrences. The old code re-read the file,
    // queried every occurrence through the index, then sorted+deduped to
    // recover the same set counts already held. That was an extra O(genome)
    // I/O pass plus O(n log n) sort for zero benefit.
    write_index(output, k, &entries).map_err(|e| e.to_string())?;
    eprintln!("[build] wrote {output} in {:.2}s", t0.elapsed().as_secs_f64());
    Ok(())
}
