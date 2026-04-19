use std::collections::HashMap;
use std::time::Instant;
use crate::{iterate_fasta, extract_kmers, Index, write_index};

pub fn run(input: &str, k: usize, output: &str) -> Result<(), String> {
    eprintln!("[build] pass 1: counting k-mers (k={k}) from {input}");
    let t0 = Instant::now();

    let mut counts: HashMap<u64, u32> = HashMap::with_capacity(1 << 22);
    iterate_fasta(input, |_name, seq| {
        extract_kmers(seq, k, |kmer| { *counts.entry(kmer).or_insert(0) += 1; });
    })?;

    let n = counts.len();
    eprintln!("[build] {n} distinct k-mers in {:.2}s", t0.elapsed().as_secs_f64());
    if n == 0 { return Err("no k-mers found".into()); }

    eprintln!("[build] pass 2: building tpoptoa index…");
    let mut idx = Index::new(k, n);
    for (&kmer, &cnt) in &counts { idx.stage(kmer, cnt); }
    drop(counts);
    idx.build();

    let mb = idx.bytes() as f64 / (1 << 20) as f64;
    eprintln!("[build] index ready in {:.2}s — {mb:.2} MB  ({:.1} bits/entry)",
              t0.elapsed().as_secs_f64(), idx.bits_per_entry());

    // Re-read the input to collect (kmer, count) pairs in genomic order, then
    // dedup so each canonical k-mer appears exactly once in the output file.
    let mut entries: Vec<(u64, u32)> = Vec::with_capacity(n);
    iterate_fasta(input, |_name, seq| {
        extract_kmers(seq, k, |kmer| {
            if let Some(cnt) = idx.find(kmer) { entries.push((kmer, cnt)); }
        });
    })?;
    entries.sort_unstable_by_key(|e| e.0);
    entries.dedup_by_key(|e| e.0);

    write_index(output, k, &entries).map_err(|e| e.to_string())?;
    eprintln!("[build] wrote {output} in {:.2}s", t0.elapsed().as_secs_f64());
    Ok(())
}
