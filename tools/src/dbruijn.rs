use std::collections::HashMap;
use std::io::{BufWriter, Write};
use std::time::Instant;
use crate::{iterate_fasta, extract_kmers, decode_kmer, Index,
            ok_canonical, ok_kmer_mask, dynamic_lookahead};

pub fn run(input: &str, k: usize, min_count: u32, output: &str) -> Result<(), String> {
    eprintln!("[dbruijn] counting k-mers from {input} (k={k}, min_count={min_count})…");
    let t0 = Instant::now();

    let mut counts: HashMap<u64, u32> = HashMap::with_capacity(1 << 22);
    iterate_fasta(input, |_name, seq| {
        extract_kmers(seq, k, |kmer| { *counts.entry(kmer).or_insert(0) += 1; });
    })?;
    counts.retain(|_, v| *v >= min_count);
    let n = counts.len();
    eprintln!("[dbruijn] {n} k-mers after filtering in {:.2}s",
              t0.elapsed().as_secs_f64());

    let mut idx = Index::new(k, n);
    for (&kmer, &cnt) in &counts { idx.stage(kmer, cnt); }
    idx.build();

    let lookahead = {
        let bytes = std::fs::metadata(input).map(|m| m.len()).unwrap_or(0);
        dynamic_lookahead(bytes)
    };
    let mask = unsafe { ok_kmer_mask(k as u64) };

    // Returns how many distinct predecessors kmer has in the graph by checking
    // all four possible (k-1)-suffix predecessors against the index.
    let in_degree = |kmer: u64| -> usize {
        let suffix = kmer >> 2;
        (0u64..4).filter(|&b| {
            let pred = unsafe { ok_canonical((b << ((k as u64 - 1) * 2)) | suffix, k as u64) };
            idx.find(pred).is_some()
        }).count()
    };

    let successors = |kmer: u64| -> Vec<u64> {
        let prefix = (kmer & (mask >> 2)) << 2;
        (0u64..4).filter_map(|b| {
            let suc = unsafe { ok_canonical(prefix | b, k as u64) };
            if idx.find(suc).is_some() { Some(suc) } else { None }
        }).collect()
    };

    let mut visited: HashMap<u64, bool> = HashMap::with_capacity(n);
    let mut unitigs:    Vec<Vec<u8>> = Vec::new();
    let mut unitig_cov: Vec<u64>     = Vec::new();

    let start_keys: Vec<u64> = counts.keys().copied().collect();

    for (ki, &start) in start_keys.iter().enumerate() {
        // Prefetch upcoming candidates while processing the current one.
        if ki + lookahead < start_keys.len() {
            idx.prefetch(start_keys[ki + lookahead]);
        }

        if visited.contains_key(&start) { continue; }

        // Skip nodes that sit in the interior of a unitig — only begin
        // traversal at true chain heads (branching nodes or nodes whose
        // single predecessor has multiple successors).
        if in_degree(start) == 1 {
            let pred_suf = start >> 2;
            let pred_candidates: Vec<u64> = (0u64..4).filter_map(|b| {
                let p = unsafe {
                    ok_canonical((b << ((k as u64 - 1) * 2)) | pred_suf, k as u64)
                };
                if idx.find(p).is_some() { Some(p) } else { None }
            }).collect();
            if pred_candidates.len() == 1 && successors(pred_candidates[0]).len() == 1 {
                continue;
            }
        }

        let mut chain = vec![start];
        visited.insert(start, true);
        let mut cur = start;

        loop {
            let sucs = successors(cur);
            if sucs.len() != 1 { break; }
            let next = sucs[0];
            if visited.contains_key(&next) { break; }
            if in_degree(next) != 1 { break; }
            visited.insert(next, true);
            chain.push(next);
            cur = next;
        }

        // The first k-mer contributes all k bases; each subsequent k-mer
        // contributes only its last base (the (k-1) overlap is shared).
        let mut seq_bytes = decode_kmer(chain[0], k).into_bytes();
        const BASES: [u8; 4] = [b'A', b'C', b'G', b'T'];
        for &km in &chain[1..] { seq_bytes.push(BASES[(km & 3) as usize]); }

        let cov: u64 = chain.iter()
            .filter_map(|&km| idx.find(km).map(|c| c as u64))
            .sum();

        unitigs.push(seq_bytes);
        unitig_cov.push(cov);
    }

    let mut gfa = BufWriter::new(std::fs::File::create(output).map_err(|e| e.to_string())?);
    writeln!(gfa, "H\tVN:Z:1.0").unwrap();
    for (i, (seq, cov)) in unitigs.iter().zip(&unitig_cov).enumerate() {
        let s = std::str::from_utf8(seq).unwrap_or("*");
        writeln!(gfa, "S\t{i}\t{s}\tRC:i:{}\tFC:i:{cov}", seq.len() - k + 1).unwrap();
    }
    eprintln!("[dbruijn] {} unitigs → {output} in {:.2}s — lookahead={lookahead}",
              unitigs.len(), t0.elapsed().as_secs_f64());
    Ok(())
}
