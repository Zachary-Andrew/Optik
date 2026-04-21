use std::collections::{HashMap, VecDeque};
use std::io::{self, BufWriter, Write};
use std::time::Instant;
use crate::{iterate_fasta, extract_minimizers, myers_align, Index, dynamic_lookahead};

pub fn run(ref_path: &str, reads_path: &str, k: usize, w: usize, max_ed: i32)
    -> Result<(), String>
{
    eprintln!("[map] loading reference {ref_path} (k={k} w={w})…");
    let t0 = Instant::now();

    let mut ref_names: Vec<String>  = Vec::new();
    let mut ref_seqs:  Vec<Vec<u8>> = Vec::new();
    iterate_fasta(ref_path, |name, seq| {
        ref_names.push(name.to_string());
        ref_seqs.push(seq.to_vec());
    })?;

    // Count minimizers in a first pass so we can reserve exactly the right
    // capacity before inserting, avoiding rehashes.
    let mut n_mini = 0usize;
    for seq in &ref_seqs { extract_minimizers(seq, k, w, |_, _| n_mini += 1); }

    let mut pos_map: HashMap<u64, Vec<(u32, u32)>> = HashMap::with_capacity(n_mini);
    for (rid, seq) in ref_seqs.iter().enumerate() {
        extract_minimizers(seq, k, w, |kmer, pos| {
            pos_map.entry(kmer).or_default().push((rid as u32, pos));
        });
    }

    // The TP+OptOA index only stores presence; the actual hit positions live
    // in pos_map. We query the index first because it is cache-friendly and
    // lets us skip the HashMap lookup entirely for minimizers with no hits.
    let mut idx = Index::new(k, pos_map.len());
    for &kmer in pos_map.keys() { idx.stage(kmer, 1); }
    idx.build();

    let lookahead = {
        let bytes = std::fs::metadata(reads_path).map(|m| m.len()).unwrap_or(0);
        dynamic_lookahead(bytes)
    };
    eprintln!("[map] {} minimizers indexed in {:.2}s — lookahead={lookahead}",
              pos_map.len(), t0.elapsed().as_secs_f64());

    let stdout = io::stdout();
    let mut out = BufWriter::new(stdout.lock());
    writeln!(out, "@HD\tVN:1.6\tSO:unsorted").unwrap();
    for (i, name) in ref_names.iter().enumerate() {
        writeln!(out, "@SQ\tSN:{name}\tLN:{}", ref_seqs[i].len()).unwrap();
    }
    writeln!(out, "@PG\tID:optik\tPN:optik\tVN:0.1").unwrap();

    let (mut n_reads, mut n_mapped) = (0u64, 0u64);
    let tmap = Instant::now();
    let avg_ref_len = if ref_seqs.is_empty() { 100 }
                      else { ref_seqs.iter().map(|s| s.len()).sum::<usize>() / ref_seqs.len() };
    let half = (avg_ref_len / 2).max(50);

    // VecDeque gives O(1) pop_front for the prefetch pipeline window.
    // The old Vec::remove(0) was O(lookahead) — a shift of the whole buffer.
    let mut mini_window: VecDeque<u64> = VecDeque::with_capacity(lookahead + 8);

    iterate_fasta(reads_path, |read_name, seq| {
        n_reads += 1;
        let (mut best_ed, mut best_rid, mut best_pos) = (max_ed + 1, 0usize, 0usize);

        mini_window.clear();
        extract_minimizers(seq, k, w, |kmer, _read_pos| {
            if mini_window.len() >= lookahead {
                idx.prefetch(kmer);
                let cur = mini_window.pop_front().unwrap(); // O(1) — was O(lookahead)
                if idx.find(cur).is_none() { return; }
                if let Some(hits) = pos_map.get(&cur) {
                    for &(rid, rpos) in hits {
                        let rseq  = &ref_seqs[rid as usize];
                        let start = (rpos as usize).saturating_sub(half);
                        let end   = ((rpos as usize) + seq.len() + half).min(rseq.len());
                        let (ed, _) = myers_align(seq, &rseq[start..end], best_ed);
                        if ed < best_ed { best_ed = ed; best_rid = rid as usize; best_pos = start; }
                    }
                }
            }
            mini_window.push_back(kmer);
        });

        for cur in mini_window.drain(..) {
            if idx.find(cur).is_none() { continue; }
            if let Some(hits) = pos_map.get(&cur) {
                for &(rid, rpos) in hits {
                    let rseq  = &ref_seqs[rid as usize];
                    let start = (rpos as usize).saturating_sub(half);
                    let end   = ((rpos as usize) + seq.len() + half).min(rseq.len());
                    let (ed, _) = myers_align(seq, &rseq[start..end], best_ed);
                    if ed < best_ed { best_ed = ed; best_rid = rid as usize; best_pos = start; }
                }
            }
        }

        let seq_s = std::str::from_utf8(seq).unwrap_or("*");
        if best_ed <= max_ed {
            n_mapped += 1;
            let mapq  = ((max_ed - best_ed + 1) * 10).min(60) as u8;
            let cigar = format!("{}M", seq.len());
            writeln!(out,
                "{read_name}\t0\t{}\t{}\t{mapq}\t{cigar}\t*\t0\t0\t{seq_s}\t*\tNM:i:{best_ed}",
                ref_names[best_rid], best_pos + 1).unwrap();
        } else {
            writeln!(out,
                "{read_name}\t4\t*\t0\t0\t*\t*\t0\t0\t{seq_s}\t*").unwrap();
        }
    })?;

    let pct = 100.0 * n_mapped as f64 / n_reads.max(1) as f64;
    eprintln!("[map] {n_reads} reads in {:.2}s — {n_mapped} mapped ({pct:.1}%)",
              tmap.elapsed().as_secs_f64());
    Ok(())
}
