use std::collections::VecDeque;
use std::io::{self, BufWriter, Write};
use std::time::Instant;
use crate::{iterate_fasta, extract_kmers, read_index, entries_to_index,
            decode_kmer, dynamic_lookahead};

pub fn run(index_path: &str, query_path: &str, only_found: bool) -> Result<(), String> {
    eprintln!("[query] loading {index_path}…");
    let t0 = Instant::now();
    let (k, entries) = read_index(index_path)?;
    let idx = entries_to_index(k, &entries);

    let lookahead = {
        let bytes = std::fs::metadata(query_path).map(|m| m.len()).unwrap_or(0);
        dynamic_lookahead(bytes)
    };
    eprintln!("[query] {} entries (k={k}) loaded in {:.2}s — lookahead={lookahead}",
              idx.size(), t0.elapsed().as_secs_f64());

    let stdout = io::stdout();
    let mut out = BufWriter::new(stdout.lock());
    let (mut n_queried, mut n_found) = (0u64, 0u64);
    let tq = Instant::now();

    // While the window is full we issue a prefetch for the kmer that just
    // arrived, then immediately resolve the oldest kmer at the front. This
    // keeps `lookahead` cache-line fetches in flight at all times.
    // VecDeque gives O(1) pop_front; the old Vec::remove(0) was O(lookahead).
    let mut window: VecDeque<u64> = VecDeque::with_capacity(lookahead + 64);

    let emit = |kmer: u64, n_q: &mut u64, n_f: &mut u64,
                    out: &mut BufWriter<io::StdoutLock>| {
        *n_q += 1;
        match idx.find(kmer) {
            Some(cnt) => {
                *n_f += 1;
                writeln!(out, "{}\t{cnt}", decode_kmer(kmer, k)).ok();
            }
            None => {
                if !only_found { writeln!(out, "{}\t0", decode_kmer(kmer, k)).ok(); }
            }
        }
    };

    iterate_fasta(query_path, |_name, seq| {
        extract_kmers(seq, k, |kmer| {
            if window.len() >= lookahead {
                idx.prefetch(kmer);
                let cur = window.pop_front().unwrap(); // O(1) — was O(lookahead)
                emit(cur, &mut n_queried, &mut n_found, &mut out);
            }
            window.push_back(kmer);
        });
        // Drain whatever is left at the end of this sequence. There is no
        // future kmer to prefetch against so we just resolve directly.
        for kmer in window.drain(..) { emit(kmer, &mut n_queried, &mut n_found, &mut out); }
    })?;

    let q_s = tq.elapsed().as_secs_f64();
    let ns_q = if n_queried > 0 { q_s * 1e9 / n_queried as f64 } else { 0.0 };
    let pct  = if n_queried > 0 { 100.0 * n_found as f64 / n_queried as f64 } else { 0.0 };
    eprintln!("[query] {n_queried} queries in {q_s:.3}s  {ns_q:.1} ns/q  \
               {n_found} found ({pct:.1}%)");
    Ok(())
}
