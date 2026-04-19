use std::io::{self, BufWriter, Write};
use crate::{iterate_fasta, myers_align};

pub fn run(query_path: &str, ref_path: &str, max_ed: i32) -> Result<(), String> {
    let mut queries: Vec<(String, Vec<u8>)> = Vec::new();
    iterate_fasta(query_path, |n, s| { queries.push((n.to_string(), s.to_vec())); })?;

    let mut refs: Vec<(String, Vec<u8>)> = Vec::new();
    iterate_fasta(ref_path, |n, s| { refs.push((n.to_string(), s.to_vec())); })?;

    let stdout = io::stdout();
    let mut out = BufWriter::new(stdout.lock());
    writeln!(out, "query\tref\tedit_dist").unwrap();

    for (qn, qs) in &queries {
        for (rn, rs) in &refs {
            let (ed, _) = myers_align(qs, rs, max_ed);
            writeln!(out, "{qn}\t{rn}\t{ed}").unwrap();
        }
    }
    Ok(())
}
