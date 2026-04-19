mod build;
mod query;
mod map;
mod align;
mod dbruijn;
mod bench;

use std::ffi::CString;
use std::io::{self, BufWriter, Write};
use std::os::raw::{c_char, c_int, c_void};
use std::time::Instant;

pub struct OkIndex        { _private: [u8; 0] }
pub struct OkOpenAddr     { _private: [u8; 0] }
pub struct OkUnorderedMap { _private: [u8; 0] }

#[allow(improper_ctypes)]
unsafe extern "C" {
    fn okidx_new(k: u64, n: u64) -> *mut OkIndex;
    fn okidx_free(h: *mut OkIndex);
    fn okidx_stage(h: *mut OkIndex, kmer: u64, val: u32);
    fn okidx_build(h: *mut OkIndex);
    fn okidx_find(h: *const OkIndex, kmer: u64) -> u32;
    fn okidx_prefetch(h: *const OkIndex, kmer: u64);
    fn okidx_size(h: *const OkIndex) -> u64;
    fn okidx_bytes(h: *const OkIndex) -> u64;
    fn okidx_bits_per_entry(h: *const OkIndex) -> f64;

    fn okopen_new(capacity: u64) -> *mut OkOpenAddr;
    fn okopen_free(h: *mut OkOpenAddr);
    fn okopen_insert(h: *mut OkOpenAddr, kmer: u64, val: u32);
    fn okopen_find(h: *const OkOpenAddr, kmer: u64) -> u32;
    fn okopen_prefetch(h: *const OkOpenAddr, kmer: u64);
    fn okopen_size(h: *const OkOpenAddr) -> u64;
    fn okopen_bytes(h: *const OkOpenAddr) -> u64;
    fn okopen_bits_per_entry(h: *const OkOpenAddr) -> f64;

    fn okumap_new(capacity: u64) -> *mut OkUnorderedMap;
    fn okumap_free(h: *mut OkUnorderedMap);
    fn okumap_insert(h: *mut OkUnorderedMap, kmer: u64, val: u32);
    fn okumap_find(h: *const OkUnorderedMap, kmer: u64) -> u32;
    fn okumap_size(h: *const OkUnorderedMap) -> u64;
    fn okumap_bytes(h: *const OkUnorderedMap) -> u64;
    fn okumap_bits_per_entry(h: *const OkUnorderedMap) -> f64;

    pub fn ok_canonical(kmer: u64, k: u64) -> u64;
    pub fn ok_kmer_mask(k: u64) -> u64;
    fn ok_iterate_fasta(
        path: *const c_char,
        cb: extern "C" fn(*mut c_void, *const c_char, u64, *const c_char, u64),
        cookie: *mut c_void,
        errbuf: *mut c_char,
        errbuf_len: u64,
    ) -> c_int;
    fn ok_extract_kmers(
        seq: *const c_char, len: u64, k: u64,
        cb: extern "C" fn(*mut c_void, u64),
        cookie: *mut c_void,
    );
    fn ok_extract_minimizers(
        seq: *const c_char, len: u64, k: u64, w: u64,
        cb: extern "C" fn(*mut c_void, u64, u32),
        cookie: *mut c_void,
    );
    fn ok_myers_align(
        query: *const c_char, qlen: c_int,
        text: *const c_char,  tlen: c_int,
        max_ed: c_int, end_out: *mut c_int,
    ) -> c_int;
}

// TinyPointerIndex with Robin Hood displacement — the full TP+OptOA implementation.
// All CLI commands use this; bench.rs also constructs OpenAddr and UnorderedMap
// as comparison targets.
pub struct Index { ptr: *mut OkIndex }
unsafe impl Send for Index {}
unsafe impl Sync for Index {}

impl Index {
    pub fn new(k: usize, n: usize) -> Self {
        let ptr = unsafe { okidx_new(k as u64, n as u64) };
        assert!(!ptr.is_null(), "okidx_new allocation failed");
        Index { ptr }
    }
    pub fn stage(&mut self, kmer: u64, val: u32) { unsafe { okidx_stage(self.ptr, kmer, val) } }
    pub fn build(&mut self)                       { unsafe { okidx_build(self.ptr) } }
    pub fn find(&self, kmer: u64) -> Option<u32> {
        let v = unsafe { okidx_find(self.ptr, kmer) };
        if v == u32::MAX { None } else { Some(v) }
    }
    pub fn prefetch(&self, kmer: u64)   { unsafe { okidx_prefetch(self.ptr, kmer) } }
    pub fn size(&self) -> usize         { unsafe { okidx_size(self.ptr) as usize } }
    pub fn bytes(&self) -> usize        { unsafe { okidx_bytes(self.ptr) as usize } }
    pub fn bits_per_entry(&self) -> f64 { unsafe { okidx_bits_per_entry(self.ptr) } }
}
impl Drop for Index { fn drop(&mut self) { unsafe { okidx_free(self.ptr) } } }

// ElasticHashTable with Robin Hood but without tiny pointers. Only used in
// bench.rs to isolate the space contribution of the tiny pointer layer.
pub struct OpenAddr { ptr: *mut OkOpenAddr }
unsafe impl Send for OpenAddr {}
unsafe impl Sync for OpenAddr {}

impl OpenAddr {
    pub fn new(capacity: usize) -> Self {
        let ptr = unsafe { okopen_new(capacity as u64) };
        assert!(!ptr.is_null());
        OpenAddr { ptr }
    }
    pub fn insert(&mut self, kmer: u64, val: u32) { unsafe { okopen_insert(self.ptr, kmer, val) } }
    pub fn find(&self, kmer: u64) -> Option<u32> {
        let v = unsafe { okopen_find(self.ptr, kmer) };
        if v == u32::MAX { None } else { Some(v) }
    }
    pub fn prefetch(&self, kmer: u64)   { unsafe { okopen_prefetch(self.ptr, kmer) } }
    pub fn size(&self) -> usize         { unsafe { okopen_size(self.ptr) as usize } }
    pub fn bytes(&self) -> usize        { unsafe { okopen_bytes(self.ptr) as usize } }
    pub fn bits_per_entry(&self) -> f64 { unsafe { okopen_bits_per_entry(self.ptr) } }
}
impl Drop for OpenAddr { fn drop(&mut self) { unsafe { okopen_free(self.ptr) } } }

// C++ std::unordered_map with chaining. No prefetch support; used in bench.rs
// as a worst-case pointer-chasing baseline.
pub struct UnorderedMap { ptr: *mut OkUnorderedMap }
unsafe impl Send for UnorderedMap {}
unsafe impl Sync for UnorderedMap {}

impl UnorderedMap {
    pub fn new(capacity: usize) -> Self {
        let ptr = unsafe { okumap_new(capacity as u64) };
        assert!(!ptr.is_null());
        UnorderedMap { ptr }
    }
    pub fn insert(&mut self, kmer: u64, val: u32) { unsafe { okumap_insert(self.ptr, kmer, val) } }
    pub fn find(&self, kmer: u64) -> Option<u32> {
        let v = unsafe { okumap_find(self.ptr, kmer) };
        if v == u32::MAX { None } else { Some(v) }
    }
    pub fn size(&self) -> usize         { unsafe { okumap_size(self.ptr) as usize } }
    pub fn bytes(&self) -> usize        { unsafe { okumap_bytes(self.ptr) as usize } }
    pub fn bits_per_entry(&self) -> f64 { unsafe { okumap_bits_per_entry(self.ptr) } }
}
impl Drop for UnorderedMap { fn drop(&mut self) { unsafe { okumap_free(self.ptr) } } }

// C requires a stable fn pointer, so we use a monomorphised extern "C" fn
// that recovers the concrete closure type by casting the cookie back.
pub fn iterate_fasta<F: FnMut(&str, &[u8])>(path: &str, mut f: F) -> Result<(), String> {
    extern "C" fn cb<F: FnMut(&str, &[u8])>(
        cookie: *mut c_void,
        name: *const c_char, nlen: u64,
        seq:  *const c_char, slen: u64,
    ) {
        let f = unsafe { &mut *(cookie as *mut F) };
        let name_s = unsafe {
            std::str::from_utf8_unchecked(std::slice::from_raw_parts(name as *const u8, nlen as usize))
        };
        let seq_b = unsafe { std::slice::from_raw_parts(seq as *const u8, slen as usize) };
        f(name_s, seq_b);
    }
    let cpath = CString::new(path).map_err(|e| e.to_string())?;
    let mut errbuf = vec![0i8; 512];
    let rc = unsafe {
        ok_iterate_fasta(cpath.as_ptr(), cb::<F>,
                         &mut f as *mut F as *mut c_void,
                         errbuf.as_mut_ptr(), errbuf.len() as u64)
    };
    if rc != 0 {
        let msg = unsafe {
            std::ffi::CStr::from_ptr(errbuf.as_ptr()).to_string_lossy().into_owned()
        };
        return Err(msg);
    }
    Ok(())
}

pub fn extract_kmers<F: FnMut(u64)>(seq: &[u8], k: usize, mut f: F) {
    extern "C" fn cb<F: FnMut(u64)>(cookie: *mut c_void, kmer: u64) {
        unsafe { (&mut *(cookie as *mut F))(kmer) }
    }
    unsafe {
        ok_extract_kmers(seq.as_ptr() as *const c_char, seq.len() as u64, k as u64,
                         cb::<F>, &mut f as *mut F as *mut c_void);
    }
}

pub fn extract_minimizers<F: FnMut(u64, u32)>(seq: &[u8], k: usize, w: usize, mut f: F) {
    extern "C" fn cb<F: FnMut(u64, u32)>(cookie: *mut c_void, kmer: u64, pos: u32) {
        unsafe { (&mut *(cookie as *mut F))(kmer, pos) }
    }
    unsafe {
        ok_extract_minimizers(seq.as_ptr() as *const c_char, seq.len() as u64,
                              k as u64, w as u64,
                              cb::<F>, &mut f as *mut F as *mut c_void);
    }
}

pub fn myers_align(query: &[u8], text: &[u8], max_ed: i32) -> (i32, i32) {
    let mut end: c_int = -1;
    let ed = unsafe {
        ok_myers_align(query.as_ptr() as *const c_char, query.len() as c_int,
                       text.as_ptr()  as *const c_char, text.len()  as c_int,
                       max_ed, &mut end)
    };
    (ed, end)
}

// Chosen to differ from the C++ binary magic (0x54504F50544F4100) so the two
// tools never silently cross-load each other's index files.
pub const MAGIC: u64 = 0x4F50_544B_4944_5800;

pub fn write_index(path: &str, k: usize, entries: &[(u64, u32)]) -> io::Result<()> {
    let mut f = BufWriter::new(std::fs::File::create(path)?);
    f.write_all(&MAGIC.to_le_bytes())?;
    f.write_all(&(k as u64).to_le_bytes())?;
    f.write_all(&(entries.len() as u64).to_le_bytes())?;
    for &(kmer, cnt) in entries {
        f.write_all(&kmer.to_le_bytes())?;
        f.write_all(&cnt.to_le_bytes())?;
    }
    Ok(())
}

pub fn read_index(path: &str) -> Result<(usize, Vec<(u64, u32)>), String> {
    let data = std::fs::read(path).map_err(|e| e.to_string())?;
    if data.len() < 24 { return Err("index file too short".into()); }
    let magic = u64::from_le_bytes(data[0..8].try_into().unwrap());
    if magic != MAGIC { return Err("bad magic — rebuild with: optik build".into()); }
    let k = u64::from_le_bytes(data[8..16].try_into().unwrap()) as usize;
    let n = u64::from_le_bytes(data[16..24].try_into().unwrap()) as usize;
    let mut entries = Vec::with_capacity(n);
    for i in 0..n {
        let off = 24 + i * 12;
        if off + 12 > data.len() { break; }
        let kmer = u64::from_le_bytes(data[off..off+8].try_into().unwrap());
        let cnt  = u32::from_le_bytes(data[off+8..off+12].try_into().unwrap());
        entries.push((kmer, cnt));
    }
    Ok((k, entries))
}

pub fn entries_to_index(k: usize, entries: &[(u64, u32)]) -> Index {
    let mut idx = Index::new(k, entries.len());
    for &(kmer, cnt) in entries { idx.stage(kmer, cnt); }
    idx.build();
    idx
}

pub fn decode_kmer(mut kmer: u64, k: usize) -> String {
    const BASES: [char; 4] = ['A', 'C', 'G', 'T'];
    let mut s = vec!['N'; k];
    for i in (0..k).rev() { s[i] = BASES[(kmer & 3) as usize]; kmer >>= 2; }
    s.into_iter().collect()
}

pub fn mean(v: &[f64]) -> f64 { v.iter().sum::<f64>() / v.len() as f64 }

pub fn stddev(v: &[f64]) -> f64 {
    if v.len() < 2 { return 0.0; }
    let m = mean(v);
    (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (v.len() - 1) as f64).sqrt()
}

// When the index fits in LLC, a long prefetch pipeline is cheap and hides
// latency well. Once the index exceeds LLC capacity every access goes to DRAM
// (~100 ns), and a longer pipeline just wastes reorder-buffer slots without
// covering additional latency, so we scale down. The crossover is around 4 M
// entries on most current servers.
pub const LOOKAHEAD_MIN: usize = 4;
pub const LOOKAHEAD_MAX: usize = 24;

pub fn dynamic_lookahead(file_bytes: u64) -> usize {
    let file_mb = (file_bytes as f64 / (1024.0 * 1024.0)).max(0.001);
    let l = 40.84 / file_mb + 7.83;
    (l.round() as usize).clamp(LOOKAHEAD_MIN, LOOKAHEAD_MAX)
}

pub struct RssWatermark { before_kb: u64, peak_kb: u64 }
impl RssWatermark {
    pub fn new() -> Self {
        let r = Self::read_kb();
        RssWatermark { before_kb: r, peak_kb: r }
    }
    pub fn sample(&mut self) {
        let r = Self::read_kb();
        if r > self.peak_kb { self.peak_kb = r; }
    }
    pub fn net_mb(&self) -> f64 {
        self.peak_kb.saturating_sub(self.before_kb) as f64 / 1024.0
    }
    fn read_kb() -> u64 {
        std::fs::read_to_string("/proc/self/status")
            .unwrap_or_default()
            .lines()
            .find(|l| l.starts_with("VmRSS:"))
            .and_then(|l| l.split_whitespace().nth(1))
            .and_then(|v| v.parse().ok())
            .unwrap_or(0)
    }
}

// Chases a randomly permuted linked list through a buffer larger than L3 so
// that every step misses the cache and hits DRAM. The average step time is
// therefore DRAM latency. As a side effect the sweep forces any pending DRAM
// row refreshes to complete, which stabilises subsequent timed measurements.
pub fn dram_latency_ns(buffer_bytes: usize) -> f64 {
    const LINE: usize = 64 / 8;
    let n = (buffer_bytes / 64).max(1024);
    let mut buf: Vec<u64> = vec![0u64; n * LINE];
    for i in 0..n { buf[i * LINE] = i as u64; }
    let mut rng = 42u64;
    for i in (1..n).rev() {
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
        let j = (rng as usize) % (i + 1);
        buf.swap(i * LINE, j * LINE);
    }
    let mut cur = 0usize;
    for _ in 0..n { cur = buf[cur * LINE] as usize; }
    std::hint::black_box(cur);
    let t0 = Instant::now();
    let mut cur = 0usize;
    for _ in 0..4 { for _ in 0..n { cur = buf[cur * LINE] as usize; } }
    std::hint::black_box(cur);
    t0.elapsed().as_nanos() as f64 / (n * 4) as f64
}

// Interpolates between a 4 ns L1-hit floor and the measured DRAM ceiling to
// estimate what fraction of queries caused an LLC miss.
pub fn estimate_llc_misses_per_1k(query_ns: f64, dram_ns: f64) -> f64 {
    const L1_NS: f64 = 4.0;
    if dram_ns <= L1_NS { return 0.0; }
    ((query_ns - L1_NS) / (dram_ns - L1_NS)).clamp(0.0, 1.0) * 1000.0
}

fn usage() {
    eprintln!(
"optik — genomic sequence analysis on the tpoptoa index

USAGE
  optik build   -i <fasta/fastq>  [-k INT]           -o <index.bin>
  optik query   -x <index.bin>    -q <fasta/fastq>   [-a]
  optik map     -r <ref.fa>       -q <reads.fa/fq>   [-k INT] [-w INT] [-e INT]
  optik align   -q <query.fa>     -r <ref.fa>        [-e INT]
  optik dbruijn -i <fasta/fastq>  [-k INT] [-c INT]  -o <graph.gfa>
  optik bench   -i <fasta/fastq>  [-k INT] [-n INT] [-o TSV]

FLAGS
  -k  k-mer length          (default 31)
  -w  minimizer window size (default 10, map only)
  -e  max edit distance     (default 5,  map/align)
  -n  benchmark trials      (default 5)
  -c  min k-mer count       (default 2,  dbruijn)
  -a  only-found mode       (query: suppress absent k-mers)
  -o  output path
");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 { usage(); std::process::exit(1); }

    let sub  = &args[1];
    let rest = &args[2..];

    let get = |flag: &str| -> Option<&str> {
        rest.windows(2).find(|w| w[0] == flag).map(|w| w[1].as_str())
    };
    let has = |flag: &str| -> bool { rest.iter().any(|a| a == flag) };

    let k: usize = get("-k").and_then(|v| v.parse().ok()).unwrap_or(31);
    let w: usize = get("-w").and_then(|v| v.parse().ok()).unwrap_or(10);
    let e: i32   = get("-e").and_then(|v| v.parse().ok()).unwrap_or(5);
    let n: usize = get("-n").and_then(|v| v.parse().ok()).unwrap_or(5);
    let c: u32   = get("-c").and_then(|v| v.parse().ok()).unwrap_or(2);

    let result = match sub.as_str() {
        "build"   => build::run(get("-i").expect("-i required"), k,
                                get("-o").unwrap_or("index.bin")),
        "query"   => query::run(get("-x").expect("-x required"),
                                get("-q").expect("-q required"), has("-a")),
        "map"     => map::run(get("-r").expect("-r required"),
                              get("-q").expect("-q required"), k, w, e),
        "align"   => align::run(get("-q").expect("-q required"),
                                get("-r").expect("-r required"), e),
        "dbruijn" => dbruijn::run(get("-i").expect("-i required"), k, c,
                                  get("-o").unwrap_or("graph.gfa")),
        "bench"   => bench::run(get("-i").expect("-i required"), k, n,
                                get("-o").unwrap_or("benchmark.tsv")),
        _ => { usage(); std::process::exit(1); }
    };

    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
