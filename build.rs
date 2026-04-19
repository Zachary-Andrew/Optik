fn main() {
    // Read the actual CPU flags from /proc/cpuinfo once and derive which
    // SIMD tiers to enable.  The C++ code guards every intrinsic behind the
    // corresponding preprocessor macro, so the scalar fallback is always
    // compiled in and the binary is correct on any x86-64 CPU.
    let flags = cpu_flags();

    let avx512f  = flags.contains("avx512f");
    let avx512bw = flags.contains("avx512bw");
    let avx512dq = flags.contains("avx512dq");
    let avx2     = flags.contains("avx2");

    let mut build = cc::Build::new();
    build
        .cpp(true)
        .file("src/ffi.cpp")
        .include("include")
        .flag("-std=c++17")
        .flag("-O3")
        .flag("-march=native")   // lets the compiler use all safe scalar opts
        .flag("-funroll-loops")
        .flag("-ffast-math")
        .flag("-Wall");

    // Enable SIMD tiers only when the host CPU actually supports them.
    // The C++ headers check __AVX512F__ / __AVX2__ etc. before using any
    // intrinsic, so adding a flag here is both necessary and sufficient.
    if avx512f && avx512bw && avx512dq {
        build
            .flag("-mavx512f")
            .flag("-mavx512bw")
            .flag("-mavx512dq");
        println!("cargo:warning=optik: AVX-512 (F+BW+DQ) enabled");
    } else if avx512f && avx512bw {
        build
            .flag("-mavx512f")
            .flag("-mavx512bw");
        println!("cargo:warning=optik: AVX-512 (F+BW) enabled, no DQ");
    } else if avx2 {
        build.flag("-mavx2");
        println!("cargo:warning=optik: AVX2 enabled");
    } else {
        println!("cargo:warning=optik: no SIMD extensions, using scalar path");
    }

    build.compile("optik_native");
}

fn cpu_flags() -> String {
    // /proc/cpuinfo is Linux-only.  On other platforms we return empty string
    // so every SIMD tier stays disabled and the scalar path is used.
    std::fs::read_to_string("/proc/cpuinfo").unwrap_or_default()
}
