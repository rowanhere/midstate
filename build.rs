use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=src/core/cuda/midstate_kernel.cu");
    println!("cargo:rerun-if-env-changed=MIDSTATE_CUDA_EMBED");
    println!("cargo:rerun-if-env-changed=MIDSTATE_CUDA_ARCH");
    println!("cargo:rerun-if-env-changed=NVCC");
    println!("cargo:rustc-check-cfg=cfg(midstate_cuda_embedded)");

    if env::var("MIDSTATE_CUDA_EMBED").ok().as_deref() != Some("1") {
        return;
    }

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let source = manifest_dir.join("src/core/cuda/midstate_kernel.cu");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let arch = env::var("MIDSTATE_CUDA_ARCH").unwrap_or_else(|_| "sm_120".to_string());
    let fatbin = out_dir.join(format!("midstate_{arch}.fatbin"));
    let nvcc = env::var("NVCC").unwrap_or_else(|_| "nvcc".to_string());

    let status = Command::new(&nvcc)
        .arg("--fatbin")
        .arg("-O3")
        .arg("--std=c++11")
        .arg(format!("-arch={arch}"))
        .arg(&source)
        .arg("-o")
        .arg(&fatbin)
        .status()
        .unwrap_or_else(|e| panic!("failed to run {nvcc}: {e}"));

    if !status.success() {
        panic!("nvcc failed while building embedded CUDA fatbin for {arch}");
    }

    println!("cargo:rustc-cfg=midstate_cuda_embedded");
    println!("cargo:rustc-env=MIDSTATE_CUDA_FATBIN={}", fatbin.display());
}
