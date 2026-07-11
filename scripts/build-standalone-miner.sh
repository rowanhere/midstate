#!/usr/bin/env bash
set -euo pipefail

arch="${MIDSTATE_CUDA_ARCH:-sm_120}"
out="${1:-miner-linux-${arch}}"

if ! command -v nvcc >/dev/null 2>&1; then
  echo "error: nvcc not found. Build on a CUDA toolkit/devel machine, then copy the output miner file." >&2
  exit 1
fi

MIDSTATE_CUDA_EMBED=1 MIDSTATE_CUDA_ARCH="${arch}" cargo build --release --bin miner
cp target/release/miner "${out}"
chmod +x "${out}"

echo "Built ${out}"
echo "Runtime dependency check:"
if ldd "${out}" | grep -Ei 'nvrtc|cudart'; then
  echo "warning: unexpected CUDA runtime/compiler dependency shown above" >&2
else
  echo "OK: no libnvrtc/libcudart dependency. Target still needs NVIDIA driver/libcuda."
fi
