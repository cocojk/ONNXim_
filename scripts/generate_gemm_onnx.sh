#!/usr/bin/env bash

# generate_gemm_onnx.sh
# Generate a single-op GEMM ONNX of size (M,K,N) and its model list, without running the simulator.
# Output paths follow ONNXim's convention:
#   models/matmul_M_K_N/matmul_M_K_N.onnx
#   model_lists/matmul_M_K_N.json
#
# Usage:
#   ./scripts/generate_gemm_onnx.sh --M 512 --K 1024 --N 256 [--dtype fp16|fp32] [--onnxim_home /workspace/ONNXim]
#
# Example follow-up run:
#   ./build/bin/Simulator --config configs/isca_configs/base/sHBM.json \
#     --models_list model_lists/matmul_512_1024_256.json --log_level info \
#     > configs/isca_configs/log/gemm_HBM4_512_1024_256.log 2>&1

set -euo pipefail

# Defaults
M=512
K=1024
N=256
DTYPE="fp16"   # or fp32
ONNXIM_HOME="${ONNXIM_HOME:-$(cd "$(dirname "$0")/.." && pwd)}"

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --M) M="$2"; shift 2;;
    --K) K="$2"; shift 2;;
    --N) N="$2"; shift 2;;
    --dtype) DTYPE="$2"; shift 2;;
    --onnxim_home) ONNXIM_HOME="$2"; shift 2;;
    -h|--help)
      sed -n '1,40p' "$0" | sed 's/^# \{0,1\}//'; exit 0;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

MODELS_DIR="$ONNXIM_HOME/models"
MODEL_LISTS_DIR="$ONNXIM_HOME/model_lists"
MODEL_NAME="matmul_${M}_${K}_${N}"
MODEL_FOLDER="$MODELS_DIR/$MODEL_NAME"

# Map dtype
PYTORCH_DTYPE="torch.float16"
if [[ "$DTYPE" == "fp32" ]]; then PYTORCH_DTYPE="torch.float32"; fi

mkdir -p "$MODEL_FOLDER" "$MODEL_LISTS_DIR"

python3 - <<PY
import os, json, torch
from pathlib import Path

M, K, N = int(${M}), int(${K}), int(${N})
dtype = ${PYTORCH_DTYPE}
home = Path("${ONNXIM_HOME}")

class OneGemm(torch.nn.Module):
    def __init__(self, k, n, dtype):
        super().__init__()
        self.fc = torch.nn.Linear(k, n, bias=False, dtype=dtype)
    def forward(self, x):
        return self.fc(x)

name = f"matmul_{M}_{K}_{N}"
model_dir = home / "models" / name
model_dir.mkdir(parents=True, exist_ok=True)
onnx_path = model_dir / f"{name}.onnx"

m = OneGemm(K, N, dtype)
A = torch.zeros([M, K], dtype=dtype)
torch.onnx.export(
    m, A, onnx_path, export_params=True,
    input_names=['input'], output_names=['output'],
    opset_version=17
)

ml_dir = home / "model_lists"
ml_dir.mkdir(parents=True, exist_ok=True)
ml_path = ml_dir / f"{name}.json"
with ml_path.open("w") as f:
    json.dump({"models":[{"name":name, "request_time":0}]}, f, indent=2)

print(f"Wrote ONNX: {onnx_path}")
print(f"Wrote model list: {ml_path}")
PY

echo "Done. You can now run, for example:"
echo "  ./build/bin/Simulator --config configs/isca_configs/base/sHBM.json \\
    --models_list model_lists/${MODEL_NAME}.json --log_level info \\
    > configs/isca_configs/log/gemm_HBM4_${M}_${K}_${N}.log 2>&1"


