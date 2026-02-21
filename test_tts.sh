#!/bin/bash
#SBATCH --job-name=test-tts
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=tts_test-%j.out

set -euo pipefail

echo "== Node & Time =="
hostname
date

echo "== Setting up Environment =="
CONDA_ROOT="/vol/bitbucket/jl10525/miniconda3"
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate manga

echo "✅ Conda environment 'manga' activated"
echo "   Python: $(which python)"
echo "   Conda:  $CONDA_DEFAULT_ENV"

# ─── Redirect model caches to /vol/bitbucket (avoid HOME quota) ──
export TTS_HOME="/vol/bitbucket/jl10525/tts_cache"
export LIBROSA_CACHE_DIR="/vol/bitbucket/jl10525/librosa_cache"
export HF_HOME="/vol/bitbucket/jl10525/hf_cache"
export PIP_CACHE_DIR="/vol/bitbucket/jl10525/pip_cache"
export COQUI_TOS_AGREED=1
mkdir -p "$TTS_HOME" "$LIBROSA_CACHE_DIR" "$HF_HOME" "$PIP_CACHE_DIR"

# ─── GPU & CUDA Environment ───────────────────────────────────────
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
CUDA_PATHS="/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:/usr/lib64"
export LD_LIBRARY_PATH="${CUDA_PATHS}:${LD_LIBRARY_PATH:-}"

echo "== Standardizing ML dependencies (transformers, peft, diffusers) and TTS (edge-tts) =="
pip install --upgrade "transformers==4.44.2" "peft==0.12.0" "diffusers==0.30.3" "pydantic<2.0" "edge-tts" "pydub" --quiet

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    PROJECT_DIR="$SLURM_SUBMIT_DIR"
else
    PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
fi
cd "$PROJECT_DIR"
echo "   Project dir: $(pwd)"

echo "== GPU Status =="
nvidia-smi || echo "no nvidia-smi"

echo "== Running TTS Test =="
python test_tts.py --text "Storms birth power. This is a test of the Coqui VITS phoneme generation. Let's see if the noise is gone!" --out "debug_clean.wav"
echo "✅ Test complete!"
