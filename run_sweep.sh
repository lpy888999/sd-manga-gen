#!/bin/bash
#SBATCH --job-name=manga-sweep
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm-sweep-%j.out

# ═══════════════════════════════════════════════════════════════
#  Parameter Matrix Sweep (Two-Stage SD Generation)
# ═══════════════════════════════════════════════════════════════

set -euo pipefail

echo "== Node & Time =="
hostname
date

export PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
cd "$PROJECT_ROOT"

# ─── Environment Activation ───
echo "== Setting up Environment =="
CONDA_ROOT="/vol/bitbucket/jl10525/miniconda3"
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate manga
echo "✅ Conda environment 'manga' activated"

# ─── Redirect model caches ───
export TTS_HOME="/vol/bitbucket/jl10525/tts_cache"
export LIBROSA_CACHE_DIR="/vol/bitbucket/jl10525/librosa_cache"
export HF_HOME="/vol/bitbucket/jl10525/hf_cache"
export HF_HUB_CACHE="/vol/bitbucket/jl10525/hf_cache/hub"
export HF_ASSETS_CACHE="/vol/bitbucket/jl10525/hf_cache/assets"
export PIP_CACHE_DIR="/vol/bitbucket/jl10525/pip_cache"
export TMPDIR="/vol/bitbucket/jl10525/tmp"
export XDG_CACHE_HOME="/vol/bitbucket/jl10525/xdg_cache"
export COQUI_TOS_AGREED=1

echo "== Checking Face Restoration Cascade =="
CASCADE_PATH="/vol/bitbucket/jl10525/lbpcascade_animeface.xml"
if [ ! -f "$CASCADE_PATH" ]; then
    wget -q https://raw.githubusercontent.com/nagadomi/lbpcascade_animeface/master/lbpcascade_animeface.xml -O "$CASCADE_PATH"
fi

echo "== Starting Parameter Matrix Execution =="

# Execute the python sweeping script
python scripts/run_param_sweep.py

echo ""
echo "═══════════════════════════════════════════════════"
echo "  Sweep complete!"
echo "  Check output/sweep/parameter_sweep_matrix_with_restoration.jpg"
echo "═══════════════════════════════════════════════════"
date
