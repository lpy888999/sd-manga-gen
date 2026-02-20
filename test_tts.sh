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
export LD_LIBRARY_PATH="/vol/bitbucket/jl10525/miniconda3/envs/manga/lib:$LD_LIBRARY_PATH"

echo "== Running TTS Test =="
python test_tts.py --text "Storms birth power. This is a test of the Coqui VITS phoneme generation. Let's see if the noise is gone!" --out "debug_clean.wav"
echo "✅ Test complete!"
