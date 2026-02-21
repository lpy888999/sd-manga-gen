#!/bin/bash
#SBATCH --job-name=manga-gen
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm-%j.out

# ═══════════════════════════════════════════════════════════════
#  SDXL Manga Generator — 5-Story Batch Run
# ═══════════════════════════════════════════════════════════════

set -euo pipefail

echo "== Node & Time =="
hostname
date

# ─── 1. Output Directory Check (Fail Fast) ─────────────────────
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_BASE="/vol/bitbucket/jl10525/output/batch_${TIMESTAMP}"

echo "== Checking Output Permissions =="
if ! mkdir -p "$OUTPUT_BASE"; then
    echo "❌ FATAL: Cannot create output directory: $OUTPUT_BASE"
    exit 1
fi
echo "✅ Output directory created: $OUTPUT_BASE"

# ─── 2. Environment Activation ─────────────────────────────────
echo "== Setting up Environment =="

CONDA_ROOT="/vol/bitbucket/jl10525/miniconda3"

# ✅ 修复：不再 source ~/.bashrc，直接加载 conda.sh
source "${CONDA_ROOT}/etc/profile.d/conda.sh"

# 激活环境
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
echo "   TTS cache: $TTS_HOME"
echo "   HF cache:  $HF_HOME"

# ─── Fix Python Environment ─────────────────────────────────────
# The server has an unusual transformers 5.1.0 which breaks peft/diffusers.
# Force-install stable, compatible versions for SDXL.
echo "== Standardizing ML dependencies (transformers, peft, diffusers) and TTS (edge-tts) =="
pip install --upgrade "transformers==4.44.2" "peft==0.12.0" "diffusers==0.30.3" "pydantic<2.0" "edge-tts" "pydub" --quiet

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    PROJECT_DIR="$SLURM_SUBMIT_DIR"
else
    PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
fi
cd "$PROJECT_DIR"
echo "   Project dir: $(pwd)"
# ─── 3. GPU & CUDA Check ───────────────────────────────────────
echo "== GPU Status =="
nvidia-smi || echo "no nvidia-smi"

echo "== CUDA Check =="
python - << 'PYEOF'
import torch
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device name:", torch.cuda.get_device_name(0))
PYEOF

# ─── 4. TTS Reference Audio ────────────────────────────────────
# Generate reference WAVs for XTTS voice cloning (only if not already present)
if [ ! -f "assets/tts_refs/narrator.wav" ]; then
    echo "== Generating TTS reference audio =="
    python scripts/fetch_tts_refs.py
else
    echo "== TTS reference audio already exists, skipping =="
fi

# ─── 5. Ollama Setup ───────────────────────────────────────────
OLLAMA=/homes/jl10525/bin/ollama
export OLLAMA_MODELS="/vol/bitbucket/jl10525/ollama_data"
mkdir -p "$OLLAMA_MODELS"
echo "   Ollama models: $OLLAMA_MODELS"

# ✅ 让 Ollama 使用 Slurm 分配的 GPU
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OLLAMA_NUM_GPU=999

export OLLAMA_KEEP_ALIVE=0
export OLLAMA_MAX_LOADED_MODELS=1

# 尝试自动定位 CUDA 库路径以解决 0 VRAM 问题 (Imperial Doc 集群常用路径)
# 常见的 CUDA 库路径
CUDA_PATHS="/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:/usr/lib64"
export LD_LIBRARY_PATH="${CUDA_PATHS}:${LD_LIBRARY_PATH:-}"

# Generate a random port between 11435 and 15000 to avoid conflicts
RANDOM_PORT=$((11435 + RANDOM % 3565))
export OLLAMA_HOST="127.0.0.1:${RANDOM_PORT}"
export OLLAMA_BASE_URL="http://127.0.0.1:${RANDOM_PORT}/v1/"

echo "== Starting Ollama server (Port: ${RANDOM_PORT}, GPU: ${CUDA_VISIBLE_DEVICES}) =="
# 显式使用环境变量运行并将日志重定向到文件以减少 Slurm 输出噪音
$OLLAMA serve > "${OUTPUT_BASE}/ollama_server.log" 2>&1 &
OLLAMA_PID=$!
sleep 15

$OLLAMA list || { echo "ERROR: Ollama failed to start"; kill $OLLAMA_PID 2>/dev/null; exit 1; }

echo "== Pulling LLM models =="
$OLLAMA pull qwen3:8b || echo "WARNING: failed to pull qwen3:8b"
$OLLAMA pull qwen3-vl:4b || echo "WARNING: failed to pull qwen3-vl:4b"

echo "== Verifying Ollama GPU Usage =="
nvidia-smi


echo ""
echo "═══════════════════════════════════════════════════"
echo "  Output directory: $OUTPUT_BASE"
echo "═══════════════════════════════════════════════════"
echo ""

# ─── Helper function ───────────────────────────────────────────
run_story() {
    local IDX="$1"
    local PANELS="$2"
    local PROMPT="$3"
    local SEED="$4"

    local OUT_DIR="${OUTPUT_BASE}/story_${IDX}"
    local OUT_IMG="${OUT_DIR}/comic.png"
    local LOG_FILE="${OUT_DIR}/pipeline.log"
    mkdir -p "$OUT_DIR"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Story $IDX / 5  |  ${PANELS} panels  |  seed=$SEED"
    echo "  Prompt: ${PROMPT:0:80}..."
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    python main.py \
        --reference "tests/fixtures/reference_character.png" \
        -p "$PROMPT" \
        --panels "$PANELS" \
        --seed "$SEED" \
        --audio \
        --audio-dir "${OUT_DIR}/audio" \
        -o "$OUT_IMG" \
        -v \
        2>&1 | tee "$LOG_FILE"

    if [ -f "$OUT_IMG" ]; then
        echo "  ✅ Story $IDX complete → $OUT_IMG"
    else
        echo "  ❌ Story $IDX FAILED — check $LOG_FILE"
    fi
}

# ═══════════════════════════════════════════════════════════════
#  5 Pre-designed Stories (Ninja/Shinobi Theme)
# ═══════════════════════════════════════════════════════════════

# Story 1 — Midnight Duel (Action) / 4 panels
run_story 1 4 \
    "The protagonist ninja stands on a tiled roof under a full moon. An enemy shinobi in grey armor suddenly drops from the shadows with a kunai. They clash mid-air, sparks flying. The protagonist lands gracefully while the enemy retreats into the darkness." \
    777

# Story 2 — Hidden Village Training (Daily/Training) / 6 panels
run_story 2 6 \
    "The protagonist ninja practices throwing shuriken at a wooden target in a forest clearing. A stern sensei with a scarred face watches from the sidelines. The protagonist misses the center and looks frustrated. The sensei walks over and gently corrects the protagonist's stance. They both throw at once, hitting the bullseye together. They share a small smile." \
    888

# Story 3 — Secret Rendezvous (Romance/Casual) / 4 panels
run_story 3 4 \
    "The protagonist ninja sits on a high bridge, looking at the sunset. A male companion in dark shinobi attire joins, carrying a small paper bag of steamed buns. They sit together, sharing the food. As they talk, the companion reaches out to brush a stray leaf from the protagonist's hair." \
    999

# Story 4 — Forest Infiltration (Adventure) / 6 panels
run_story 4 6 \
    "The protagonist ninja is crouching in high grass, spying on a heavily guarded gate. The partner, a large ninja with a giant scroll on his back, signals that the coast is clear. They move like ghosts through the trees. Suddenly, a guard dog barks in their direction. They freeze as a guard walks past their hiding spot. They successfully leap over the high wall together." \
    123

# Story 5 — The Mission Briefing (Mission/Serious) / 4 panels
run_story 5 4 \
    "Inside a dimly lit room, the protagonist ninja looks at a secret map unrolled on a low table. An elder ninja with white hair points to a hidden fortress marked in red. A younger recruit stands by the door, holding a flickering torch. The protagonist nods, preparing for the upcoming mission." \
    456

# ═══════════════════════════════════════════════════════════════
#  Cleanup
# ═══════════════════════════════════════════════════════════════

echo ""
echo "═══════════════════════════════════════════════════"
echo "  All 5 stories complete!"
echo "  Results:  $OUTPUT_BASE/"
echo "  Slurm log: slurm-${SLURM_JOB_ID}.out"
echo "═══════════════════════════════════════════════════"

# Stop Ollama server
kill $OLLAMA_PID 2>/dev/null || true

echo "== GPU Status (final) =="
nvidia-smi || echo "no nvidia-smi"
date