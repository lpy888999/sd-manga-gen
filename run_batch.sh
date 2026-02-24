#!/bin/bash
#SBATCH --job-name=manga-gen
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm-%j.out

# ═══════════════════════════════════════════════════════════════
#  SDXL Manga Generator — 10-Story Batch Run
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
export HF_HUB_CACHE="/vol/bitbucket/jl10525/hf_cache/hub"
export HF_ASSETS_CACHE="/vol/bitbucket/jl10525/hf_cache/assets"
export PIP_CACHE_DIR="/vol/bitbucket/jl10525/pip_cache"
export TMPDIR="/vol/bitbucket/jl10525/tmp"
export XDG_CACHE_HOME="/vol/bitbucket/jl10525/xdg_cache"
export COQUI_TOS_AGREED=1
mkdir -p "$TTS_HOME" "$LIBROSA_CACHE_DIR" "$HF_HOME" "$HF_HUB_CACHE" "$HF_ASSETS_CACHE" "$PIP_CACHE_DIR" "$TMPDIR" "$XDG_CACHE_HOME"
echo "   TTS cache: $TTS_HOME"
echo "   HF cache:  $HF_HOME"

# ─── Fix Python Environment ─────────────────────────────────────
# The server has an unusual transformers 5.1.0 which breaks peft/diffusers.
# Force-install stable, compatible versions for SDXL.
echo "== Standardizing ML dependencies (transformers, peft, diffusers) and TTS (edge-tts) =="
pip install "numpy<2.0.0" "pydantic<2.0.0" --quiet

echo "== Downloading IP-Adapter Plus Models =="
hf download h94/IP-Adapter sdxl_models/ip-adapter-plus-face_sdxl_vit-h.bin \
    --local-dir "/vol/bitbucket/jl10525/hf_cache/ip-adapter" \
    --quiet
echo "== Downloading Face Restoration Cascade =="
CASCADE_PATH="/vol/bitbucket/jl10525/lbpcascade_animeface.xml"
if [ ! -f "$CASCADE_PATH" ]; then
    wget -q https://raw.githubusercontent.com/nagadomi/lbpcascade_animeface/master/lbpcascade_animeface.xml -O "$CASCADE_PATH"
fi

echo "== Downloading ControlNet Canny SDXL (for two-stage generation) =="
hf download diffusers/controlnet-canny-sdxl-1.0 \
    --local-dir "/vol/bitbucket/jl10525/hf_cache/controlnet-canny-sdxl" \
    --quiet

# Ensure opencv-python is available for Canny edge extraction
echo "== Checking opencv-python =="
python -c "import cv2" 2>/dev/null || pip install opencv-python-headless --no-cache-dir --quiet && echo "   cv2 ready"

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
    local CONSOLE_LOG="${OUT_DIR}/console.log"
    mkdir -p "$OUT_DIR"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Story $IDX / 10  |  ${PANELS} panels  |  seed=$SEED"
    echo "  Prompt: ${PROMPT:0:80}..."
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    python main.py \
        --reference "tests/fixtures/luffy.jpg" \
        -p "$PROMPT" \
        --panels "$PANELS" \
        --seed "$SEED" \
        --audio \
        --audio-dir "${OUT_DIR}/audio" \
        -o "$OUT_IMG" \
        -v > "$CONSOLE_LOG"

    if [ -f "$OUT_IMG" ]; then
        echo "  ✅ Story $IDX complete → $OUT_IMG"
    else
        echo "  ❌ Story $IDX FAILED — check $LOG_FILE or $CONSOLE_LOG"
    fi
}

# ═══════════════════════════════════════════════════════════════
# Story 1 — 4 panels (Eating Meat)
run_story 1 4 \
    "A young pirate boy in a straw hat sits on a barrel holding a giant piece of meat. He takes a huge bite. He wipes his mouth, smiling brightly. He raises the meat high, cheering under the sunny sky." \
    111

# Story 2 — 4 panels (Finding Treasure)
run_story 2 4 \
    "A young pirate boy in a straw hat walks on a sandy beach. He spots a shiny golden box half buried. He digs it out with his bare hands. He opens the box to see glowing coins. He laughs loudly with his hands on his hips." \
    222

# Story 3 — 4 panels (Nap Interrupted)
run_story 3 4 \
    "A young pirate boy in a straw hat sleeps under a palm tree with a snot bubble. A crab pinches his toe. He wakes up yelling with his eyes wide. He angrily chases the crab down the beach." \
    333

# Story 4 — 6 panels (Practice Punch)
run_story 4 6 \
    "A young pirate boy in a straw hat stands in a fighting stance holding his fists up. He glares at a wooden training dummy. He steps forward. He throws a powerful punch. The dummy splinters into pieces. He smiles proudly, blowing to cool his fist." \
    444

# Story 5 — 4 panels (Stowaway)
run_story 5 4 \
    "A young pirate boy in a straw hat sneaks onto a large merchant ship at night. He hides behind some crates. A guard walks past him with a lantern. The boy quietly tiptoes towards the kitchen. He opens the door with a mischievous grin." \
    555

# Story 6 — 4 panels (Stormy Sea)
run_story 6 4 \
    "A young pirate boy in a straw hat stands at the bow of a wooden ship. Dark storm clouds cover the sky and rain pours. He holds firmly onto his hat. He points forward fearlessly toward the giant waves." \
    666

# Story 7 — 4 panels (Fishing Trouble)
run_story 7 4 \
    "A young pirate boy in a straw hat sits on the edge of a dock holding a fishing rod. The line suddenly tugs hard. He pulls back with all his strength, gritting his teeth. A giant angry fish flies out of the water. He drops the rod and runs away in panic." \
    777

# Story 8 — 4 panels (Sunset Promise)
run_story 8 4 \
    "A young pirate boy in a straw hat sits at the edge of a high cliff looking at the sunset. He stands up, looking serious. He raises his right fist into the air. He shouts his dream to the colorful sky." \
    888

# Story 9 — 6 panels (Apple Thief)
run_story 9 6 \
    "A young pirate boy in a straw hat walks through a town market. He sees a cart full of red apples. He quickly grabs one when the seller looks away. The seller turns around angrily. The boy runs fast down the street. He eats the apple while running and laughing." \
    999

# Story 10 — 4 panels (New Journey)
run_story 10 4 \
    "A young pirate boy in a straw hat unties a small wooden rowboat from a dock. He hops into the boat, picking up the oars. The sun rises over the calm ocean behind him. He rows excitedly toward the horizon." \
    1010

# ═══════════════════════════════════════════════════════════════
#  Cleanup
# ═══════════════════════════════════════════════════════════════

echo ""
echo "═══════════════════════════════════════════════════"
echo "  All 10 stories complete!"
echo "  Results:  $OUTPUT_BASE/"
echo "  Slurm log: slurm-${SLURM_JOB_ID}.out"
echo "═══════════════════════════════════════════════════"

# Stop Ollama server
kill $OLLAMA_PID 2>/dev/null || true

echo "== GPU Status (final) =="
nvidia-smi || echo "no nvidia-smi"
date