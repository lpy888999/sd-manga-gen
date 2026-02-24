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
# pip install --upgrade "transformers==4.44.2" "peft==0.12.0" "diffusers==0.30.3" "pydantic<2.0" "edge-tts" "pydub" "insightface>=0.7.3" "onnxruntime-gpu>=1.16.3" --quiet

echo "== Downloading IP-Adapter Plus Models =="
hf download h94/IP-Adapter sdxl_models/ip-adapter-plus-face_sdxl_vit-h.bin \
    --local-dir "/vol/bitbucket/jl10525/hf_cache/ip-adapter" \
    --local-dir-use-symlinks False --quiet

echo "== Downloading ControlNet Canny SDXL (for two-stage generation) =="
hf download diffusers/controlnet-canny-sdxl-1.0 \
    --local-dir "/vol/bitbucket/jl10525/hf_cache/controlnet-canny-sdxl" \
    --local-dir-use-symlinks False --quiet

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
# ─── Improved Wait Logic ───
echo "== Waiting for Ollama server to be ready... =="
MAX_RETRIES=12
COUNT=0
while ! $OLLAMA list > /dev/null 2>&1; do
    COUNT=$((COUNT + 1))
    if [ $COUNT -ge $MAX_RETRIES ]; then
        echo "ERROR: Ollama server failed to start after $MAX_RETRIES attempts."
        echo "Last 20 lines of ollama_server.log:"
        tail -n 20 "${OUTPUT_BASE}/ollama_server.log"
        kill $OLLAMA_PID 2>/dev/null
        exit 1
    fi
    sleep 5
    echo "  Attempt $COUNT/$MAX_RETRIES..."
done
echo "== Ollama server is ready! =="

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
    local CONFIG_FILE="$5"
    local PROMPT_CACHE="$6"

    local OUT_DIR="${OUTPUT_BASE}/story_${IDX}"
    local OUT_IMG="${OUT_DIR}/comic.png"
    local LOG_FILE="${OUT_DIR}/pipeline.log"
    local CONSOLE_LOG="${OUT_DIR}/console.log"
    mkdir -p "$OUT_DIR"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Story $IDX  |  ${PANELS} panels  |  seed=$SEED"
    echo "  Config: $CONFIG_FILE"
    echo "  Prompt: ${PROMPT:0:80}..."
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    python main.py \
        --config "$CONFIG_FILE" \
        --reference "tests/fixtures/luffy.jpg" \
        -p "$PROMPT" \
        --panels "$PANELS" \
        --seed "$SEED" \
        --audio \
        --audio-dir "${OUT_DIR}/audio" \
        ${PROMPT_CACHE:+--prompt-cache "$PROMPT_CACHE"} \
        -o "$OUT_IMG" \
        -v > "$CONSOLE_LOG"

    if [ -f "$OUT_IMG" ]; then
        echo "  ✅ Story $IDX complete → $OUT_IMG"
    else
        echo "  ❌ Story $IDX FAILED — check $LOG_FILE or $CONSOLE_LOG"
        echo "  Last 20 lines of console output:"
        tail -n 20 "$CONSOLE_LOG"
    fi
}

# ═══════════════════════════════════════════════════════════════
#  Male Protagonist IP-Adapter Plus (ViT-H) Test
# ═══════════════════════════════════════════════════════════════

MALE_PROMPT="A young man stands on a roof at night. There is a full moon glowing brightly behind him. A ninja in dark clothing jumps down towards him. He holds a sword. The young man and the ninja fight fiercely on the roof. The young man lands safely on the roof. The ninja was defeated."
SEED=42

# ─── Config Toggles ───
# Create temporary config files for the experiment
CONFIG_IP_ON="${OUTPUT_BASE}/config_ip_on.yaml"
CONFIG_IP_OFF="${OUTPUT_BASE}/config_ip_off.yaml"

cp config.yaml "$CONFIG_IP_ON"
cp config.yaml "$CONFIG_IP_OFF"

# Disable IP-Adapter in the "OFF" config using sed (macOS/Linux compatible)
sed -i.bak 's/enable: true/enable: false/g' "$CONFIG_IP_OFF" && rm -f "${CONFIG_IP_OFF}.bak"

# ─── Prompt Cache ───
CACHE_FILE="${OUTPUT_BASE}/prompts_cache.json"

# 1. Run with IP-Adapter ON (IP-Adapter Plus ViT-H)
run_story "1_ip_on" 4 "$MALE_PROMPT" "$SEED" "$CONFIG_IP_ON" "$CACHE_FILE"

# 2. Run with IP-Adapter OFF 
run_story "2_ip_off" 4 "$MALE_PROMPT" "$SEED" "$CONFIG_IP_OFF" "$CACHE_FILE"

# ═══════════════════════════════════════════════════════════════
#  Cleanup
# ═══════════════════════════════════════════════════════════════

echo ""
echo "═══════════════════════════════════════════════════"
echo "  All stories complete!"
echo "  Results:  $OUTPUT_BASE/"
echo "  Slurm log: slurm-${SLURM_JOB_ID}.out"
echo "═══════════════════════════════════════════════════"

# Stop Ollama server
kill $OLLAMA_PID 2>/dev/null || true

echo "== GPU Status (final) =="
nvidia-smi || echo "no nvidia-smi"
date