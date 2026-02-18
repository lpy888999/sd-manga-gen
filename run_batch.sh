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

# Generate a random port between 11435 and 15000 to avoid conflicts
RANDOM_PORT=$((11435 + RANDOM % 3565))
export OLLAMA_HOST="127.0.0.1:${RANDOM_PORT}"
export OLLAMA_BASE_URL="http://127.0.0.1:${RANDOM_PORT}/v1/"

echo "== Starting Ollama server (Port: ${RANDOM_PORT}) =="
$OLLAMA serve &
OLLAMA_PID=$!
sleep 5

$OLLAMA list || { echo "ERROR: Ollama failed to start"; kill $OLLAMA_PID 2>/dev/null; exit 1; }

echo "== Pulling LLM models =="
$OLLAMA pull qwen3.5:cloud || echo "WARNING: failed to pull qwen3.5:cloud"
$OLLAMA pull gemma3:12b-cloud || echo "WARNING: failed to pull gemma3:12b-cloud"


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
        --reference "tests/fixtures/meining.jpg" \
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
#  5 Pre-designed Stories
# ═══════════════════════════════════════════════════════════════

# Story 1 — Xianxia (Cultivation) / 4 panels
run_story 1 4 \
    "A female cultivator in elegant hanfu stands atop a misty peak, her long sleeves fluttering in the wind. She performs a hand seal, and a glowing green flying sword unsheathes from her back. She leaps onto the sword and soars through a sea of clouds towards a floating celestial palace. Thousands of golden cranes fly alongside her as she reaches the palace gate." \
    777

# Story 2 — Palace Intrigue / 6 panels
run_story 2 6 \
    "An elegant noble lady walks through a quiet imperial garden at sunset, holding a delicate silk fan. She notices a hidden scroll tucked behind a decorative rock and carefully retrieves it. In the flickering candlelight of her chamber, she unrolls the scroll to reveal a secret map of the palace. Suddenly, a shadow falls across the paper, and she quickly hides it under her embroidery. She turns to face a mysterious guard standing in the doorway, her expression calm but guarded. They exchange a silent, meaningful look as the palace bells toll in the distance." \
    888

# Story 3 — Wuxia (Bamboo Forest Duel) / 4 panels
run_story 3 4 \
    "A female warrior in red and white martial robes stands motionless in the center of a dense bamboo forest. A rain of leaves falls as an unseen assassin strikes, and she parries the blade with a silver flute. She kicks off a bamboo stalk, performing a graceful mid-air spin, her silk ribbons trailing behind. With a swift strike of her concealed dagger, she disarms the assassin, who disappears back into the emerald shadows." \
    999

# Story 4 — Mythology (The Phoenix) / 6 panels
run_story 4 6 \
    "A maiden travels to the edge of a volcanic crater under a blood-red moon to find the legendary Fire Phoenix. She begins to play a soul-stirring melody on her guzheng, the notes echoing through the rocky canyons. The magma below begins to glow intensely, and a magnificent phoenix made of pure flame emerges, wings outstretched. The maiden stands her ground, unaffected by the heat, as the phoenix bows its head to her. The creature transforms into a glowing feather, which she carefully tucks into her hair as a protective amulet. She walks away as the first light of dawn touches the mountain peaks." \
    123

# Story 5 — Poetic (Moonlit Zither) / 4 panels
run_story 5 4 \
    "Under a blooming peach tree by a moonlit lake, a young woman in light blue silk plays the zither. Soft blossoms fall onto the water's surface as her melody calms the ripples. An old friend arrives on a small wooden boat, carrying a jar of osmanthus wine and two jade cups. They sit together in silence, watching the reflection of the silver moon as the night breeze carries the scent of flowers." \
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