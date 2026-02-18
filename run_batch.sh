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

# ─── 4. Ollama Setup ───────────────────────────────────────────
OLLAMA=/homes/jl10525/bin/ollama
export OLLAMA_MODELS=$HOME/ollama_data

echo "== Starting Ollama server =="
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

# Story 1 — Action / 4 panels
run_story 1 4 \
    "A lone samurai stands guard on a rain-soaked rooftop. A massive combat mech crashes through the street below. The samurai leaps off the building, blade drawn, slicing through the rain. He lands on the mech's shoulder and drives his katana into its core." \
    42

# Story 2 — Fantasy / 6 panels
run_story 2 6 \
    "A young sorceress arrives at the entrance of an ancient crystal cave deep in a mystical forest. She discovers a sealed stone door covered in glowing runes and uses her staff to unlock it. Inside she finds a vast underground lake reflecting thousands of crystal stalactites. A massive water dragon emerges from the lake and roars, sending waves crashing. The sorceress raises her staff and casts a barrier of light, taming the dragon. She rides the dragon out of the cave as the sun rises over the forest." \
    101

# Story 3 — Cyberpunk / 4 panels
run_story 3 4 \
    "A hacker sits in a neon-lit underground den surrounded by holographic screens. He jacks into a corporate mainframe, his cybernetic eye flickering with data streams. Alarms blare as security drones swarm the corridor outside. He smashes through a window and escapes on a hoverbike into the rain-drenched city." \
    77

# Story 4 — Horror / 6 panels
run_story 4 6 \
    "A girl enters an abandoned hospital at midnight, her flashlight cutting through the dusty darkness. She finds old patient records scattered on the floor with strange symbols drawn in blood. A shadow moves at the end of the hallway, and she freezes. She follows the shadow into an operating room where all the surgical tools are arranged in a perfect circle. The lights flicker and a ghostly figure appears behind her in the reflection of a cracked mirror. She screams and runs toward the exit as the entire building shakes." \
    256

# Story 5 — Sci-Fi / 4 panels
run_story 5 4 \
    "An astronaut floats through the wreckage of a destroyed space station, debris and sparks drifting in zero gravity. He spots an escape pod still intact, glowing faintly through the twisted metal. He pushes off a wall fragment and glides toward the pod, dodging a spinning piece of hull. He seals the pod door and launches into the stars as the station explodes behind him." \
    512

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