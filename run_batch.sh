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

# ─── Environment ────────────────────────────────────────────────
# Ollama — adjust path to match your setup
OLLAMA=/homes/jl10525/bin/ollama
export OLLAMA_MODELS=$HOME/ollama_data

# Start Ollama server in background
echo "== Starting Ollama server =="
$OLLAMA serve &
OLLAMA_PID=$!
sleep 5  # wait for server to initialize

# Verify Ollama is responsive
$OLLAMA list || { echo "ERROR: Ollama failed to start"; kill $OLLAMA_PID 2>/dev/null; exit 1; }

# Pull required models (no-op if already present)
echo "== Pulling LLM models =="
$OLLAMA pull qwen3-coder-next:cloud || echo "WARNING: failed to pull qwen3-coder-next:cloud"
$OLLAMA pull gemma3:12b-cloud       || echo "WARNING: failed to pull gemma3:12b-cloud"

# Project directory
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

# Activate conda env (uncomment and adjust as needed)
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate manga

# ─── Output directories ────────────────────────────────────────
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_BASE="output/batch_${TIMESTAMP}"
mkdir -p "$OUTPUT_BASE"

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
    local REF_IMG="tests/fixtures/meining.jpg"

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
        -r "$REF_IMG" \
        -p "$PROMPT" \
        --panels "$PANELS" \
        --seed "$SEED" \
        --audio \
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
    "A lone samurai stands guard on a rain-soaked rooftop. A massive combat mech crashes through the street below. The samurai leaps off the building, blade drawn, slicing through the rain. She lands on the mech's shoulder and drives her katana into its core." \
    42

# Story 2 — Fantasy / 6 panels
run_story 2 6 \
    "A young sorceress arrives at the entrance of an ancient crystal cave deep in a mystical forest. She discovers a sealed stone door covered in glowing runes and uses her staff to unlock it. Inside she finds a vast underground lake reflecting thousands of crystal stalactites. A massive water dragon emerges from the lake and roars, sending waves crashing. The sorceress raises her staff and casts a barrier of light, taming the dragon. She rides the dragon out of the cave as the sun rises over the forest." \
    101

# Story 3 — Cyberpunk / 4 panels
run_story 3 4 \
    "A hacker sits in a neon-lit underground den surrounded by holographic screens. She jacks into a corporate mainframe, her cybernetic eye flickering with data streams. Alarms blare as security drones swarm the corridor outside. She smashes through a window and escapes on a hoverbike into the rain-drenched city." \
    77

# Story 4 — Angsty Romance / 6 panels
run_story 4 6 \
    "A girl waits alone at a cherry blossom-covered train platform at dusk, clutching a letter she never sent. A boy appears on the opposite platform, their eyes meet through the falling petals. She runs toward the crossing but the signal turns red and a train roars between them. The train passes and he is gone, only his scarf caught on the railing remains. She presses the scarf against her face, tears streaming down, cherry blossoms swirling around her. She walks away into the sunset, the unsent letter slipping from her fingers onto the tracks." \
    256

# Story 5 — Sci-Fi / 4 panels
run_story 5 4 \
    "An astronaut floats through the wreckage of a destroyed space station, debris and sparks drifting in zero gravity. She spots an escape pod still intact, glowing faintly through the twisted metal. She pushes off a wall fragment and glides toward the pod, dodging a spinning piece of hull. She seals the pod door and launches into the stars as the station explodes behind her." \
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

# Stop Ollama
kill $OLLAMA_PID 2>/dev/null || true

echo "== GPU Status (final) =="
nvidia-smi || echo "no nvidia-smi"
date
