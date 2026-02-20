#!/bin/bash
#PBS -N train_ghibli_lora
#PBS -l select=1:ncpus=4:mem=32gb:ngpus=1
#PBS -l walltime=24:00:00

# --- EMAIL NOTIFICATIONS ---
# Send email on abort (a), begin (b), and end (e)
#PBS -m abe
# Send emails to this address:
#PBS -M your.name@imperial.ac.uk
# --------------------------------

cd $PBS_O_WORKDIR

# Activate environment
source $HOME/miniconda/bin/activate
conda activate sdxl_lora

# Redirect Cache to avoid 100GB quota limits
export HF_HOME=$EPHEMERAL/huggingface_cache
mkdir -p $HF_HOME

# Navigate into the diffusers folder before running
cd diffusers

# Run Training
accelerate launch examples/text_to_image/train_text_to_image_lora_sdxl.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
  --pretrained_vae_model_name_or_path="madebyollin/sdxl-vae-fp16-fix" \
  --dataset_name="moving-j/ghibli-style-100" \
  --caption_column="text" \
  --resolution=1024 \
  --random_flip \
  --train_batch_size=1 \
  --num_train_epochs=50 \
  --checkpointing_steps=500 \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --mixed_precision="fp16" \
  --seed=42 \
  --output_dir="sdxl-ghibli-100-lora-final" \
  --gradient_checkpointing \
  --use_8bit_adam