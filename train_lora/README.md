# 🚀 Training SDXL LoRA on Imperial College HPC (CX3)

This guide documents the steps our group took to train our Generative AI coursework LoRA model using Imperial College's High-Performance Computing (HPC) cluster.

## 1. Prerequisites
- **Imperial VPN:** You must be connected to the Unified Access VPN to reach the HPC.
- **HPC Account:** Ensure you are registered for RCS HPC access.
- **Hugging Face Token:** You need a HF token with **WRITE** permissions.

Log into the HPC via SSH:
```bash
ssh your_username@login.hpc.imperial.ac.uk
```

## 2. Environment Setup (Login Node)
Once logged in, set up a local Miniconda environment and install the Hugging Face `diffusers` library:

```bash
# 1. Install Miniconda (Required if you haven't used Conda on HPC before)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda

# 2. Download and clone diffusers
git clone https://github.com/huggingface/diffusers.git

# 3. Activate Conda and accept Terms of Service
source ~/miniconda/bin/activate
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# 4. Create and activate environment
conda create -n sdxl_lora python=3.10 -y
conda activate sdxl_lora

# 5. Install dependencies
cd diffusers
pip install --upgrade pip
pip install -r examples/text_to_image/requirements.txt
pip install -e .
pip install accelerate transformers peft bitsandbytes datasets huggingface_hub
```

## 3. The PBS Job Script (Compute Node)
We cannot train on the login node. Create a job script named `train_lora.sh` in your home directory (`cd ~`):

```bash
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
```
You can adjust the resource requests (CPUs, memory, walltime) as needed. Additionally, you can adapt the training parameters (epochs, batch size, learning rate) and dataset selection (dataset_name) and pre-trained model (pretrained_model_name_or_path) based on your requirements.
Submit the job to the queue using:
```bash
qsub train_lora.sh
```

## 4. Uploading the Finished Model to Hugging Face
Once the job finishes successfully (check via `qstat`), the weights are saved in the output directory. We used a custom Python script (`upload.py`) to push the `.safetensors` files to the Hub.

Create `upload.py` inside the `diffusers` folder:

```python
from huggingface_hub import HfApi
api = HfApi()

REPO_ID = "CheCui/sdxl-ghibli-100-lora"
TOKEN = "hf_YOUR_WRITE_TOKEN_HERE" 

api.create_repo(repo_id=REPO_ID, token=TOKEN, repo_type="model", exist_ok=True)
api.upload_folder(
    folder_path="sdxl-ghibli-100-lora-final",
    repo_id=REPO_ID,
    repo_type="model",
    token=TOKEN
)
```
Ensure your environment is active, then run the script on the login node: 
```bash
python upload.py
```