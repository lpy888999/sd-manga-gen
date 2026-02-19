from huggingface_hub import HfApi
api = HfApi()

REPO_ID = "YourUsername/sdxl-ghibli-100-lora"  # Change to your own repo
TOKEN = "hf_YOUR_WRITE_TOKEN_HERE" 

api.create_repo(repo_id=REPO_ID, token=TOKEN, repo_type="model", exist_ok=True)
api.upload_folder(
    folder_path="sdxl-ghibli-100-lora-final",
    repo_id=REPO_ID,
    repo_type="model",
    token=TOKEN
)