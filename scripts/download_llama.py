from huggingface_hub import hf_hub_download
import os

# -------------------------------
# SETTINGS
# -------------------------------
# Repo where quantized LLaMA3 is hosted
repo_id = "bartowski/Meta-Llama-3-8B-Instruct-GGUF"

# Which file you want (check Hugging Face repo "Files and versions" tab)
filename = "Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"

# Local folder to store models
local_dir = os.path.join("models")
os.makedirs(local_dir, exist_ok=True)

# -------------------------------
# DOWNLOAD
# -------------------------------
print(f"Downloading {filename} from {repo_id} ...")
model_path = hf_hub_download(
    repo_id=repo_id,
    filename=filename,
    local_dir=local_dir
)

print(f"Download complete! Saved at: {model_path}")
