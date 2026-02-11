# Databricks notebook source
# MAGIC %md
# MAGIC # Prov-GigaPath Model Preparation
# MAGIC
# MAGIC This notebook downloads Prov-GigaPath model artifacts from HuggingFace to:
# MAGIC
# MAGIC `/Volumes/main/guanyu_chen/sgc/gigapath/model`
# MAGIC
# MAGIC It supports:
# MAGIC - `slide_only`: download only `slide_encoder.pth` (enough for current finetune path)
# MAGIC - `snapshot`: download a broader model snapshot

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1) Configuration

# COMMAND ----------

import os
from pathlib import Path

# Volume destination
CATALOG = "main"
SCHEMA = "guanyu_chen"
VOLUME_NAME = "sgc"
MODEL_REL_PATH = "gigapath/model"

# HuggingFace source
HF_REPO_ID = "prov-gigapath/prov-gigapath"
HF_REPO_TYPE = "model"

# Download mode:
# - "slide_only": only slide_encoder.pth
# - "snapshot": download multiple files from the model repo
DOWNLOAD_MODE = "slide_only"

# Optional token (required if your HF account needs accepted terms/auth)
# You can set it as Databricks env var HF_TOKEN, or paste directly here.
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# If True, force redownload from HF even if local files exist.
FORCE_DOWNLOAD = False

# If True and DOWNLOAD_MODE == "snapshot", removes previous snapshot folder first.
RESET_SNAPSHOT_DIR = False

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2) Setup helpers

# COMMAND ----------

import shutil

VOLUME_ROOT = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME_NAME}"
MODEL_DIR = f"{VOLUME_ROOT}/{MODEL_REL_PATH}"
SLIDE_ENCODER_PATH = f"{MODEL_DIR}/slide_encoder.pth"
SNAPSHOT_DIR = f"{MODEL_DIR}/hf_snapshot"

print("=" * 80)
print("CONFIGURATION")
print("=" * 80)
print(f"VOLUME_ROOT      : {VOLUME_ROOT}")
print(f"MODEL_DIR        : {MODEL_DIR}")
print(f"SLIDE_ENCODER    : {SLIDE_ENCODER_PATH}")
print(f"SNAPSHOT_DIR     : {SNAPSHOT_DIR}")
print(f"HF_REPO_ID       : {HF_REPO_ID}")
print(f"DOWNLOAD_MODE    : {DOWNLOAD_MODE}")
print(f"FORCE_DOWNLOAD   : {FORCE_DOWNLOAD}")
print("=" * 80)

os.makedirs(MODEL_DIR, exist_ok=True)


def file_size_mb(path: str) -> float:
    return Path(path).stat().st_size / (1024 * 1024)


def print_train_yaml_snippet(pretrained_value: str):
    print("\nUse this in SGC_gigapath/train.yaml:")
    print(f"GIGAPATH_PRETRAINED: {pretrained_value}")


# COMMAND ----------

# MAGIC %md
# MAGIC ## 3) Install/import `huggingface_hub`

# COMMAND ----------

try:
    from huggingface_hub import hf_hub_download, snapshot_download, HfApi
except Exception:
    import subprocess
    subprocess.check_call(["pip", "install", "huggingface-hub"])
    from huggingface_hub import hf_hub_download, snapshot_download, HfApi

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4) Inspect available files on HF repo (optional but useful)

# COMMAND ----------

api = HfApi()
files = api.list_repo_files(repo_id=HF_REPO_ID, repo_type=HF_REPO_TYPE, token=HF_TOKEN if HF_TOKEN else None)
print(f"Found {len(files)} files in {HF_REPO_ID}:")
for f in files[:30]:
    print(f"  - {f}")
if len(files) > 30:
    print("  ...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5) Download model artifacts

# COMMAND ----------

if DOWNLOAD_MODE == "slide_only":
    local_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        repo_type=HF_REPO_TYPE,
        filename="slide_encoder.pth",
        local_dir=MODEL_DIR,
        force_download=FORCE_DOWNLOAD,
        token=HF_TOKEN if HF_TOKEN else None,
    )
    if not os.path.exists(SLIDE_ENCODER_PATH):
        # hf_hub_download may return nested cache location in some environments.
        # Ensure the training-friendly path exists by copying.
        shutil.copy2(local_path, SLIDE_ENCODER_PATH)
    print(f"Downloaded: {SLIDE_ENCODER_PATH} ({file_size_mb(SLIDE_ENCODER_PATH):.1f} MB)")
    print_train_yaml_snippet(SLIDE_ENCODER_PATH)

elif DOWNLOAD_MODE == "snapshot":
    if RESET_SNAPSHOT_DIR and os.path.exists(SNAPSHOT_DIR):
        shutil.rmtree(SNAPSHOT_DIR)

    # Keep snapshot focused on commonly needed model files.
    allow_patterns = [
        "*.pth",
        "*.pt",
        "*.bin",
        "*.safetensors",
        "*.json",
        "*.txt",
        "*.md",
    ]
    snapshot_path = snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type=HF_REPO_TYPE,
        local_dir=SNAPSHOT_DIR,
        allow_patterns=allow_patterns,
        token=HF_TOKEN if HF_TOKEN else None,
        force_download=FORCE_DOWNLOAD,
    )
    print(f"Snapshot downloaded to: {snapshot_path}")

    # If slide_encoder.pth exists in snapshot, provide exact path for training.
    candidate = f"{SNAPSHOT_DIR}/slide_encoder.pth"
    if os.path.exists(candidate):
        print_train_yaml_snippet(candidate)
    else:
        print("slide_encoder.pth not found at snapshot root.")
        print("Check files under snapshot dir and pick the correct local path.")

else:
    raise ValueError("DOWNLOAD_MODE must be either 'slide_only' or 'snapshot'")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6) Final check

# COMMAND ----------

if os.path.exists(SLIDE_ENCODER_PATH):
    print(f"✅ slide_encoder.pth ready: {SLIDE_ENCODER_PATH}")
else:
    print("ℹ️ slide_encoder.pth not at the default root path (may still be in snapshot).")

print("\nRecommended env var in SGC_gigapath/train.yaml:")
print(f"GIGAPATH_PRETRAINED: {SLIDE_ENCODER_PATH}")

