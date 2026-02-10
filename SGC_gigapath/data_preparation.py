# Databricks notebook source
# MAGIC %md
# MAGIC # Prov-GigaPath PANDA Data Preparation
# MAGIC
# MAGIC This notebook prepares PANDA slide embedding files for `SGC_gigapath` training.
# MAGIC
# MAGIC It supports two data source modes:
# MAGIC 1. `volume_zip`: You already uploaded `GigaPath_PANDA_embeddings.zip` to your volume.
# MAGIC 2. `hf_download`: Download `GigaPath_PANDA_embeddings.zip` from HuggingFace, then unzip.
# MAGIC
# MAGIC Run this notebook on a CPU cluster first, then run `sgcli` training.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1) Configuration (edit this section)

# COMMAND ----------

import os
import glob
import shutil
import subprocess
from pathlib import Path

# -----------------------------
# Databricks volume configuration
# -----------------------------
CATALOG = "main"
SCHEMA = "guanyu_chen"
VOLUME_NAME = "sgc"

# -----------------------------
# Data layout inside your volume
# -----------------------------
# Final target path expected by training:
# /Volumes/{CATALOG}/{SCHEMA}/{VOLUME_NAME}/gigapath/data/dinov2_features/h5_files
GIGAPATH_DATA_ROOT_REL = "gigapath/data"
TARGET_PARENT_REL = "gigapath/data/dinov2_features"
TARGET_H5_REL = "gigapath/data/dinov2_features/h5_files"

# -----------------------------
# Source mode
# -----------------------------
# "volume_zip" -> use ZIP already present in your volume
# "hf_download" -> download ZIP from HuggingFace first
SOURCE_MODE = "volume_zip"

# If SOURCE_MODE == "volume_zip", set this to your zip location.
# Example:
# /Volumes/main/guanyu_chen/sgc/gigapath/data/GigaPath_PANDA_embeddings.zip
VOLUME_ZIP_PATH = "/Volumes/main/guanyu_chen/sgc/gigapath/data/GigaPath_PANDA_embeddings.zip"

# If SOURCE_MODE == "hf_download", configure HF source.
HF_REPO_ID = "prov-gigapath/prov-gigapath-tile-embeddings"
HF_REPO_TYPE = "dataset"
HF_ZIP_FILENAME = "GigaPath_PANDA_embeddings.zip"
# Set token if required by dataset access terms.
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# If True, remove existing target folder before unzip/download.
FORCE_RECREATE = False

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2) Paths and helper functions

# COMMAND ----------

VOLUME_ROOT = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME_NAME}"
DATA_ROOT = f"{VOLUME_ROOT}/{GIGAPATH_DATA_ROOT_REL}"
TARGET_PARENT = f"{VOLUME_ROOT}/{TARGET_PARENT_REL}"
TARGET_H5_DIR = f"{VOLUME_ROOT}/{TARGET_H5_REL}"
DOWNLOAD_ZIP_PATH = f"{DATA_ROOT}/{HF_ZIP_FILENAME}"

print("=" * 80)
print("CONFIG")
print("=" * 80)
print(f"VOLUME_ROOT      : {VOLUME_ROOT}")
print(f"DATA_ROOT        : {DATA_ROOT}")
print(f"TARGET_PARENT    : {TARGET_PARENT}")
print(f"TARGET_H5_DIR    : {TARGET_H5_DIR}")
print(f"SOURCE_MODE      : {SOURCE_MODE}")
print(f"VOLUME_ZIP_PATH  : {VOLUME_ZIP_PATH}")
print(f"HF_REPO_ID       : {HF_REPO_ID}")
print(f"HF_ZIP_FILENAME  : {HF_ZIP_FILENAME}")
print(f"FORCE_RECREATE   : {FORCE_RECREATE}")
print("=" * 80)


def run_cmd(cmd: list[str], desc: str):
    print(f"\n[RUN] {desc}")
    print(" ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr)
        raise RuntimeError(f"Failed: {desc}")
    if proc.stdout.strip():
        print(proc.stdout)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def count_h5(path: str) -> int:
    return len(glob.glob(os.path.join(path, "*.h5")))


def validate_output(path: str):
    n = count_h5(path)
    print(f"\nFound {n} .h5 files in: {path}")
    if n <= 0:
        raise RuntimeError(
            "No .h5 files found after preparation. Check source ZIP and unzip location."
        )
    # Print a few examples for quick sanity check
    examples = glob.glob(os.path.join(path, "*.h5"))[:5]
    print("Sample files:")
    for x in examples:
        print(f"  - {x}")


def reset_target_if_needed():
    if FORCE_RECREATE and os.path.exists(TARGET_PARENT):
        print(f"Removing existing target parent: {TARGET_PARENT}")
        shutil.rmtree(TARGET_PARENT)


# COMMAND ----------

# MAGIC %md
# MAGIC ## 3) Prepare data from selected source mode

# COMMAND ----------

reset_target_if_needed()
ensure_dir(DATA_ROOT)
ensure_dir(TARGET_PARENT)

if SOURCE_MODE == "volume_zip":
    if not os.path.exists(VOLUME_ZIP_PATH):
        raise FileNotFoundError(
            f"ZIP not found at VOLUME_ZIP_PATH: {VOLUME_ZIP_PATH}\n"
            "Upload the ZIP to your volume path or switch SOURCE_MODE to hf_download."
        )
    # unzip -n keeps existing files if rerun
    run_cmd(["unzip", "-n", VOLUME_ZIP_PATH, "-d", TARGET_PARENT], "Unzip volume ZIP")

elif SOURCE_MODE == "hf_download":
    try:
        from huggingface_hub import hf_hub_download
    except Exception:
        run_cmd(["pip", "install", "huggingface-hub"], "Install huggingface-hub")
        from huggingface_hub import hf_hub_download

    print("\nDownloading ZIP from HuggingFace...")
    local_zip = hf_hub_download(
        repo_id=HF_REPO_ID,
        repo_type=HF_REPO_TYPE,
        filename=HF_ZIP_FILENAME,
        token=HF_TOKEN if HF_TOKEN else None,
        local_dir=DATA_ROOT,
        local_dir_use_symlinks=False,
    )
    print(f"Downloaded ZIP to: {local_zip}")
    run_cmd(["unzip", "-n", local_zip, "-d", TARGET_PARENT], "Unzip downloaded ZIP")

else:
    raise ValueError("SOURCE_MODE must be either 'volume_zip' or 'hf_download'")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4) Validate prepared folder

# COMMAND ----------

# Some archives unzip directly to h5_files/, while others may nest paths.
# If expected TARGET_H5_DIR does not exist, try to discover one.
if not os.path.exists(TARGET_H5_DIR):
    candidates = glob.glob(f"{TARGET_PARENT}/**/h5_files", recursive=True)
    if candidates:
        print(f"TARGET_H5_DIR not found. Using discovered candidate: {candidates[0]}")
        TARGET_H5_DIR = candidates[0]

validate_output(TARGET_H5_DIR)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5) Copy this value into `SGC_gigapath/train.yaml`

# COMMAND ----------

print("\nUse this path in SGC workload env var:")
print(f"GIGAPATH_ROOT_PATH: {TARGET_H5_DIR}")

print("\nExample snippet for SGC_gigapath/train.yaml:")
print(f"""
environment:
  env_variables:
    GIGAPATH_ROOT_PATH: {TARGET_H5_DIR}
""")

