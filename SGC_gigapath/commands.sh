#!/usr/bin/env bash
set -euo pipefail

echo ">>> Starting Prov-GigaPath finetuning"
cd "$HOME/composer_geneformer_pretrain/SGC_gigapath"

# Root path should point to PANDA H5 embeddings directory.
# Example: /Volumes/main/<schema>/<volume>/gigapath/data/GigaPath_PANDA_embeddings/h5_files
ROOT_PATH="${GIGAPATH_ROOT_PATH:-/Volumes/main/guanyu_chen/sgc/gigapath/data/GigaPath_PANDA_embeddings/h5_files}"
OUTPUT_DIR="${GIGAPATH_OUTPUT_DIR:-outputs/PANDA}"
EPOCHS="${GIGAPATH_EPOCHS:-5}"
SAVE_INTERVAL_EPOCHS="${GIGAPATH_SAVE_INTERVAL_EPOCHS:-5}"
AUTORESUME="${GIGAPATH_AUTORESUME:-1}"
PRETRAINED="${GIGAPATH_PRETRAINED:-}"

count_train_matches() {
  local candidate_root="$1"
  python - "$candidate_root" <<'PY'
import csv
import os
import sys

root = sys.argv[1]
train_csv = "dataset_csv/PANDA/train_0.csv"
if not os.path.exists(root):
    print(0)
    raise SystemExit(0)

count = 0
with open(train_csv, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        sid = row["slide_id"]
        if os.path.exists(os.path.join(root, f"{sid}.h5")):
            count += 1
print(count)
PY
}

CURRENT_MATCHES="$(count_train_matches "${ROOT_PATH}")"
if [[ "${CURRENT_MATCHES}" -eq 0 ]]; then
  echo ">>> No training .h5 files found under ROOT_PATH=${ROOT_PATH}"
  echo ">>> Trying common fallback directories..."
  CANDIDATES=(
    "/Volumes/main/guanyu_chen/sgc/gigapath/data/dinov2_features/h5_files"
    "/Volumes/main/guanyu_chen/sgc/gigapath/data/GigaPath_PANDA_embeddings/h5_files"
    "/Volumes/main/guanyu_chen/sgc/gigapath/data/h5_files"
  )
  BEST_ROOT=""
  BEST_MATCHES=0
  for cand in "${CANDIDATES[@]}"; do
    matches="$(count_train_matches "${cand}")"
    echo ">>> candidate=${cand} train_matches=${matches}"
    if [[ "${matches}" -gt "${BEST_MATCHES}" ]]; then
      BEST_MATCHES="${matches}"
      BEST_ROOT="${cand}"
    fi
  done
  if [[ "${BEST_MATCHES}" -gt 0 ]]; then
    ROOT_PATH="${BEST_ROOT}"
    echo ">>> Auto-selected ROOT_PATH=${ROOT_PATH} (train_matches=${BEST_MATCHES})"
  else
    echo ">>> ERROR: Could not find any matching PANDA .h5 files."
    echo ">>> Set GIGAPATH_ROOT_PATH to the directory containing <slide_id>.h5 files."
    exit 1
  fi
fi

echo ">>> ROOT_PATH=${ROOT_PATH}"
echo ">>> OUTPUT_DIR=${OUTPUT_DIR}"
echo ">>> EPOCHS=${EPOCHS}"
echo ">>> SAVE_INTERVAL_EPOCHS=${SAVE_INTERVAL_EPOCHS}"
echo ">>> AUTORESUME=${AUTORESUME}"
if [[ -z "${PRETRAINED}" ]]; then
  echo ">>> PRETRAINED is empty: training with random initialization"
else
  echo ">>> PRETRAINED=${PRETRAINED}"
fi

python finetune/main.py \
  --task_cfg_path finetune/task_configs/panda.yaml \
  --dataset_csv dataset_csv/PANDA/PANDA.csv \
  --pre_split_dir dataset_csv/PANDA \
  --root_path "${ROOT_PATH}" \
  --model_arch gigapath_slide_enc12l768d \
  --epochs "${EPOCHS}" \
  --save_interval_epochs "${SAVE_INTERVAL_EPOCHS}" \
  --autoresume "${AUTORESUME}" \
  --folds 1 \
  --save_dir "${OUTPUT_DIR}" \
  --report_to tensorboard \
  --pretrained "${PRETRAINED}"
