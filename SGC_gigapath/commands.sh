#!/usr/bin/env bash
set -euo pipefail

echo ">>> Starting Prov-GigaPath finetuning"
cd "$HOME/composer_geneformer_pretrain/prov_gigapath"

# Root path should point to PANDA H5 embeddings directory.
# Example: /Volumes/main/<schema>/<volume>/gigapath/data/GigaPath_PANDA_embeddings/h5_files
ROOT_PATH="${GIGAPATH_ROOT_PATH:-/Volumes/main/guanyu_chen/sgc/gigapath/data/GigaPath_PANDA_embeddings/h5_files}"
OUTPUT_DIR="${GIGAPATH_OUTPUT_DIR:-outputs/PANDA}"
EPOCHS="${GIGAPATH_EPOCHS:-5}"
PRETRAINED="${GIGAPATH_PRETRAINED:-}"

echo ">>> ROOT_PATH=${ROOT_PATH}"
echo ">>> OUTPUT_DIR=${OUTPUT_DIR}"
echo ">>> EPOCHS=${EPOCHS}"
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
  --folds 1 \
  --save_dir "${OUTPUT_DIR}" \
  --report_to tensorboard \
  --pretrained "${PRETRAINED}"
