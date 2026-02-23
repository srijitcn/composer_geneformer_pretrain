# SGC GigaPath: Prov-GigaPath Fine-Tuning on Databricks SGC

This folder contains an adaptation of [Prov-GigaPath](https://github.com/prov-gigapath/prov-gigapath) for **multi-node distributed training** on [Databricks Serverless GPU Compute (SGC)](https://docs.databricks.com/). It runs slide-level fine-tuning of the GigaPath slide encoder on the PANDA dataset using `torchrun` with DDP across multiple H100 nodes.

## Relationship to the Original Repo

This code is derived from [prov-gigapath/prov-gigapath](https://github.com/prov-gigapath/prov-gigapath), the official implementation of the paper:

> Xu, H., Usuyama, N., et al. "A whole-slide foundation model for digital pathology from real-world data." *Nature* (2024).

The original repository provides single-GPU training scripts. This adaptation preserves the model architecture and dataset code while adding multi-node distributed training, checkpoint management, and Databricks integration. See [Changes from Upstream](#changes-from-upstream) for details.

## Changes from Upstream

**Distributed training** -- `finetune/main.py` and `finetune/training.py` rewritten to support multi-node DDP via `torchrun` and `torch.distributed`. Each process binds to its local GPU rank.

**DistributedSampler** -- `finetune/utils.py` uses `DistributedSampler` when `world_size > 1`, replacing the single-process `RandomSampler`.

**Gradient sync optimization** -- `model.no_sync()` is used during gradient accumulation steps to avoid redundant all-reduce until the final accumulation step.

**Autoresume** -- Training saves `checkpoint_latest.pt` (full state: model, optimizer, scaler, epoch) at configurable intervals (`save_interval_epochs`). On restart, training automatically resumes from the latest checkpoint.

**MLflow logging** -- Optional integration with Databricks MLflow for experiment tracking, including system metrics (GPU utilization, memory). Controlled by the `GIGAPATH_MLFLOW_EXPERIMENT` environment variable.

**SGC workload config** -- `train.yaml`, `commands.sh`, and `dependencies.yaml` are new files that define the SGC workload for submission via SGCLI.

**Databricks data/model staging notebooks** -- `model_preparation.py` downloads pretrained weights from HuggingFace to a Unity Catalog Volume. `data_preparation.py` stages PANDA H5 embeddings to a Volume.

**Updated dependencies** -- `xformers==0.0.31` and `flash-attn==2.8.2` for H100 compatibility (upstream uses `xformers==0.0.18` and `flash-attn==2.5.8`).

## Prerequisites

- A Databricks workspace with **Serverless GPU Compute (SGC)** enabled
- **SGCLI** installed locally from the internal wheel (not yet on PyPI): `pip install /path/to/databricks_sgcli-*.whl`
- A **HuggingFace account** with access to the [Prov-GigaPath model](https://huggingface.co/prov-gigapath/prov-gigapath) (accept the license terms)
- A **Unity Catalog Volume** on Databricks for storing data and model artifacts

## Quick Start

### 1. Prepare data

Run `data_preparation.py` as a Databricks notebook (on any CPU cluster). It stages the PANDA H5 tile embeddings to your Volume. Two source modes are supported:

- **`volume_zip`** -- You have already uploaded `GigaPath_PANDA_embeddings.zip` to your Volume. The notebook unzips it in place.
- **`hf_download`** -- The notebook downloads the ZIP from [HuggingFace](https://huggingface.co/datasets/prov-gigapath/prov-gigapath-tile-embeddings/tree/main) (32 GB), then unzips.

Edit the `CATALOG`, `SCHEMA`, `VOLUME_NAME`, and `SOURCE_MODE` variables at the top of the notebook to match your workspace.

### 2. Prepare model weights

Run `model_preparation.py` as a Databricks notebook. It downloads `slide_encoder.pth` from HuggingFace to your Volume. Set `HF_TOKEN` as needed.

After this step you will have a path like:

```
/Volumes/<catalog>/<schema>/<volume>/gigapath/model/slide_encoder.pth
```

### 3. Configure `train.yaml`

Edit `train.yaml` to point at your data and model paths:

```yaml
environment:
  env_variables:
    GIGAPATH_ROOT_PATH: /Volumes/<catalog>/<schema>/<volume>/gigapath/data/<path_to_h5_files>
    GIGAPATH_OUTPUT_DIR: /Volumes/<catalog>/<schema>/<volume>/gigapath/outputs/
    GIGAPATH_PRETRAINED: /Volumes/<catalog>/<schema>/<volume>/gigapath/model/slide_encoder.pth
    GIGAPATH_EPOCHS: "200"
    GIGAPATH_SAVE_INTERVAL_EPOCHS: "5"
    GIGAPATH_AUTORESUME: "1"
  dependencies: dependencies.yaml
compute:
  gpus: 16          # total GPUs across all nodes (e.g. 2 nodes x 8 H100)
  gpu_type: h100
code_source:
  type: snapshot
  snapshot:
    repo_path: /path/to/your/local/repo/
command: |-
  cd $HOME/composer_geneformer_pretrain/SGC_gigapath
  bash commands.sh
```

Also update `code_source.snapshot.repo_path` to the local path of this repository on your machine.

### 4. Submit the workload

```bash
cd SGC_gigapath
sgcli run -f train.yaml --watch
```

`--watch` streams logs to your terminal. Other useful commands:

```bash
sgcli get runs                          # list recent runs
sgcli get status <run-id> -p profile    # check run status
sgcli get logs <run-id> -p profile      # fetch run logs
```

## Configuration Reference

All training behavior is controlled through environment variables in `train.yaml`. The entry script `commands.sh` reads these and passes them to `torchrun`.

| Variable | Default | Description |
|---|---|---|
| `GIGAPATH_ROOT_PATH` | *(required)* | Path to the directory containing `<slide_id>.h5` embedding files |
| `GIGAPATH_OUTPUT_DIR` | `outputs/PANDA` | Directory for checkpoints, logs, and results |
| `GIGAPATH_EPOCHS` | `5` | Number of training epochs |
| `GIGAPATH_SAVE_INTERVAL_EPOCHS` | `5` | Save a checkpoint every N epochs |
| `GIGAPATH_AUTORESUME` | `1` | `1` to resume from `checkpoint_latest.pt` if it exists, `0` to start fresh |
| `GIGAPATH_PRETRAINED` | *(empty)* | Path to pretrained `slide_encoder.pth`. Empty means random initialization |
| `GIGAPATH_MLFLOW_EXPERIMENT` | `/mlflow_experiments/gigapath_finetuning` | MLflow experiment path in Databricks |
| `MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING` | `true` | Enable GPU/CPU/memory metrics in MLflow |

Multi-node environment variables (`NODE_RANK`, `MASTER_ADDR`, `MASTER_PORT`, `NNODES`) are provided automatically by the SGC platform.

## File Structure

```
SGC_gigapath/
├── train.yaml                 # SGC workload definition (compute, env vars, command)
├── commands.sh                # Entry script: reads env vars, launches torchrun
├── dependencies.yaml          # Pip dependencies for the SGC remote environment
├── model_preparation.py       # Databricks notebook: download model from HuggingFace
├── data_preparation.py        # Databricks notebook: stage PANDA H5 embeddings
│
├── finetune/                  # ** Modified from upstream **
│   ├── main.py                #   DDP init, per-rank logging, distributed effective batch size
│   ├── training.py            #   DDP wrapping, autoresume, MLflow, gradient sync
│   ├── utils.py               #   DistributedSampler support
│   ├── params.py              #   Added --save_interval_epochs, --autoresume args
│   ├── metrics.py             #   (unchanged)
│   ├── datasets/
│   │   └── slide_datatset.py  #   (unchanged)
│   └── task_configs/
│       ├── panda.yaml         #   PANDA: multi-class, 6 Gleason grades
│       ├── mutation_5_gene.yaml
│       └── utils.py
│
├── gigapath/                  # Model library (unchanged from upstream)
│   ├── slide_encoder.py       #   LongNet-based slide encoder
│   ├── classification_head.py #   Slide encoder + linear classifier
│   ├── pipeline.py
│   ├── pos_embed.py
│   ├── preprocessing/         #   WSI tiling and foreground segmentation
│   └── torchscale/            #   LongNet / dilated attention components
│
├── dataset_csv/               # Pre-split CSVs for PANDA, PCam, mutation tasks
├── demo/                      # Demo notebooks from upstream (tile/slide encoder usage)
├── linear_probe/              # Tile-level linear probing on PCam (from upstream)
├── scripts/                   # Original single-GPU run scripts (from upstream)
├── images/                    # Figures from original repo
│
├── environment.yaml           # Conda environment for local development
├── pyproject.toml             # Package metadata (pip install -e .)
├── requirements.txt           # Pip requirements (from upstream)
└── LICENSE                    # Apache 2.0
```

## Local Development

For local development or debugging outside of Databricks:

1. Create the conda environment:

```bash
conda env create -f environment.yaml
conda activate gigapath
pip install -e .
```

2. Run single-GPU training directly:

```bash
python finetune/main.py \
  --task_cfg_path finetune/task_configs/panda.yaml \
  --dataset_csv dataset_csv/PANDA/PANDA.csv \
  --pre_split_dir dataset_csv/PANDA \
  --root_path /path/to/h5_files \
  --model_arch gigapath_slide_enc12l768d \
  --epochs 5 \
  --folds 1 \
  --save_dir outputs/PANDA
```

Or multi-GPU on a single node:

```bash
torchrun --nproc_per_node=4 finetune/main.py \
  --task_cfg_path finetune/task_configs/panda.yaml \
  --dataset_csv dataset_csv/PANDA/PANDA.csv \
  --pre_split_dir dataset_csv/PANDA \
  --root_path /path/to/h5_files \
  --model_arch gigapath_slide_enc12l768d \
  --epochs 5 \
  --folds 1 \
  --save_dir outputs/PANDA
```

## Citation

This work builds on Prov-GigaPath. If you use this code, please cite the original paper:

```bibtex
@article{xu2024gigapath,
  title={A whole-slide foundation model for digital pathology from real-world data},
  author={Xu, Hanwen and Usuyama, Naoto and Bagga, Jaspreet and Zhang, Sheng and Rao, Rajesh and Naumann, Tristan and Wong, Cliff and Gero, Zelalem and Gonz{\'a}lez, Javier and Gu, Yu and Xu, Yanbo and Wei, Mu and Wang, Wenhui and Ma, Shuming and Wei, Furu and Yang, Jianwei and Li, Chunyuan and Gao, Jianfeng and Rosemon, Jaylen and Bower, Tucker and Lee, Soohee and Weerasinghe, Roshanthi and Wright, Bill J. and Robicsek, Ari and Piening, Brian and Bifulco, Carlo and Wang, Sheng and Poon, Hoifung},
  journal={Nature},
  year={2024},
  publisher={Nature Publishing Group UK London}
}
```

## License

Apache 2.0 -- see [LICENSE](LICENSE).
