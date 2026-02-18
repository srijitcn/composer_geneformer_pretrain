import os
import sys
import re
import json
import glob
from pathlib import Path
from contextlib import nullcontext

# For convinience
this_file_dir = Path(__file__).resolve().parent
sys.path.append(str(this_file_dir.parent))

import time
import wandb
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import torch.utils.tensorboard as tensorboard
try:
    import mlflow
except Exception:
    mlflow = None

from gigapath.classification_head import get_model
from metrics import calculate_metrics_with_task_cfg
from utils import (get_optimizer, get_loss_function, \
                  Monitor_Score, get_records_array,
                  log_writer, adjust_learning_rate)


def _is_mlflow_enabled() -> bool:
    flag = os.environ.get("GIGAPATH_ENABLE_MLFLOW", "true").strip().lower()
    return flag in {"1", "true", "yes", "y", "on"}


def _is_primary_process() -> bool:
    """
    Return True only for global rank 0 so MLflow logging is not duplicated
    across nodes/processes.
    """
    # If torch.distributed is initialized, trust it first.
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank() == 0
    except Exception:
        pass

    # Fallback to common launcher env vars.
    for key in ("RANK", "WORLD_RANK", "NODE_RANK"):
        value = os.environ.get(key)
        if value is not None:
            try:
                return int(value) == 0
            except ValueError:
                continue

    # Default single-process behavior.
    return True


def _setup_mlflow(args):
    """
    Prefer the active SGC-provided MLflow run context.
    Only create a run if none exists.
    """
    if mlflow is None:
        return False, False, None

    tracking_uri = os.environ.get("GIGAPATH_MLFLOW_TRACKING_URI", "databricks")
    mlflow.set_tracking_uri(tracking_uri)

    active = mlflow.active_run()
    started_here = False
    if active is None:
        mlflow.start_run(run_name=args.exp_code)
        started_here = True

    run = mlflow.active_run()
    run_id = run.info.run_id if run is not None else "unknown"
    exp_id = run.info.experiment_id if run is not None else "unknown"
    print(
        "MLflow configured: "
        f"tracking_uri={tracking_uri}, "
        f"experiment_id={exp_id}, run_id={run_id}"
    )

    return True, started_here, run_id


def _sanitize_mlflow_metrics(metrics: dict) -> dict:
    out = {}
    for k, v in metrics.items():
        if 'prob' in k or 'label' in k:
            continue
        if isinstance(v, (int, float, np.floating, np.integer)):
            out[k] = float(v)
    return out


def _checkpoint_epoch_from_filename(path: str):
    match = re.search(r"checkpoint_epoch_(\d+)\.pt$", os.path.basename(path))
    if not match:
        return None
    # Filename is 1-based epoch number; internal epoch is 0-based.
    return int(match.group(1)) - 1


def _checkpoint_epoch_from_file(path: str):
    try:
        ckpt = torch.load(path, map_location="cpu")
        if isinstance(ckpt, dict) and "epoch" in ckpt:
            return int(ckpt["epoch"])
    except Exception:
        pass
    return _checkpoint_epoch_from_filename(path)


def _write_checkpoint_index(index_path: str, latest_path: str, latest_epoch: int, all_epoch_ckpts: list):
    payload = {
        "latest_checkpoint": latest_path,
        "latest_epoch_0_based": int(latest_epoch),
        "latest_epoch_1_based": int(latest_epoch) + 1,
        "epoch_checkpoints": sorted(all_epoch_ckpts),
    }
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _select_resume_checkpoint(model_dir: str):
    latest_ckpt = os.path.join(model_dir, "checkpoint_latest.pt")
    if os.path.exists(latest_ckpt):
        return latest_ckpt

    epoch_ckpts = glob.glob(os.path.join(model_dir, "checkpoint_epoch_*.pt"))
    if not epoch_ckpts:
        return None

    best_path = None
    best_epoch = -1
    for ckpt_path in epoch_ckpts:
        epoch = _checkpoint_epoch_from_file(ckpt_path)
        if epoch is None:
            continue
        if epoch > best_epoch:
            best_epoch = epoch
            best_path = ckpt_path
    return best_path


def train(dataloader, fold, args):
    train_loader, val_loader, test_loader = dataloader
    fold_save_dir = os.path.join(args.save_dir, f'fold_{fold}')
    model_dir = os.path.join(args.save_dir, "model", f'fold_{fold}')
    os.makedirs(model_dir, exist_ok=True)
    checkpoint_index_path = os.path.join(model_dir, "checkpoint_index.json")

    def save_training_state(epoch: int, filename: str):
        ckpt_path = os.path.join(model_dir, filename)
        state = {
            "epoch": int(epoch),
            "epoch_1_based": int(epoch) + 1,
            "model_state_dict": raw_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": fp16_scaler.state_dict() if fp16_scaler is not None else None,
            "args": vars(args),
        }
        torch.save(state, ckpt_path)
        epoch_ckpts = glob.glob(os.path.join(model_dir, "checkpoint_epoch_*.pt"))
        _write_checkpoint_index(
            index_path=checkpoint_index_path,
            latest_path=os.path.join(model_dir, "checkpoint_latest.pt"),
            latest_epoch=epoch,
            all_epoch_ckpts=epoch_ckpts,
        )
        print(f"Saved checkpoint from epoch {epoch + 1}: {ckpt_path}")

    # TensorBoard event files need append support; UC volume paths can raise OSError(29).
    # Keep checkpoints on args.save_dir, but write TensorBoard logs to local disk.
    tb_root = os.environ.get("GIGAPATH_TENSORBOARD_DIR", "/tmp/gigapath_tensorboard")
    writer_dir = os.path.join(tb_root, args.exp_code, f'fold_{fold}')
    if not os.path.isdir(writer_dir):
        os.makedirs(writer_dir, exist_ok=True)

    # set up the writer
    writer = tensorboard.SummaryWriter(writer_dir, flush_secs=15)
    mlflow_enabled = _is_mlflow_enabled() and (mlflow is not None) and _is_primary_process()
    mlflow_started_here = False
    mlflow_run_id = None
    if _is_mlflow_enabled() and (mlflow is not None) and not mlflow_enabled:
        print("MLflow logging disabled on non-primary process to avoid duplicate metrics.")
    if mlflow_enabled:
        try:
            mlflow_enabled, mlflow_started_here, mlflow_run_id = _setup_mlflow(args)
            mlflow.log_params({
                "exp_code": args.exp_code,
                "task": args.task,
                "epochs": int(args.epochs),
                "save_interval_epochs": int(args.save_interval_epochs),
                "autoresume": int(args.autoresume),
                "batch_size": int(args.batch_size),
                "gc": int(args.gc),
                "lr_scheduler": str(args.lr_scheduler),
                "blr": float(args.blr),
                "optim": str(args.optim),
                "optim_wd": float(args.optim_wd),
                "root_path": str(args.root_path),
                "save_dir": str(args.save_dir),
                "report_to": str(args.report_to),
                "mlflow_run_id": str(mlflow_run_id or ""),
            })
        except Exception as e:
            print(f"Warning: MLflow setup failed, continue without MLflow logging. Error: {e}")
            mlflow_enabled = False
    # set up writer
    if "wandb" in args.report_to:
        wandb.init(
            project=args.exp_code,
            name=args.exp_code + '_fold_' + str(fold),
            id='fold_' + str(fold),
            tags=[],
            config=vars(args),
        )
        writer = wandb
    elif "tensorboard" in args.report_to:
        writer = tensorboard.SummaryWriter(writer_dir, flush_secs=15)

    # set up the model
    model = get_model(**vars(args))
    model = model.to(args.device)
    # set up the optimizer
    optimizer = get_optimizer(args, model)
    # set up the loss function
    loss_fn = get_loss_function(args.task_config)
    # set up the monitor
    monitor = Monitor_Score()
    # set up the fp16 scaler
    fp16_scaler = None
    if args.fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()
        print('Using fp16 training')

    # Optional autoresume from latest checkpoint.
    start_epoch = 0
    resume_ckpt_path = _select_resume_checkpoint(model_dir) if int(args.autoresume) == 1 else None
    if int(args.autoresume) == 1 and resume_ckpt_path is not None:
        try:
            ckpt = torch.load(resume_ckpt_path, map_location="cpu")
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                model.load_state_dict(ckpt["model_state_dict"])
                if "optimizer_state_dict" in ckpt and ckpt["optimizer_state_dict"] is not None:
                    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                if fp16_scaler is not None and ckpt.get("scaler_state_dict") is not None:
                    fp16_scaler.load_state_dict(ckpt["scaler_state_dict"])
                start_epoch = int(ckpt.get("epoch", -1)) + 1
                print(f"Auto-resume: loaded {resume_ckpt_path}, restarting at epoch {start_epoch}")
            else:
                print(f"Auto-resume: checkpoint format not recognized at {resume_ckpt_path}, starting fresh")
        except Exception as e:
            print(f"Auto-resume failed to load {resume_ckpt_path}: {e}")
            print("Starting training from epoch 0")

    # Wrap model with DDP for multi-GPU training
    if getattr(args, 'world_size', 1) > 1:
        model = DDP(model, device_ids=[args.local_rank], find_unused_parameters=True)
    raw_model = model.module if hasattr(model, 'module') else model
    is_main = getattr(args, 'rank', 0) == 0

    if is_main:
        print('Training on {} samples'.format(len(train_loader.dataset)))
        print('Validating on {} samples'.format(len(val_loader.dataset))) if val_loader is not None else None
        print('Testing on {} samples'.format(len(test_loader.dataset))) if test_loader is not None else None
        print('Training starts!')

    # test evaluate function
    # val_records = evaluate(val_loader, model, fp16_scaler, loss_fn, 0, args)

    val_records, test_records = None, None

    last_epoch_ran = start_epoch - 1
    for i in range(start_epoch, args.epochs):
        last_epoch_ran = i
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(i)
        if is_main:
            print('Epoch: {}'.format(i))
        train_records = train_one_epoch(train_loader, model, fp16_scaler, optimizer, loss_fn, i, args)

        if is_main:
            if val_loader is not None:
                val_records = evaluate(val_loader, model, fp16_scaler, loss_fn, i, args)

                log_dict = {'train_' + k: v for k, v in train_records.items() if 'prob' not in k and 'label' not in k}
                log_dict.update({'val_' + k: v for k, v in val_records.items() if 'prob' not in k and 'label' not in k})
                log_writer(log_dict, i, args.report_to, writer)
                if mlflow_enabled:
                    try:
                        mlflow.log_metrics(_sanitize_mlflow_metrics(log_dict), step=i)
                    except Exception as e:
                        print(f"Warning: MLflow metric logging failed at epoch {i}: {e}")
                scores = val_records['macro_auroc']

            if args.model_select == 'val' and val_loader is not None:
                monitor(scores, raw_model, ckpt_name=os.path.join(model_dir, "checkpoint.pt"))
            elif args.model_select == 'last_epoch' and i == args.epochs - 1:
                torch.save(raw_model.state_dict(), os.path.join(model_dir, "checkpoint.pt"))

            save_training_state(i, "checkpoint_latest.pt")
            if int(args.save_interval_epochs) > 0 and ((i + 1) % int(args.save_interval_epochs) == 0):
                save_training_state(i, f"checkpoint_epoch_{i+1}.pt")

    if is_main:
        selected_ckpt = os.path.join(model_dir, "checkpoint.pt")
        if os.path.exists(selected_ckpt):
            raw_model.load_state_dict(torch.load(selected_ckpt))
        else:
            print(f"Selected checkpoint not found at {selected_ckpt}; evaluating current in-memory model.")
        eval_epoch = max(last_epoch_ran, 0)
        test_records = evaluate(test_loader, model, fp16_scaler, loss_fn, eval_epoch, args)
        log_dict = {'test_' + k: v for k, v in test_records.items() if 'prob' not in k and 'label' not in k}
        log_writer(log_dict, fold, args.report_to, writer)
        if mlflow_enabled:
            try:
                mlflow.log_metrics(_sanitize_mlflow_metrics(log_dict), step=int(args.epochs))
            except Exception as e:
                print(f"Warning: MLflow test metric logging failed: {e}")
        wandb.finish() if "wandb" in args.report_to else None
        if mlflow_enabled and mlflow_started_here:
            try:
                mlflow.end_run()
            except Exception:
                pass

    return val_records, test_records


def train_one_epoch(train_loader, model, fp16_scaler, optimizer, loss_fn, epoch, args):
    model.train()
    start_time = time.time()
    seq_len = 0
    records = get_records_array(len(train_loader), args.n_classes)
    is_main = getattr(args, 'rank', 0) == 0
    use_ddp_sync = getattr(args, 'world_size', 1) > 1

    for batch_idx, batch in enumerate(train_loader):
        if batch_idx % args.gc == 0 and args.lr_scheduler == 'cosine':
            adjust_learning_rate(optimizer, batch_idx / len(train_loader) + epoch, args)

        images, img_coords, label = batch['imgs'], batch['coords'], batch['labels']
        images = images.to(args.device, non_blocking=True)
        img_coords = img_coords.to(args.device, non_blocking=True)
        label = label.to(args.device, non_blocking=True).long()
        seq_len += images.shape[1]

        with torch.cuda.amp.autocast(dtype=torch.float16 if args.fp16 else torch.float32):
            logits = model(images, img_coords)
            if isinstance(loss_fn, torch.nn.BCEWithLogitsLoss):
                label = label.squeeze(-1).float()
            else:
                label = label.squeeze(-1).long()
            loss = loss_fn(logits, label)
            loss /= args.gc

        # Skip gradient sync on accumulation steps for efficiency
        is_accumulating = (batch_idx + 1) % args.gc != 0
        sync_ctx = model.no_sync if (use_ddp_sync and is_accumulating) else nullcontext
        with sync_ctx():
            if fp16_scaler is None:
                loss.backward()
            else:
                fp16_scaler.scale(loss).backward()

        if not is_accumulating:
            if fp16_scaler is None:
                optimizer.step()
                optimizer.zero_grad()
            else:
                fp16_scaler.step(optimizer)
                fp16_scaler.update()
                optimizer.zero_grad()

        records['loss'] += loss.item() * args.gc

        if is_main and (batch_idx + 1) % 20 == 0:
            time_per_it = (time.time() - start_time) / (batch_idx + 1)
            print('Epoch: {}, Batch: {}, Loss: {:.4f}, LR: {:.4f}, Time: {:.4f} sec/it, Seq len: {:.1f}, Slide ID: {}' \
                  .format(epoch, batch_idx, records['loss']/batch_idx, optimizer.param_groups[0]['lr'], time_per_it, \
                          seq_len/(batch_idx+1), batch['slide_id'][-1] if 'slide_id' in batch else 'None'))

    records['loss'] = records['loss'] / len(train_loader)
    if is_main:
        print('Epoch: {}, Loss: {:.4f}'.format(epoch, loss))
    return records


def evaluate(loader, model, fp16_scaler, loss_fn, epoch, args):
    model.eval()

    # set the evaluation records
    records = get_records_array(len(loader), args.n_classes)
    # get the task setting
    task_setting = args.task_config.get('setting', 'multi_class')
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            # load the batch and transform this batch
            images, img_coords, label = batch['imgs'], batch['coords'], batch['labels']
            images = images.to(args.device, non_blocking=True)
            img_coords = img_coords.to(args.device, non_blocking=True)
            label = label.to(args.device, non_blocking=True).long()

            with torch.cuda.amp.autocast(fp16_scaler is not None, dtype=torch.float16):
                # get the logits
                logits = model(images, img_coords)
                # get the loss
                if isinstance(loss_fn, torch.nn.BCEWithLogitsLoss):
                    label = label.squeeze(-1).float()
                else:
                    label = label.squeeze(-1).long()
                loss = loss_fn(logits, label)

            # update the records
            records['loss'] += loss.item()
            if task_setting == 'multi_label':
                Y_prob = torch.sigmoid(logits)
                records['prob'][batch_idx] = Y_prob.cpu().numpy()
                records['label'][batch_idx] = label.cpu().numpy()
            elif task_setting == 'multi_class' or task_setting == 'binary':
                Y_prob = torch.softmax(logits, dim=1).cpu()
                records['prob'][batch_idx] = Y_prob.numpy()
                # convert label to one-hot
                label_ = torch.zeros_like(Y_prob).scatter_(1, label.cpu().unsqueeze(1), 1)
                records['label'][batch_idx] = label_.numpy()

    records.update(calculate_metrics_with_task_cfg(records['prob'], records['label'], args.task_config))
    records['loss'] = records['loss'] / len(loader)

    if task_setting == 'multi_label':
        info = 'Epoch: {}, Loss: {:.4f}, Micro AUROC: {:.4f}, Macro AUROC: {:.4f}, Micro AUPRC: {:.4f}, Macro AUPRC: {:.4f}'.format(epoch, records['loss'], records['micro_auroc'], records['macro_auroc'], records['micro_auprc'], records['macro_auprc'])
    else:
        info = 'Epoch: {}, Loss: {:.4f}, AUROC: {:.4f}, ACC: {:.4f}, BACC: {:.4f}'.format(epoch, records['loss'], records['macro_auroc'], records['acc'], records['bacc'])
        for metric in args.task_config.get('add_metrics', []):
            info += ', {}: {:.4f}'.format(metric, records[metric])
    print(info)
    return records