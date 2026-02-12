import os
import sys
import re
import json
import glob
from pathlib import Path

# For convinience
this_file_dir = Path(__file__).resolve().parent
sys.path.append(str(this_file_dir.parent))

import time
import wandb
import torch
import numpy as np
import torch.utils.tensorboard as tensorboard
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
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
    is_main = getattr(args, "is_main_process", True)
    distributed = getattr(args, "distributed", False)
    fold_save_dir = os.path.join(args.save_dir, f'fold_{fold}')
    model_dir = os.path.join(args.save_dir, "model", f'fold_{fold}')
    os.makedirs(model_dir, exist_ok=True)
    checkpoint_index_path = os.path.join(model_dir, "checkpoint_index.json")

    def save_training_state(epoch: int, filename: str):
        if not is_main:
            return
        ckpt_path = os.path.join(model_dir, filename)
        model_to_save = model.module if hasattr(model, "module") else model
        state = {
            "epoch": int(epoch),
            "epoch_1_based": int(epoch) + 1,
            "model_state_dict": model_to_save.state_dict(),
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
    writer = tensorboard.SummaryWriter(writer_dir, flush_secs=15) if is_main else None
    mlflow_enabled = is_main and _is_mlflow_enabled() and (mlflow is not None)
    mlflow_started_here = False
    if mlflow_enabled:
        try:
            if mlflow.active_run() is None:
                mlflow.start_run(run_name=args.exp_code)
                mlflow_started_here = True
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
            })
        except Exception as e:
            print(f"Warning: MLflow setup failed, continue without MLflow logging. Error: {e}")
            mlflow_enabled = False
    # set up writer
    if is_main and "wandb" in args.report_to:
        wandb.init(
            project=args.exp_code,
            name=args.exp_code + '_fold_' + str(fold),
            id='fold_' + str(fold),
            tags=[],
            config=vars(args),
        )
        writer = wandb
    elif is_main and "tensorboard" in args.report_to:
        writer = tensorboard.SummaryWriter(writer_dir, flush_secs=15)

    # set up the model
    model = get_model(**vars(args))
    model = model.to(args.device)
    if distributed:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)
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
                if hasattr(model, "module"):
                    model.module.load_state_dict(ckpt["model_state_dict"])
                else:
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
        if distributed and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(i)
        if is_main:
            print('Epoch: {}'.format(i))
        train_records = train_one_epoch(train_loader, model, fp16_scaler, optimizer, loss_fn, i, args)

        if val_loader is not None:
            val_records = evaluate(val_loader, model, fp16_scaler, loss_fn, i, args)

            # update the writer for train and val
            log_dict = {'train_' + k: v for k, v in train_records.items() if 'prob' not in k and 'label' not in k}
            log_dict.update({'val_' + k: v for k, v in val_records.items() if 'prob' not in k and 'label' not in k})
            log_writer(log_dict, i, args.report_to, writer)
            if mlflow_enabled:
                try:
                    mlflow.log_metrics(_sanitize_mlflow_metrics(log_dict), step=i)
                except Exception as e:
                    print(f"Warning: MLflow metric logging failed at epoch {i}: {e}")
            # update the monitor scores
            scores = val_records['macro_auroc']

        if args.model_select == 'val' and val_loader is not None:
            model_to_save = model.module if hasattr(model, "module") else model
            monitor(scores, model_to_save, ckpt_name=os.path.join(model_dir, "checkpoint.pt"))
        elif args.model_select == 'last_epoch' and i == args.epochs - 1:
            if is_main:
                model_to_save = model.module if hasattr(model, "module") else model
                torch.save(model_to_save.state_dict(), os.path.join(model_dir, "checkpoint.pt"))

        # Always keep a rolling latest checkpoint for recovery/debug.
        save_training_state(i, "checkpoint_latest.pt")
        # Optional periodic checkpoints, e.g. every 5 epochs.
        if int(args.save_interval_epochs) > 0 and ((i + 1) % int(args.save_interval_epochs) == 0):
            save_training_state(i, f"checkpoint_epoch_{i+1}.pt")
        if distributed:
            dist.barrier()

    # load model for test (prefer selected checkpoint if available).
    selected_ckpt = os.path.join(model_dir, "checkpoint.pt")
    if os.path.exists(selected_ckpt):
        model_to_load = model.module if hasattr(model, "module") else model
        model_to_load.load_state_dict(torch.load(selected_ckpt))
    elif is_main:
        print(f"Selected checkpoint not found at {selected_ckpt}; evaluating current in-memory model.")
    # test the model
    eval_epoch = max(last_epoch_ran, 0)
    test_records = evaluate(test_loader, model, fp16_scaler, loss_fn, eval_epoch, args) if test_loader is not None else None
    # update the writer for test
    if test_records is not None:
        log_dict = {'test_' + k: v for k, v in test_records.items() if 'prob' not in k and 'label' not in k}
        log_writer(log_dict, fold, args.report_to, writer)
        if mlflow_enabled:
            try:
                mlflow.log_metrics(_sanitize_mlflow_metrics(log_dict), step=int(args.epochs))
            except Exception as e:
                print(f"Warning: MLflow test metric logging failed: {e}")
    if is_main and "wandb" in args.report_to:
        wandb.finish()
    if mlflow_enabled and mlflow_started_here:
        try:
            mlflow.end_run()
        except Exception:
            pass

    return val_records, test_records


def train_one_epoch(train_loader, model, fp16_scaler, optimizer, loss_fn, epoch, args):
    model.train()
    # set the start time
    start_time = time.time()

    # monitoring sequence length
    seq_len = 0

    # setup the records
    records = get_records_array(len(train_loader), args.n_classes)

    for batch_idx, batch in enumerate(train_loader):
        # we use a per iteration lr scheduler
        if batch_idx % args.gc == 0 and args.lr_scheduler == 'cosine':
            adjust_learning_rate(optimizer, batch_idx / len(train_loader) + epoch, args)

        # load the batch and transform this batch
        images, img_coords, label = batch['imgs'], batch['coords'], batch['labels']
        images = images.to(args.device, non_blocking=True)
        img_coords = img_coords.to(args.device, non_blocking=True)
        label = label.to(args.device, non_blocking=True).long()

        # add the sequence length
        seq_len += images.shape[1]

        with torch.cuda.amp.autocast(dtype=torch.float16 if args.fp16 else torch.float32):

            # get the logits
            logits = model(images, img_coords)
            # get the loss
            if isinstance(loss_fn, torch.nn.BCEWithLogitsLoss):
                label = label.squeeze(-1).float()
            else:
                label = label.squeeze(-1).long()

            loss = loss_fn(logits, label)
            loss /= args.gc

            if fp16_scaler is None:
                loss.backward()
                # update the parameters with gradient accumulation
                if (batch_idx + 1) % args.gc == 0:
                    optimizer.step()
                    optimizer.zero_grad()
            else:
                fp16_scaler.scale(loss).backward()
                # update the parameters with gradient accumulation
                if (batch_idx + 1) % args.gc == 0:
                    fp16_scaler.step(optimizer)
                    fp16_scaler.update()
                    optimizer.zero_grad()

        # update the records
        records['loss'] += loss.item() * args.gc

        if (batch_idx + 1) % 20 == 0 and getattr(args, "is_main_process", True):
            time_per_it = (time.time() - start_time) / (batch_idx + 1)
            print('Epoch: {}, Batch: {}, Loss: {:.4f}, LR: {:.4f}, Time: {:.4f} sec/it, Seq len: {:.1f}, Slide ID: {}' \
                  .format(epoch, batch_idx, records['loss']/batch_idx, optimizer.param_groups[0]['lr'], time_per_it, \
                          seq_len/(batch_idx+1), batch['slide_id'][-1] if 'slide_id' in batch else 'None'))

    records['loss'] = records['loss'] / len(train_loader)
    if getattr(args, "is_main_process", True):
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