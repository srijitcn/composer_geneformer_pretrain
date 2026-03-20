# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import time
import threading
from typing import Optional, cast

from composer import Trainer, algorithms
from composer.callbacks import ( LRMonitor, MemoryMonitor,
                                OptimizerMonitor, RuntimeEstimator,
                                SpeedMonitor)
from composer.loggers import WandBLogger, MLFlowLogger
from composer.optim import DecoupledAdamW
from composer.optim.scheduler import (ConstantWithWarmupScheduler,
                                      CosineAnnealingWithWarmupScheduler,
                                      LinearWithWarmupScheduler)
from composer.utils import dist, reproducibility

from omegaconf import DictConfig
from omegaconf import OmegaConf as om

def build_model_config(cfg:DictConfig, token_dictionary:dict):
    model_config = om.to_container(cfg.get("model_config",{}))
    model_config["pad_token_id"] = token_dictionary.get("<pad>")
    model_config["vocab_size"] = len(token_dictionary)
    return model_config

def log_config(cfg: DictConfig):
    print(om.to_yaml(cfg))
    if 'wandb' in cfg.get('loggers', {}):
        try:
            import wandb
        except ImportError as e:
            raise e
        if wandb.run:
            wandb.config.update(om.to_container(cfg, resolve=True))


def build_algorithm(name, kwargs):
    if name == 'gradient_clipping':
        return algorithms.GradientClipping(**kwargs)
    elif name == 'alibi':
        return algorithms.Alibi(**kwargs)
    elif name == 'fused_layernorm':
        return algorithms.FusedLayerNorm(**kwargs)
    elif name == 'gated_linear_units':
        return algorithms.GatedLinearUnits(**kwargs)
    elif name == 'low_precision_layernorm':
        return algorithms.LowPrecisionLayerNorm(**kwargs)
    else:
        raise ValueError(f'Not sure how to build algorithm: {name}')


def build_callback(name, kwargs):
    if name == 'lr_monitor':
        return LRMonitor()
    elif name == 'memory_monitor':
        return MemoryMonitor()
    elif name == 'speed_monitor':
        return SpeedMonitor(window_size=kwargs.get('window_size', 1),
                            gpu_flops_available=kwargs.get(
                                'gpu_flops_available', None))
    elif name == 'runtime_estimator':
        return RuntimeEstimator()
    elif name == 'optimizer_monitor':
        return OptimizerMonitor(log_optimizer_metrics=kwargs.get(
            'log_optimizer_metrics', True),)
    else:
        raise ValueError(f'Not sure how to build callback: {name}')


class HangingMLFlowLogger(MLFlowLogger):
    """
    Simulates ES-1788744: MLflow monitor.finish() hangs after training
    completes. Training succeeds, NCCL cleans up, execution summary prints,
    but the MLflow logger's close/flush never finishes — keeping the process
    alive and the cluster from being released.
    """

    def __init__(self, hang_duration_minutes: int = 180, **kwargs):
        super().__init__(**kwargs)
        self.hang_duration_minutes = hang_duration_minutes

    def close(self, state, logger):
        print("[MLflow Logger] Flushing remaining metrics...")

        def _emit_warnings():
            while True:
                time.sleep(30 * 60)
                print("[MLflow Logger][Warning] No new logs have been emitted in the last 30 minutes.")

        warning_thread = threading.Thread(target=_emit_warnings, daemon=True)
        warning_thread.start()

        # Simulate the hang — block for the configured duration
        time.sleep(self.hang_duration_minutes * 60)

        # If we ever get past the hang, call the real close
        super().close(state, logger)


def build_logger(name, kwargs):
    if name == 'wandb':
        return WandBLogger(**kwargs)
    elif name == 'mlflow':
        # Check if we should use the hanging variant
        simulate_hang = kwargs.pop('simulate_hang', False)
        hang_duration_minutes = kwargs.pop('hang_duration_minutes', 180)
        if simulate_hang:
            return HangingMLFlowLogger(
                hang_duration_minutes=hang_duration_minutes, **kwargs)
        return MLFlowLogger(**kwargs)
    else:
        raise ValueError(f'Not sure how to build logger: {name}')


def build_scheduler(cfg):
    if cfg.name == 'constant_with_warmup':
        return ConstantWithWarmupScheduler(t_warmup=cfg.t_warmup)
    elif cfg.name == 'cosine_with_warmup':
        return CosineAnnealingWithWarmupScheduler(t_warmup=cfg.t_warmup,
                                                  alpha_f=cfg.alpha_f)
    elif cfg.name == 'linear_decay_with_warmup':
        return LinearWithWarmupScheduler(t_warmup=cfg.t_warmup,
                                         alpha_f=cfg.alpha_f)
    else:
        raise ValueError(f'Not sure how to build scheduler: {cfg.name}')


def build_optimizer(cfg, model):
    if cfg.name == 'decoupled_adamw':
        # Convert betas to tuple to avoid omegaconf.ListConfig serialization issues during checkpointing
        betas = tuple(cfg.betas) if cfg.betas else (0.9, 0.999)
        return DecoupledAdamW(model.parameters(),
                              lr=cfg.lr,
                              betas=betas,
                              eps=cfg.eps,
                              weight_decay=cfg.weight_decay)
    else:
        raise ValueError(f'Not sure how to build optimizer: {cfg.name}')
