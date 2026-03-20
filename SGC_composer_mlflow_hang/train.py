"""
Minimal Composer job to reproduce ES-1788744:
MLflow monitor.finish() hangs after training completes successfully.

Uses synthetic random data — no downloads or external data needed.
Training finishes in ~30 seconds, then the MLflow close() hangs.
"""

import os
import time
import threading

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from composer import Trainer
from composer.models import ComposerClassifier
from composer.loggers import MLFlowLogger
from composer.callbacks import SpeedMonitor, LRMonitor


class HangingMLFlowLogger(MLFlowLogger):
    """Simulates ES-1788744: monitor.finish() hangs after training completes."""

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
        time.sleep(self.hang_duration_minutes * 60)
        super().close(state, logger)


class TinyModel(nn.Module):
    """Minimal model — trains instantly."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 10),
        )

    def forward(self, x):
        return self.net(x)


def main():
    print(f"[info] WORLD_SIZE={os.getenv('WORLD_SIZE')} NODE_RANK={os.getenv('NODE_RANK')}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"[gpu] {i}: {torch.cuda.get_device_name(i)}")

    # Synthetic data — 512 samples, 32-dim input, 10 classes
    train_dl = DataLoader(
        TensorDataset(torch.randn(512, 32), torch.randint(0, 10, (512,))),
        batch_size=64, shuffle=True,
    )
    eval_dl = DataLoader(
        TensorDataset(torch.randn(128, 32), torch.randint(0, 10, (128,))),
        batch_size=64,
    )

    model = ComposerClassifier(TinyModel())

    # Toggle hang via env var: SIMULATE_MLFLOW_HANG=true (default) or false
    simulate = os.getenv("SIMULATE_MLFLOW_HANG", "true").lower() == "true"
    hang_min = int(os.getenv("MLFLOW_HANG_MINUTES", "180"))

    if simulate:
        print(f"\n*** MLFLOW HANG SIMULATION ENABLED ({hang_min} min) ***\n")
        mlflow_logger = HangingMLFlowLogger(
            hang_duration_minutes=hang_min,
            tracking_uri="databricks",
            experiment_name="mlflow_experiments/composer_mlflow_hang_test",
        )
    else:
        mlflow_logger = MLFlowLogger(
            tracking_uri="databricks",
            experiment_name="mlflow_experiments/composer_mlflow_hang_test",
        )

    trainer = Trainer(
        model=model,
        train_dataloader=train_dl,
        eval_dataloader=eval_dl,
        max_duration="3ep",
        eval_interval="1ep",
        device="gpu",
        loggers=[mlflow_logger],
        callbacks=[SpeedMonitor(window_size=5), LRMonitor()],
        progress_bar=True,
    )

    trainer.fit()
    print(f"\nTrain metrics: {trainer.state.train_metrics}")
    print(f"Eval metrics: {trainer.state.eval_metrics}")
    print("*************Done")


if __name__ == "__main__":
    main()
