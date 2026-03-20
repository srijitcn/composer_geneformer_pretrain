##1 Set parameters

import datetime
import json
import os
import pickle
import random
import subprocess
import sys
from typing import cast

import numpy as np
import pytz

import torch
from torch.utils.data import DataLoader

from transformers import BertConfig, BertForMaskedLM, DataCollatorForLanguageModeling

import geneformer
from geneformer.pretrainer import GeneformerPreCollator

from composer.models.huggingface import HuggingFaceModel
from composer.utils import reproducibility, dist
from composer import Trainer
from composer import Callback, Event, Logger, State

from streaming import StreamingDataset

from omegaconf import DictConfig
from omegaconf import OmegaConf as om

from cfgutils import *


# ============================================
# Failure Testing Callback
# ============================================
class FailureTestCallback(Callback):
    """
    A callback that intentionally fails training at a specified epoch to test 
    checkpoint recovery and autoresume functionality.
    
    The callback tracks the number of failures using a persistent file and only
    fails up to `max_failures` times. After that, training continues normally.
    
    IMPORTANT: In distributed training, only rank 0 manages the failure counter
    to avoid race conditions and double-counting across nodes.
    
    Args:
        fail_at_epoch: The epoch number at which to trigger a failure (0-indexed)
        max_failures: Maximum number of times to fail before allowing training to continue
        failure_counter_path: Path to file that tracks failure count across restarts
        enabled: Whether the failure testing is enabled
    
    Example usage in parameters.yaml:
        failure_test:
          enabled: true
          fail_at_epoch: 7
          max_failures: 3
          failure_counter_path: /Volumes/.../failure_counter.json
    """
    
    def __init__(
        self,
        fail_at_epoch: int = 7,
        max_failures: int = 3,
        failure_counter_path: str = "/tmp/failure_counter.json",
        enabled: bool = False,
    ):
        self.fail_at_epoch = fail_at_epoch
        self.max_failures = max_failures
        self.failure_counter_path = failure_counter_path
        self.enabled = enabled
        self._should_fail = False  # Will be set by rank 0 and used by all ranks
        self._new_count = 0
        
    def _is_rank_zero(self) -> bool:
        """Check if this is the main process (rank 0)."""
        return dist.get_global_rank() == 0
    
    def _log_rank_zero(self, message: str):
        """Print message only on rank 0 to avoid duplicate logs."""
        if self._is_rank_zero():
            print(message)
    
    def _get_failure_count(self) -> int:
        """Read the current failure count from the counter file."""
        if os.path.exists(self.failure_counter_path):
            try:
                with open(self.failure_counter_path, 'r') as f:
                    data = json.load(f)
                    return data.get('failure_count', 0)
            except (json.JSONDecodeError, IOError):
                return 0
        return 0
    
    def _increment_failure_count(self) -> int:
        """Increment and persist the failure count. Returns the new count. Only call from rank 0."""
        current_count = self._get_failure_count()
        new_count = current_count + 1
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.failure_counter_path), exist_ok=True)
        
        with open(self.failure_counter_path, 'w') as f:
            json.dump({
                'failure_count': new_count,
                'last_failure_epoch': self.fail_at_epoch,
                'max_failures': self.max_failures,
            }, f, indent=2)
        
        return new_count
    
    def _reset_failure_count(self):
        """Reset the failure counter (call after successful training completion). Only call from rank 0."""
        if os.path.exists(self.failure_counter_path):
            os.remove(self.failure_counter_path)
            print(f"🔄 Failure counter reset: {self.failure_counter_path}")
    
    def _print_status(self):
        """Print the failure test status. Only call from rank 0."""
        current_count = self._get_failure_count()
        print(f"\n{'='*60}")
        print("⚠️  FAILURE TEST MODE ENABLED")
        print(f"{'='*60}")
        print(f"  - Will fail at epoch: {self.fail_at_epoch}")
        print(f"  - Max failures: {self.max_failures}")
        print(f"  - Counter file: {self.failure_counter_path}")
        print(f"  - Current failure count: {current_count}")
        
        if current_count >= self.max_failures:
            print(f"  - Status: Already failed {current_count} times, will NOT fail again")
        else:
            print(f"  - Status: Will fail {self.max_failures - current_count} more time(s)")
        print(f"{'='*60}\n")
    
    def run_event(self, event: Event, state: State, logger: Logger):
        """Called at each training event."""
        if not self.enabled:
            return
        
        # Print status at the start of training (only on rank 0)
        if event == Event.FIT_START:
            if self._is_rank_zero():
                self._print_status()
        
        # Check at epoch start
        if event == Event.EPOCH_START:
            current_epoch = int(state.timestamp.epoch)
            
            if current_epoch == self.fail_at_epoch:
                # Rank 0 makes the decision and increments counter if needed
                should_fail = False
                new_count = 0
                
                if self._is_rank_zero():
                    failure_count = self._get_failure_count()
                    
                    if failure_count < self.max_failures:
                        new_count = self._increment_failure_count()
                        should_fail = True
                    else:
                        should_fail = False
                        print(f"\n{'='*60}")
                        print(f"✅ FAILURE TEST: Skipping failure (already failed {failure_count} times)")
                        print(f"   Training will continue normally from checkpoint")
                        print(f"{'='*60}\n")
                
                # Broadcast decision from rank 0 to all ranks using torch.distributed
                # This ensures all ranks make the same decision
                should_fail_tensor = torch.tensor([1 if should_fail else 0], dtype=torch.int64, device='cuda')
                torch.distributed.broadcast(should_fail_tensor, src=0)
                should_fail = should_fail_tensor.item() == 1
                
                # Broadcast the new count for the error message
                new_count_tensor = torch.tensor([new_count], dtype=torch.int64, device='cuda')
                torch.distributed.broadcast(new_count_tensor, src=0)
                new_count = new_count_tensor.item()
                
                if should_fail:
                    # All ranks raise the same error together
                    rank = dist.get_global_rank()
                    if rank == 0:
                        error_msg = (
                            f"\n{'='*60}\n"
                            f"💥 INTENTIONAL FAILURE (Test Mode)\n"
                            f"{'='*60}\n"
                            f"  Epoch: {current_epoch}\n"
                            f"  Failure count: {new_count}/{self.max_failures}\n"
                            f"  Remaining failures: {self.max_failures - new_count}\n"
                            f"{'='*60}\n"
                            f"This failure is intentional to test checkpoint recovery.\n"
                            f"The job should restart and resume from the last checkpoint.\n"
                            f"{'='*60}"
                        )
                    else:
                        error_msg = f"Intentional failure (rank {rank}, synced with rank 0)"
                    raise RuntimeError(error_msg)
        
        # Reset counter when training completes successfully (only on rank 0)
        if event == Event.FIT_END:
            if self._is_rank_zero():
                self._reset_failure_count()


def build_volume_path(cfg: DictConfig) -> str:
    """Build the Databricks volume path from catalog, schema, and volume_name."""
    volume_cfg = cfg.volume
    return f"/Volumes/{volume_cfg.catalog}/{volume_cfg.schema}/{volume_cfg.volume_name}"


def get_data_paths(cfg: DictConfig, volume_path: str) -> dict:
    """Get all data paths based on volume path and config."""
    data_cfg = cfg.data
    return {
        'source_dataset': f"{volume_path}/{data_cfg.source_dataset}",
        'streaming_dataset': f"{volume_path}/{data_cfg.streaming_dataset}",
        'token_dictionary': f"{volume_path}/{data_cfg.token_dictionary}",
        'train_dir': f"{volume_path}/{data_cfg.streaming_dataset}/train",
        'test_dir': f"{volume_path}/{data_cfg.streaming_dataset}/test",
    }


def verify_data_exists(paths: dict):
    """
    Verify that all required data exists.
    Raises FileNotFoundError with instructions if data is missing.
    """
    errors = []
    
    # Check token dictionary
    if not os.path.exists(paths['token_dictionary']):
        errors.append(f"Token dictionary not found: {paths['token_dictionary']}")
    
    # Check train MDS directory and index
    train_index = os.path.join(paths['train_dir'], 'index.json')
    if not os.path.exists(train_index):
        errors.append(f"Train MDS dataset not found: {paths['train_dir']}")
    
    # Check test MDS directory and index
    test_index = os.path.join(paths['test_dir'], 'index.json')
    if not os.path.exists(test_index):
        errors.append(f"Test MDS dataset not found: {paths['test_dir']}")
    
    if errors:
        error_list = "\n".join([f"  - {e}" for e in errors])
        raise FileNotFoundError(
            f"\n{'='*60}\n"
            f"DATA NOT FOUND\n"
            f"{'='*60}\n"
            f"The following required data is missing:\n{error_list}\n\n"
            f"Please run data_preparation.py notebook first to:\n"
            f"  1. Download the dataset from HuggingFace\n"
            f"  2. Download the token dictionary\n"
            f"  3. Convert to MDS format\n\n"
            f"Run the notebook with a CPU cluster before starting GPU training.\n"
            f"{'='*60}"
        )


def main(cfg: DictConfig):
    # Build volume path from config
    volume_path = build_volume_path(cfg)
    print(f"Using Databricks volume: {volume_path}")
    
    # Get all data paths
    paths = get_data_paths(cfg, volume_path)
    print(f"Token dictionary: {paths['token_dictionary']}")
    print(f"Streaming dataset: {paths['streaming_dataset']}")
    
    # Build checkpoint save folder from volume config
    checkpoint_folder = f"{volume_path}/{cfg.checkpoints.folder}"
    print(f"Checkpoint folder: {checkpoint_folder}")
    
    # Seed and reproducibility
    seed_val = cfg.seed_val
    random.seed(seed_val)
    np.random.seed(seed_val)
    
    working_dir = cfg.working_dir
    
    # batch size for training and eval
    train_batch_size = cfg.train_batch_size
    eval_batch_size = cfg.eval_batch_size
    mlm_probability = cfg.mlm_probability
    
    #############################################
    ### Start processing
    reproducibility.configure_deterministic_mode()
    reproducibility.seed_all(seed_val)

    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"[gpu-check] CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES')}")
        print(f"[gpu-check] torch.cuda.device_count()={gpu_count}")
        for i in range(gpu_count):
            p = torch.cuda.get_device_properties(i)
            print(f"[gpu-check] {i}: {p.name} {p.total_memory/1e9:.1f} GB")
    else:
        print("[gpu-check] CUDA not available")
    print(
        f"[gpu-check] NNODES={os.getenv('NNODES')} "
        f"NPROC_PER_NODE={os.getenv('NPROC_PER_NODE')} "
        f"WORLD_SIZE={os.getenv('WORLD_SIZE')}"
    )
    print(
        f"[gpu-check] MASTER_ADDR={os.getenv('MASTER_ADDR')} "
        f"MASTER_PORT={os.getenv('MASTER_PORT')} "
        f"NODE_RANK={os.getenv('NODE_RANK')}"
    )

    loggers = [
        build_logger(name, logger_cfg)
        for name, logger_cfg in cfg.get('loggers', {}).items()
    ]

    # Callbacks
    callbacks = [
        build_callback(name, callback_cfg)
        for name, callback_cfg in cfg.get('callbacks', {}).items()
    ]
    
    # Add FailureTestCallback if enabled in config
    failure_test_cfg = cfg.get('failure_test', {})
    if failure_test_cfg.get('enabled', False):
        # Default failure counter path in the checkpoint folder
        default_counter_path = f"{checkpoint_folder}/failure_counter.json"
        
        failure_callback = FailureTestCallback(
            fail_at_epoch=failure_test_cfg.get('fail_at_epoch', 7),
            max_failures=failure_test_cfg.get('max_failures', 3),
            failure_counter_path=failure_test_cfg.get('failure_counter_path', default_counter_path),
            enabled=True,
        )
        callbacks.append(failure_callback)

    # Algorithms
    algorithms = [
        build_algorithm(name, algorithm_cfg)
        for name, algorithm_cfg in cfg.get('algorithms', {}).items()
    ]
    
    # Read the token dictionary file
    print(f"\nLoading token dictionary from: {paths['token_dictionary']}")
    with open(paths['token_dictionary'], 'rb') as f:
        token_dictionary = pickle.load(f)
    print(f"Token dictionary loaded with {len(token_dictionary)} tokens")

    ### Load model
    model_config = build_model_config(cfg, token_dictionary)

    print("\n=============================")
    print("Model Configuration:")
    print(model_config)
    print("=============================\n")

    config = BertConfig(**model_config)
    model = BertForMaskedLM(config)
    tokenizer = GeneformerPreCollator(token_dictionary=token_dictionary)
    model.train()
    print(model)

    # Initialize distributed training before creating StreamingDataset
    dist.initialize_dist(device=cfg.get("device", "gpu"))
    world_size = dist.get_world_size()
    global_rank = dist.get_global_rank()
    print(f"\nDistributed initialized: world_size={world_size}, rank={global_rank}")
    
    # For multi-node training, calculate num_canonical_nodes (number of physical nodes)
    # Assuming 8 GPUs per node for H100 clusters
    gpus_per_node = 8
    num_canonical_nodes = max(1, world_size // gpus_per_node)
    print(f"Using num_canonical_nodes={num_canonical_nodes} for StreamingDataset")

    # Verify all required data exists (only on rank 0, then sync)
    if dist.get_global_rank() == 0:
        verify_data_exists(paths)
        print(f"\n✅ All data verified at: {paths['streaming_dataset']}")
    
    # Synchronize all ranks
    dist.barrier()

    # Create streaming dataset - using remote for source data and local for SSD cache
    print(f"\nLoading streaming dataset from: {paths['streaming_dataset']}")
    streaming_dataset_train = StreamingDataset(
        remote=paths['train_dir'],              # Read compressed data from here
        local="/local_disk0/streaming_cache/train",  # Cache decompressed data here (local SSD)
        batch_size=train_batch_size,
        num_canonical_nodes=num_canonical_nodes,
        shuffle=True,  # Enable shuffling for training
    )
    streaming_dataset_eval = StreamingDataset(
        remote=paths['test_dir'],               # Read compressed data from here
        local="/local_disk0/streaming_cache/test",   # Cache decompressed data here (local SSD)
        batch_size=eval_batch_size,
        num_canonical_nodes=num_canonical_nodes,
        shuffle=False,  # No shuffling for eval
    )

    # Prepare composer model
    composer_model = HuggingFaceModel(model)

    # Build optimizer
    optimizer = build_optimizer(cfg.optimizer, model)

    # Scheduler
    scheduler = build_scheduler(cfg.scheduler)

    # Data collator - wrapped to convert input_ids to Long dtype
    base_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=True, 
        mlm_probability=mlm_probability
    )
    
    def data_collator(features):
        """Wrapper collator that ensures input_ids are Long dtype before MLM masking."""
        for feature in features:
            if 'input_ids' in feature:
                if isinstance(feature['input_ids'], torch.Tensor):
                    feature['input_ids'] = feature['input_ids'].long()
                elif isinstance(feature['input_ids'], np.ndarray):
                    feature['input_ids'] = feature['input_ids'].astype(np.int64)
        return base_collator(features)

    train_dataloader = DataLoader(
        streaming_dataset_train,
        shuffle=False, 
        drop_last=False, 
        collate_fn=data_collator,
        batch_size=train_batch_size,
        num_workers=32,
        pin_memory=True,
        persistent_workers=True
    )

    eval_dataloader = DataLoader(
        streaming_dataset_eval,
        shuffle=False, 
        drop_last=False, 
        collate_fn=data_collator,
        batch_size=eval_batch_size,
        num_workers=32,
        pin_memory=True,
        persistent_workers=True
    )

    # Create Trainer Object
    trainer = Trainer(
        model=composer_model, 
        algorithms=algorithms,
        train_dataloader=train_dataloader,    
        eval_dataloader=eval_dataloader,
        max_duration=cfg.max_duration,
        eval_interval=cfg.eval_interval,
        optimizers=optimizer,
        schedulers=[scheduler],
        device=cfg.get("device", "gpu"),
        device_train_microbatch_size=cfg.get("device_train_microbatch_size", "auto"),
        save_folder=checkpoint_folder,  # Use volume-based checkpoint folder
        save_interval=cfg.get("save_interval", "5ep"),
        save_overwrite=cfg.get("save_overwrite", False),
        save_num_checkpoints_to_keep=cfg.get("save_num_checkpoints_to_keep", 1),
        train_subset_num_batches=cfg.get("train_subset_num_batches", -1),
        eval_subset_num_batches=cfg.get("eval_subset_num_batches", -1),
        autoresume=cfg.get("autoresume", False),
        python_log_level=cfg.get("python_log_level", None),
        seed=seed_val,        
        parallelism_config={"fsdp": cfg.get("fsdp_config", None)},
        loggers=loggers,
        callbacks=callbacks,
    )
    
    # Start training
    trainer.fit()

    print(trainer.state.train_metrics)
    print(trainer.state.eval_metrics)

    print("*************Done")


if __name__ == '__main__':
    yaml_path, args_list = sys.argv[1], sys.argv[2:]
    with open(yaml_path) as f:
        yaml_cfg = om.load(f)
    cli_cfg = om.from_cli(args_list)
    cfg = om.merge(yaml_cfg, cli_cfg)
    cfg = cast(DictConfig, cfg)
    main(cfg)
