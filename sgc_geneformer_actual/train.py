

##1 Set parameters

import datetime

# imports
import os

import pickle
import random
import subprocess

import numpy as np
import pytz

# import boto3  # Commented out - using local volume path instead of S3

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

from cfgutils import *




def main(cfg: DictConfig):
    #### Env variables
    #os.environ["NCCL_DEBUG"] = "INFO"

    seed_val = cfg.seed_val
    random.seed(seed_val)
    np.random.seed(seed_val)

    working_dir = cfg.working_dir
    # data_bucket_name = cfg.data_bucket_name  # Commented out - using local volume path
    # data_bucket_key = cfg.data_bucket_key  # Commented out - using local volume path

    token_dictionary_filename = cfg.token_dictionary_filename
    # remote_data_dir = f"s3://{data_bucket_name}/{data_bucket_key}"  # Commented out - using local volume path
    streaming_dataset_location = cfg.streaming_dataset_location

    # Use Databricks volume path instead of S3
    volume_base_path = cfg.get("volume_base_path", "/Volumes/main/srijit_nair/geneformer/data")

    # batch size for training and eval
    train_batch_size = cfg.train_batch_size  #<< This is per device batch size
    eval_batch_size = cfg.eval_batch_size
    mlm_probability = cfg.mlm_probability

    # remote_streaming_dataset_location = f"{remote_data_dir}/{streaming_dataset_location}"  # Commented out
    # local_streaming_dataset_location = f"{cfg.local_data_dir}/{streaming_dataset_location}"  # Commented out
    local_streaming_dataset_location = f"{volume_base_path}/dataset/{streaming_dataset_location}"
    streaming_dataset_cache_location = f"{working_dir}/streaming/cache"

    # Always use local volume path
    data_local = True
    # if cfg.data_location == "local":
    #     data_local = True
    # else:
    #     data_local = False

    # output directories

    #############################################
    ### Start processing
    reproducibility.configure_deterministic_mode()
    reproducibility.seed_all(seed_val)

    # Quick visibility into GPUs and distributed env.
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

    # Algorithms
    algorithms = [
        build_algorithm(name, algorithm_cfg)
        for name, algorithm_cfg in cfg.get('algorithms', {}).items()
    ]
    # Read the token dictionary file from local volume path (instead of S3)
    # s3 = boto3.resource('s3')
    # token_dictionary = pickle.loads(s3.Bucket(data_bucket_name).Object(f"{data_bucket_key}/{token_dictionary_filename}").get()['Body'].read())
    token_dictionary_path = f"{volume_base_path}/{token_dictionary_filename}"
    with open(token_dictionary_path, 'rb') as f:
        token_dictionary = pickle.load(f)
    print(f"Loaded token dictionary from: {token_dictionary_path}")

    ### Load model
    model_config = build_model_config(cfg,token_dictionary)

    print("=============================")
    print(model_config)

    config = BertConfig(**model_config)
    model = BertForMaskedLM(config)
    tokenizer = GeneformerPreCollator(token_dictionary=token_dictionary)
    model.train()
    print(model)

    # Initialize distributed training before creating StreamingDataset
    # This prevents timeout issues when StreamingDataset tries to init distributed
    dist.initialize_dist(device=cfg.get("device", "gpu"))
    world_size = dist.get_world_size()
    global_rank = dist.get_global_rank()
    print(f"Distributed initialized: world_size={world_size}, rank={global_rank}")
    
    # Add a barrier here to ensure all ranks are synchronized before StreamingDataset
    # import torch.distributed as torch_dist
    # if torch_dist.is_initialized():
    #     torch_dist.barrier()
    #     print(f"Rank {global_rank}: Passed initial barrier, proceeding to StreamingDataset")

    #Create streaming dataset - using local Databricks volume path
    print(f"Loading streaming dataset from: {local_streaming_dataset_location}")
    
    # For multi-node training, calculate num_canonical_nodes (number of physical nodes)
    # Assuming 8 GPUs per node for H100 clusters
    gpus_per_node = 8
    num_canonical_nodes = max(1, world_size // gpus_per_node)
    print(f"Using num_canonical_nodes={num_canonical_nodes} for StreamingDataset")
    
    streaming_dataset_train = StreamingDataset(
        local=f"{local_streaming_dataset_location}/train",
        batch_size=train_batch_size,
        num_canonical_nodes=num_canonical_nodes,
        shuffle=True,  # Enable shuffling for training
    )
    streaming_dataset_eval = StreamingDataset(
        local=f"{local_streaming_dataset_location}/test",
        batch_size=eval_batch_size,
        num_canonical_nodes=num_canonical_nodes,
        shuffle=False,  # No shuffling for eval
    )        
    # Commented out S3 remote option:
    # if data_local:
    #     streaming_dataset_train = StreamingDataset(local=f"{local_streaming_dataset_location}/train" ,batch_size=train_batch_size)
    #     streaming_dataset_eval = StreamingDataset(local=f"{local_streaming_dataset_location}/test" ,batch_size=eval_batch_size)        
    # else:
    #     streaming_dataset_train = StreamingDataset(remote=f"{remote_streaming_dataset_location}/train", local=f"{streaming_dataset_cache_location}/train" ,batch_size=train_batch_size)
    #     streaming_dataset_eval = StreamingDataset(remote=f"{remote_streaming_dataset_location}/test", local=f"{streaming_dataset_cache_location}/test" ,batch_size=eval_batch_size)

    #Prepare composer model
    composer_model = HuggingFaceModel(model)

    # Build optimizer
    optimizer = build_optimizer(cfg.optimizer, model)

    # Scheduler
    scheduler = build_scheduler(cfg.scheduler)

    #data collator - wrapped to convert input_ids to Long dtype
    base_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, 
            mlm=True, 
            mlm_probability=mlm_probability
        )
    
    def data_collator(features):
        """Wrapper collator that ensures input_ids are Long dtype before MLM masking."""
        # Convert input_ids to Long dtype if they are float
        for feature in features:
            if 'input_ids' in feature:
                if isinstance(feature['input_ids'], torch.Tensor):
                    feature['input_ids'] = feature['input_ids'].long()
                elif isinstance(feature['input_ids'], np.ndarray):
                    feature['input_ids'] = feature['input_ids'].astype(np.int64)
        return base_collator(features)

    train_dataloader = DataLoader(streaming_dataset_train,
                            shuffle=False, 
                            drop_last=False, 
                            collate_fn=data_collator,
                            batch_size=train_batch_size,
                            num_workers = 32,
                            pin_memory = True,
                            persistent_workers = True)

    eval_dataloader = DataLoader(streaming_dataset_eval,
                            shuffle=False, 
                            drop_last=False, 
                            collate_fn=data_collator,
                            batch_size=eval_batch_size,
                            num_workers = 32,
                            pin_memory = True,
                            persistent_workers = True)

    ##############################
    #Following code is to introduce an error after 7 epochs , 
    # to see if we can restart the training from 5th epoch
    #
    #class RaiseErrorOnEpoch7(Callback):
    #    def run_event(self, event: Event, state: State, logger: Logger):
    #        if event == Event.EPOCH_START and state.timestamp.epoch==7:
    #            raise Exception("Rescue me!!!!")
            
    #callbacks.append(RaiseErrorOnEpoch7())
    ##############################

    # Create Trainer Object
    trainer = Trainer(
        #run_name=cfg.run_name,
        model=composer_model, 
        algorithms=algorithms,
        train_dataloader=train_dataloader,    
        eval_dataloader=eval_dataloader,
        max_duration=cfg.max_duration,
        eval_interval=cfg.eval_interval,
        optimizers=optimizer,
        schedulers=[scheduler],
        device=cfg.get("device", "gpu"),
        device_train_microbatch_size=cfg.get("device_train_microbatch_size","auto"),
        save_folder=cfg.get("save_folder", None),
        save_interval=cfg.get("save_interval", "5ep"),
        save_overwrite=cfg.get("save_overwrite", False),
        save_num_checkpoints_to_keep=cfg.get("save_num_checkpoints_to_keep",1),
        train_subset_num_batches=cfg.get("train_subset_num_batches", -1),
        eval_subset_num_batches=cfg.get("eval_subset_num_batches", -1),
        autoresume=cfg.get("autoresume", False),
        #Load path required only for manual restarts
        #load_path=cfg.get("load_path", None),
        #load_weights_only=cfg.get("load_weights_only", False),
        python_log_level=cfg.get("python_log_level", None),
        seed=seed_val,        
        fsdp_config = cfg.get("fsdp_config", None),
        loggers=loggers,
        callbacks=callbacks,

    )
    # Start training; reset_time allows continued training even if elapsed time
    # already equals the configured max_duration (e.g., after a resume).
    trainer.fit(reset_time=True)

    print(trainer.state.train_metrics)
    print(trainer.state.eval_metrics)



    print("*************Done")


if __name__ == '__main__':
    yaml_path, args_list = sys.argv[1], sys.argv[2:]
    with open(yaml_path) as f:
        yaml_cfg = om.load(f)
    cli_cfg = om.from_cli(args_list)
    cfg = om.merge(yaml_cfg, cli_cfg)
    cfg = cast(DictConfig, cfg)  # for type checking
    main(cfg)
