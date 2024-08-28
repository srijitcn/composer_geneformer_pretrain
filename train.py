

##1 Set parameters

import datetime

# imports
import os

import pickle
import random
import subprocess

import numpy as np
import pytz

import boto3

import torch
from torch.utils.data import DataLoader

from transformers import BertConfig, BertForMaskedLM, DataCollatorForLanguageModeling

import geneformer
from geneformer.pretrainer import GeneformerPreCollator

from composer.models.huggingface import HuggingFaceModel
from composer.utils import reproducibility
from composer import Trainer

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
    data_bucket_name = cfg.data_bucket_name
    data_bucket_key = cfg.data_bucket_key

    token_dictionary_filename = cfg.token_dictionary_filename
    remote_data_dir = f"s3://{data_bucket_name}/{data_bucket_key}"
    streaming_dataset_location = cfg.streaming_dataset_location

    # batch size for training and eval
    train_batch_size = cfg.train_batch_size
    eval_batch_size = cfg.eval_batch_size
    mlm_probability = cfg.mlm_probability

    remote_streaming_dataset_location = f"{remote_data_dir}/{streaming_dataset_location}"
    local_streaming_dataset_location = f"{cfg.local_data_dir}/{streaming_dataset_location}"
    streaming_dataset_cache_location = f"{working_dir}/streaming/cache"

    if cfg.data_location == "local":
        data_local = True
    else:
        data_local = False

    # output directories

    #############################################
    ### Start processing
    reproducibility.configure_deterministic_mode()
    reproducibility.seed_all(seed_val)

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
    # Read the token dictionary file
    s3 = boto3.resource('s3')
    token_dictionary = pickle.loads(s3.Bucket(data_bucket_name).Object(f"{data_bucket_key}/{token_dictionary_filename}").get()['Body'].read())

    ### Load model
    model_config = build_model_config(cfg,token_dictionary)

    print("=============================")
    print(model_config)

    config = BertConfig(**model_config)
    model = BertForMaskedLM(config)
    tokenizer = GeneformerPreCollator(token_dictionary=token_dictionary)
    model.train()
    print(model)

    #Create streaming dataset

    if data_local:
        streaming_dataset_train = StreamingDataset(local=f"{local_streaming_dataset_location}/train" ,batch_size=train_batch_size)
        streaming_dataset_eval = StreamingDataset(local=f"{local_streaming_dataset_location}/test" ,batch_size=eval_batch_size)        
    else:
        streaming_dataset_train = StreamingDataset(remote=f"{remote_streaming_dataset_location}/train", local=f"{streaming_dataset_cache_location}/train" ,batch_size=train_batch_size)
        streaming_dataset_eval = StreamingDataset(remote=f"{remote_streaming_dataset_location}/test", local=f"{streaming_dataset_cache_location}/test" ,batch_size=eval_batch_size)

    #Prepare composer model
    composer_model = HuggingFaceModel(model)

    # Build optimizer
    optimizer = build_optimizer(cfg.optimizer, model)

    # Scheduler
    scheduler = build_scheduler(cfg.scheduler)

    #data collator
    data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, 
            mlm=True, 
            mlm_probability=mlm_probability
        )

    train_dataloader = DataLoader(streaming_dataset_train,
                            shuffle=False, 
                            drop_last=False, 
                            collate_fn=data_collator)

    eval_dataloader = DataLoader(streaming_dataset_eval,
                            shuffle=False, 
                            drop_last=False, 
                            collate_fn=data_collator)

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
        save_num_checkpoints_to_keep=cfg.get("save_num_checkpoints_to_keep",1),
        train_subset_num_batches=cfg.get("train_subset_num_batches", -1),
        eval_subset_num_batches=cfg.get("eval_subset_num_batches", -1),
        save_overwrite=cfg.get("save_overwrite", False),
        load_path=cfg.get("load_path", None),
        #load_weights_only=cfg.get("load_weights_only", False),
        python_log_level=cfg.get("python_log_level", None),
        seed=seed_val,        
        fsdp_config = cfg.get("fsdp_config", None),
        loggers=loggers,
        callbacks=callbacks,
        # To resume from checkpoints in save_folder
        autoresume=cfg.get("autoresume", False),
    )
    # Start training
    trainer.fit(reset_time=cfg.get("reset_time", False))

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
