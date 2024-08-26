

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
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR

from transformers import BertConfig, BertForMaskedLM, DataCollatorForLanguageModeling

import geneformer
from geneformer.pretrainer import GeneformerPreCollator

from composer.models.huggingface import HuggingFaceModel
from composer.utils import reproducibility
from composer import Trainer

from streaming import MDSWriter, StreamingDataset

from omegaconf import DictConfig

from cfgutils import *


#### Env variables
#os.environ["NCCL_DEBUG"] = "INFO"
#os.environ["OMPI_MCA_opal_cuda_support"] = "true"
#os.environ["CONDA_OVERRIDE_GLIBC"] = "2.56"

def main(cfg: DictConfig):

    seed_val = cfg.seed_val
    random.seed(seed_val)
    np.random.seed(seed_val)

    working_dir = cfg.working_dir
    data_bucket_name = "srijit-nair-test-bucket"
    data_bucket_key = "geneformer/data"

    data_location = cfg.data_location

    if data_location == "s3":
        data_dir = f"s3://{data_bucket_name}/{data_bucket_key}"
        token_dictionary_filename = f"{data_bucket_key}/{cfg.token_dictionary_filename}"
        remote_data = True
    else:
        data_dir = cfg.local_data_dir
        token_dictionary_filename = f"{data_dir}/{cfg.token_dictionary_filename}"
        remote_data = False

    

    # batch size for training and eval
    train_batch_size = cfg.train_batch_size
    eval_batch_size = cfg.eval_batch_size
    mlm_probability = cfg.mlm_probability

    streaming_dataset_location = f"{data_dir}/streaming/genecorpus_30M_2048.dataset"
    streaming_dataset_cache_location = f"{working_dir}/streaming/cache"

    # output directories

    #############################################
    ### Start processing
    reproducibility.configure_deterministic_mode()
    reproducibility.seed_all(seed_val)

    # Read the token dictionary file
    if remote_data:
        s3 = boto3.resource('s3')
        token_dictionary = pickle.loads(s3.Bucket(data_bucket_name).Object(f"{data_bucket_key}/{token_dictionary_filename}").get()['Body'].read())
    else:
        with open(token_dictionary_filename, "rb") as f:
            token_dictionary = pickle.load(f)

    ### Load model
    model_config = build_model_config(cfg)
    model_config["pad_token_id"] = token_dictionary.get("<pad>")
    model_config["vocab_size"] = len(token_dictionary)

    config = BertConfig(**model_config)
    model = BertForMaskedLM(config)
    tokenizer = GeneformerPreCollator(token_dictionary=token_dictionary)
    model.train()
    print(model)

    #Create streaming dataset
    if remote_data:
        streaming_dataset_train = StreamingDataset(remote=f"{streaming_dataset_location}/train", local=f"{streaming_dataset_cache_location}/train" ,batch_size=train_batch_size)
        streaming_dataset_eval = StreamingDataset(remote=f"{streaming_dataset_location}/test", local=f"{streaming_dataset_cache_location}/test" ,batch_size=eval_batch_size)
    else:
        streaming_dataset_train = StreamingDataset(local=f"{streaming_dataset_location}/train" ,batch_size=train_batch_size)
        streaming_dataset_eval = StreamingDataset(local=f"{streaming_dataset_location}/test" ,batch_size=eval_batch_size)

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
        run_name=cfg.run_name,
        model=composer_model, 
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
        load_weights_only=cfg.get("load_weights_only", False),
        python_log_level=cfg.get("python_log_level", None),
        seed=seed_val,        
        fsdp_config = cfg.get("fsdp_config", None)
    )
    # Start training
    #trainer.fit()

    #print(trainer.state.train_metrics)
    #print(trainer.state.eval_metrics)

    #load the model back and run some test

    print("*************Done")


if __name__ == '__main__':
    yaml_path, args_list = sys.argv[1], sys.argv[2:]
    with open(yaml_path) as f:
        yaml_cfg = om.load(f)
    cli_cfg = om.from_cli(args_list)
    cfg = om.merge(yaml_cfg, cli_cfg)
    cfg = cast(DictConfig, cfg)  # for type checking
    main(cfg)
