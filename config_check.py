

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




def main(cfg: DictConfig):
    #### Env variables
    os.environ["NCCL_DEBUG"] = "INFO"
    #os.environ["OMPI_MCA_opal_cuda_support"] = "true"
    #os.environ["CONDA_OVERRIDE_GLIBC"] = "2.56"

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
    model_config = build_model_config(cfg)
    model_config["pad_token_id"] = token_dictionary.get("<pad>")
    model_config["vocab_size"] = len(token_dictionary)

    print("=============================")
    print(cfg)
    
    print("=============================")
    print(model_config)


    config = BertConfig(**model_config)
    model = BertForMaskedLM(config)
    tokenizer = GeneformerPreCollator(token_dictionary=token_dictionary)
 


if __name__ == '__main__':
    yaml_path, args_list = sys.argv[1], sys.argv[2:]
    with open(yaml_path) as f:
        yaml_cfg = om.load(f)
    cli_cfg = om.from_cli(args_list)
    cfg = om.merge(yaml_cfg, cli_cfg)
    cfg = cast(DictConfig, cfg)  # for type checking
    main(cfg)
