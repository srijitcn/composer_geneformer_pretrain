from omegaconf import DictConfig

from cfgutils import *

import boto3
import pickle

from transformers import BertConfig, BertForMaskedLM, DataCollatorForLanguageModeling

import geneformer
from geneformer.pretrainer import GeneformerPreCollator

import torch
from torch.distributed.checkpoint import state_dict_loader
from torch.utils.data import DataLoader

from composer.utils.checkpoint import DistCPObjectStoreReader


from streaming import MDSWriter, StreamingDataset

def main(cfg: DictConfig):
    #load the model back and run some test
    working_dir = cfg.working_dir
    checkpoint_path = f"{cfg.save_folder}/ep10-ba10000"
    local_checkpoint_path = f"{working_dir}/checkpoint"
    data_bucket_name = cfg.data_bucket_name
    data_bucket_key = cfg.data_bucket_key
    token_dictionary_filename = cfg.token_dictionary_filename
    remote_data_dir = f"s3://{data_bucket_name}/{data_bucket_key}"
    streaming_dataset_location = cfg.streaming_dataset_location

    remote_streaming_dataset_location = f"{remote_data_dir}/{streaming_dataset_location}"
    streaming_dataset_cache_location = f"{working_dir}/streaming/cache"    
    mlm_probability = cfg.mlm_probability
    # Read the token dictionary file
    s3 = boto3.resource('s3')
    token_dictionary = pickle.loads(s3.Bucket(data_bucket_name).Object(f"{data_bucket_key}/{token_dictionary_filename}").get()['Body'].read())

    ### Load model
    print("Loading model")
    model_config = build_model_config(cfg,token_dictionary)

    config = BertConfig(**model_config)
    model = BertForMaskedLM(config)
    tokenizer = GeneformerPreCollator(token_dictionary=token_dictionary)
    
    ##load model weights
    print("Loading weights")
    state_dict = {
        "model": model.state_dict()
    }
    state_dict_loader.load(
        state_dict=state_dict,
        storage_reader= DistCPObjectStoreReader(source_path=checkpoint_path, destination_path=local_checkpoint_path)
    )
    model.load_state_dict(state_dict["model"])


    ##Run inference
    print("Getting test data")
    streaming_dataset_eval = StreamingDataset(remote=f"{remote_streaming_dataset_location}/test", local=f"{streaming_dataset_cache_location}/test" ,batch_size=eval_batch_size)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=True, 
        mlm_probability=mlm_probability
    )

    eval_dataloader = DataLoader(streaming_dataset_eval,
                            shuffle=False, 
                            drop_last=False, 
                            collate_fn=data_collator)
    
    
    test_data = next(iter(eval_dataloader))
    print(f"Test data: {test_data}")

if __name__ == '__main__':
    yaml_path, args_list = sys.argv[1], sys.argv[2:]
    with open(yaml_path) as f:
        yaml_cfg = om.load(f)
    cli_cfg = om.from_cli(args_list)
    cfg = om.merge(yaml_cfg, cli_cfg)
    cfg = cast(DictConfig, cfg)  # for type checking
    main(cfg)