from omegaconf import DictConfig

from cfgutils import *

import boto3
import pickle

from transformers import BertConfig, BertForMaskedLM, DataCollatorForLanguageModeling

import geneformer
from geneformer.pretrainer import GeneformerPreCollator

import torch
import torch.distributed.checkpoint as dcp
from torch.utils.data import DataLoader
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

from composer.utils.dist import initialize_dist
from composer.utils.checkpoint import DistCPObjectStoreReader
from composer.utils import S3ObjectStore

from streaming import MDSWriter, StreamingDataset

def main(cfg: DictConfig):
    #load the model back and run some test
    working_dir = cfg.working_dir
    checkpoint_path = "s3://srijit-nair-sandbox-bucket/geneformer/pretrain/checkpoints/ep10-ba10000" #f"{cfg.save_folder}/ep10-ba10000"
    checkpoint_prefix = '/'.join(checkpoint_path.replace("s3://","").split('/')[1:])

    local_checkpoint_path = f"{working_dir}/checkpoint"
    data_bucket_name = cfg.data_bucket_name
    data_bucket_key = cfg.data_bucket_key
    token_dictionary_filename = cfg.token_dictionary_filename
    remote_data_dir = f"s3://{data_bucket_name}/{data_bucket_key}"
    streaming_dataset_location = cfg.streaming_dataset_location

    remote_streaming_dataset_location = f"{remote_data_dir}/{streaming_dataset_location}"
    streaming_dataset_cache_location = f"{working_dir}/streaming/cache"    
    mlm_probability = cfg.mlm_probability
    eval_batch_size = cfg.eval_batch_size
    # Read the token dictionary file
    s3 = boto3.resource('s3')
    token_dictionary = pickle.loads(s3.Bucket(data_bucket_name).Object(f"{data_bucket_key}/{token_dictionary_filename}").get()['Body'].read())

    ##Initialize dist
    initialize_dist(device="gpu")

    ### Load model
    print("Loading model")
    model_config = build_model_config(cfg,token_dictionary)

    config = BertConfig(**model_config)
    model = BertForMaskedLM(config)
    tokenizer = GeneformerPreCollator(token_dictionary=token_dictionary)
    
    ##load model weights
    print("Loading weights")    
    model_state_dict = model.state_dict()
    #st_dict = {
    #    "model" : model_state_dict
    #}
    st_dict = { f"model.{k}":v  for k,v in model_state_dict.items()}

    for k in st_dict.keys:
        print(k)
    
    dcp.load_state_dict(
        state_dict=st_dict,
        storage_reader= DistCPObjectStoreReader(
            source_path=checkpoint_prefix, 
            destination_path=local_checkpoint_path,
            device_mesh = None,
            object_store=S3ObjectStore(
                bucket = data_bucket_name
            )            
        )
    )

    #unnecessary plumbing work...argh
    st_dict=consume_prefix_in_state_dict_if_present(model_state_dict, prefix="model.")
    model.load_state_dict(st_dict)

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