from cfgutils import *
import boto3
import pickle
import os
from transformers import BertConfig, BertForMaskedLM, DataCollatorForLanguageModeling

import geneformer
from geneformer.pretrainer import GeneformerPreCollator
from transformers import pipeline, PreTrainedTokenizerBase

import torch
import torch.distributed.checkpoint as dcp
from torch.utils.data import DataLoader
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

from composer.utils.dist import initialize_dist
from composer.utils.checkpoint import DistCPObjectStoreReader
from composer.utils import S3ObjectStore
from composer.models.huggingface import HuggingFaceModel
from composer import Trainer, Callback
from composer.loggers import Logger

from streaming import MDSWriter, StreamingDataset

import mlflow
from config import load_params

#https://docs.mosaicml.com/projects/composer/en/stable/api_reference/generated/composer.Trainer.html#composer.Trainer.predict
class PredictionSaver(Callback):
    def __init__(self, folder: str):
        self.folder = folder
        os.makedirs(self.folder, exist_ok=True)

def main(cfg):
    #load the model back and run some test
    working_dir = cfg.working_dir
    checkpoint_file = "s3://srijit-nair-sandbox-bucket/geneformer/pretrain/checkpoints_full/ep10-ba10000-rank0.pt" #f"{cfg.save_folder}/ep10-ba10000-rank0.pt"
    checkpoint_suffix = '/'.join(checkpoint_file.replace("s3://","").split('/')[1:])

    local_checkpoint_path = f"{working_dir}/checkpoint"
    local_weights_file = f"{local_checkpoint_path}/weights.pt"
    data_bucket_name = cfg.data_bucket_name
    data_bucket_key = cfg.data_bucket_key
    token_dictionary_filename = cfg.token_dictionary_filename

    remote_data_dir = f"s3://{data_bucket_name}/{data_bucket_key}"
    streaming_dataset_location = cfg.streaming_dataset_location
    remote_streaming_dataset_location = f"{remote_data_dir}/{streaming_dataset_location}"
    streaming_dataset_cache_location = f"{working_dir}/streaming/cache"

    mlm_probability = cfg.mlm_probability
    eval_batch_size = cfg.eval_batch_size

    # Download checkpoint from S3
    s3 = boto3.resource('s3')
    weight_content = s3.Bucket(data_bucket_name).Object(checkpoint_suffix).get()["Body"]
    os.makedirs(local_checkpoint_path,exist_ok=True)
    with open(local_weights_file, 'wb') as f:
        for chunk in iter(lambda: weight_content.read(4096), b''):
            f.write(chunk)

    # Read the token dictionary file
    token_dictionary = pickle.loads(s3.Bucket(data_bucket_name).Object(f"{data_bucket_key}/{token_dictionary_filename}").get()['Body'].read())
    tokenizer = GeneformerPreCollator(token_dictionary=token_dictionary)

    ### Load model
    model_config = build_model_config(cfg,token_dictionary)
    config = BertConfig(**model_config)
    model = BertForMaskedLM(config)
    # Convert to composer model
    composer_model = HuggingFaceModel(model,tokenizer)
    composer_model.eval()

    #Run inference
    print("Getting test data")
    streaming_dataset_eval = StreamingDataset(
    #    local=f"{streaming_dataset_cache_location}/test",
        remote=f"{remote_streaming_dataset_location}/test",
        download_retry=1,
        batch_size=eval_batch_size)

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

    pred_saver = PredictionSaver('f{working_dir}/predict_outputs')

    trainer = Trainer(
        model=composer_model,
        eval_dataloader=eval_dataloader,
        max_duration="10ep",
        save_overwrite=False,
        load_path=checkpoint_file,
        callbacks=pred_saver,
        device=cfg.get("device", "gpu"),
    )
    trainer.eval()
    trainer.predict(eval_dataloader)

if __name__ == '__main__':
    cfg = load_params(sys.argv[1], sys.argv[2:])
    main(cfg)
