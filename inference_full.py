from omegaconf import DictConfig

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

from streaming import MDSWriter, StreamingDataset

import mlflow
from mlflow.models.signature import infer_signature

class GeneformerTorch(mlflow.pyfunc.PythonModel):
    def __init__(self, config, model, tokenizer):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
    def predict(self, test_data):
        return self.model(test_data["input_ids"])

def main(cfg: DictConfig):
    #os.environ["MASTER_ADDR"] = "127.0.0.1"

    #load the model back and run some test
    working_dir = cfg.working_dir
    checkpoint_file = "s3://srijit-nair-sandbox-bucket/geneformer/pretrain/checkpoints_full/ep10-ba10000-rank0.pt" #f"{cfg.save_folder}/ep10-ba10000-rank0.pt"
    checkpoint_prefix = '/'.join(checkpoint_file.replace("s3://","").split('/')[1:])

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
    # Read the token dictionary file
    s3 = boto3.resource('s3')
    token_dictionary = pickle.loads(s3.Bucket(data_bucket_name).Object(f"{data_bucket_key}/{token_dictionary_filename}").get()['Body'].read())

    ##load model weights
    print("Loading weights")
    #copy weight to local folder
    os.makedirs(local_checkpoint_path,exist_ok=True)

    weight_content = s3.Bucket(data_bucket_name).Object(checkpoint_prefix).get()["Body"]
    with open(local_weights_file, 'wb') as f:
        for chunk in iter(lambda: weight_content.read(4096), b''):
            f.write(chunk)

    model_state_dict = torch.load(local_weights_file)["state"]["model"]

    #removing prefix from huggingface model
    consume_prefix_in_state_dict_if_present(model_state_dict, prefix="model.")

    ### Load model
    print("Loading model")
    model_config = build_model_config(cfg,token_dictionary)
    config = BertConfig(**model_config)
    model = BertForMaskedLM(config)
    model.load_state_dict(model_state_dict)
    tokenizer = GeneformerPreCollator(token_dictionary=token_dictionary)
    pyfunc_model = GeneformerTorch(config, model, tokenizer)

    #Run inference
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

    print("Perform inference")
    result = model(test_data["input_ids"])

    print("Result")
    print(result)

    signature = infer_signature(test_data["input_ids"], result)

    print("Log the model to mlflow")
    mlflow.set_tracking_uri("databricks")
    experiment_base_path = f"/Users/yen.low@databricks.com/mlflow_experiments/geneformer_pretraining"
    experiment = mlflow.set_experiment(experiment_base_path)

    with mlflow.start_run(experiment_id=experiment.experiment_id) as mlflow_run:
        mlflow.pyfunc.log_model(
            python_model=pyfunc_model,
            artifact_path="model",
            signature=signature,
            registered_model_name='geneformer_10ep'
        )
    run_id = mlflow_run.info.run_id
    loaded_model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
    results_mlflow = loaded_model.predict(test_data)
    print(results_mlflow)

if __name__ == '__main__':
    yaml_path, args_list = sys.argv[1], sys.argv[2:]
    with open(yaml_path) as f:
        yaml_cfg = om.load(f)
    cli_cfg = om.from_cli(args_list)
    cfg = om.merge(yaml_cfg, cli_cfg)
    cfg = cast(DictConfig, cfg)  # for type checking
    main(cfg)