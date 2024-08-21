

##1 Set parameters

import datetime

# imports
import os

import pickle
import random
import subprocess

import numpy as np
import pytz

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR

from datasets import load_from_disk

from transformers import BertConfig, BertForMaskedLM, DataCollatorForLanguageModeling

import geneformer
from geneformer.pretrainer import GeneformerPreCollator

from composer.models.huggingface import HuggingFaceModel
from composer.utils import reproducibility
from composer import Trainer


from streaming import MDSWriter, StreamingDataset


#### Env variables
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["OMPI_MCA_opal_cuda_support"] = "true"
os.environ["CONDA_OVERRIDE_GLIBC"] = "2.56"

seed_num = 0
random.seed(seed_num)
np.random.seed(seed_num)
seed_val = 42

# set local time/directories
timezone = pytz.timezone("US/Eastern")
rootdir = "/composer_output/output"
datadir = "/Geneformer/data"

# set model parameters
# model type
model_type = "bert"
# max input size
max_input_size = 2**11  # 2048
# number of layers
num_layers = 6
# number of attention heads
num_attn_heads = 4
# number of embedding dimensions
num_embed_dim = 256
# intermediate size
intermed_size = num_embed_dim * 2
# activation function
activ_fn = "relu"
# initializer range, layer norm, dropout
initializer_range = 0.02
layer_norm_eps = 1e-12
attention_probs_dropout_prob = 0.02
hidden_dropout_prob = 0.02

# set training parameters
# total number of examples in Genecorpus-30M after QC filtering:
num_examples = 27_406_208
# number gpus
num_gpus = 12
# batch size for training and eval
geneformer_batch_size = 12
# max learning rate
max_lr = 1e-3
# learning schedule
lr_schedule_fn = "linear"
# warmup steps
warmup_steps = 10_000
# number of epochs
epochs = 3
# optimizer
optimizer = "adamw"
# weight_decay
weight_decay = 0.001

mlm_probability = 0.15

dataset_file = f"{datadir}/dataset/genecorpus_30M_2048.dataset"
streaming_dataset_location = f"{datadir}/streaming/genecorpus_30M_2048.dataset"
streaming_dataset_cache_location = f"{datadir}/streaming/cache"
example_lengths_file = f"{datadir}/dataset/genecorpus_30M_2048_lengths.pkl"

# output directories
run_name = f"geneformer_30M_L{num_layers}_emb{num_embed_dim}_SL{max_input_size}_E{epochs}_B{geneformer_batch_size}"
training_output_dir = f"{rootdir}/models/{run_name}/"
logging_dir = f"{rootdir}/runs/{run_name}/"
model_output_dir = os.path.join(training_output_dir, "models/")

# ensure not overwriting previously saved model
model_output_file = os.path.join(model_output_dir, "pytorch_model.bin")
if os.path.isfile(model_output_file) is True:
    raise Exception("Model already saved to this directory.")

# make training and model output directories
subprocess.call(f"mkdir -p {training_output_dir}", shell=True)
subprocess.call(f"mkdir -p {model_output_dir}", shell=True)

with open(f"{datadir}/token_dictionary.pkl", "rb") as fp:
    token_dictionary = pickle.load(fp)

##2  Get model and tokenizer
config = {
    "hidden_size": num_embed_dim,
    "num_hidden_layers": num_layers,
    "initializer_range": initializer_range,
    "layer_norm_eps": layer_norm_eps,
    "attention_probs_dropout_prob": attention_probs_dropout_prob,
    "hidden_dropout_prob": hidden_dropout_prob,
    "intermediate_size": intermed_size,
    "hidden_act": activ_fn,
    "max_position_embeddings": max_input_size,
    "model_type": model_type,
    "num_attention_heads": num_attn_heads,
    "pad_token_id": token_dictionary.get("<pad>"),
    "vocab_size": len(token_dictionary),  # genes+2 for <mask> and <pad> tokens
}

### Load model

reproducibility.configure_deterministic_mode()
reproducibility.seed_all(seed_val)


config = BertConfig(**config)
model = BertForMaskedLM(config)
tokenizer = GeneformerPreCollator(token_dictionary=token_dictionary)
model.train()
print(model)


#Create streaming dataset
streaming_dataset = StreamingDataset(local=streaming_dataset_location,batch_size=geneformer_batch_size)

#eval_dataloader = DataLoader(train_test_split["test"],batch_size=geneformer_batch_size, shuffle=False, drop_last=False, collate_fn=data_collator)

#Prepare composer model
composer_model = HuggingFaceModel(model)

optimizer = AdamW(
    params=composer_model.parameters(),
    lr=3e-5, betas=(0.9, 0.98),
    eps=1e-6, weight_decay=3e-6
)
linear_lr_decay = LinearLR(
    optimizer, start_factor=1.0,
    end_factor=0, total_iters=150
)

#data collator
data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=True, 
        mlm_probability=mlm_probability
    )

train_dataloader = DataLoader(streaming_dataset,
                        shuffle=False, 
                        drop_last=False, 
                        #sampler=sampler,
                        collate_fn=data_collator)

# Create Trainer Object
trainer = Trainer(
    model=composer_model, # This is the model from the HuggingFaceModel wrapper class.
    train_dataloader=train_dataloader,
    eval_dataloader=None,
    max_duration="2ep",
    optimizers=optimizer,
    schedulers=[linear_lr_decay],
    device="gpu" ,
    train_subset_num_batches=150,
    save_folder=model_output_dir,
    save_interval="1ep",
    save_overwrite=True,
    run_name=run_name,
    seed=seed_val,
    deepspeed_config={
        "train_batch_size": 8,
        "fp16": {"enabled": True},
    }
)
# Start training
trainer.fit()

print(trainer.state.train_metrics)

print(f"Trained model available at : {model_output_dir}")

print("*************Done")