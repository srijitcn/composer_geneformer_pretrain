from datasets import load_from_disk
from streaming import MDSWriter, StreamingDataset
from transformers.trainer_pt_utils import LengthGroupedSampler
import pickle
import torch
from torch.utils.data import DataLoader

datadir = "/Geneformer/data"
dataset_file = f"{datadir}/dataset/genecorpus_30M_2048.dataset"
streaming_dataset_location = f"{datadir}/streaming/genecorpus_30M_2048.dataset"
example_lengths_file = f"{datadir}/dataset/genecorpus_30M_2048_lengths.pkl"
geneformer_batch_size = 12

columns = {
    'input_ids': "ndarray",
    'length': 'int'
}

##3 Prepare dataset
dataset = load_from_disk(dataset_file)
print(dataset)


###*******************************************************************************************************************
# Split dataset into train and validation sets
train_test_split = dataset.train_test_split(test_size=0.1)
train_dataset = train_test_split["train"]
test_dataset = train_test_split["test"]

with open(example_lengths_file, "rb") as f:
    example_lengths = pickle.load(f)

#Get a sampler
generator = torch.Generator()
generator.manual_seed(
    int(torch.empty((), dtype=torch.int64).random_().item())
)

sampler = LengthGroupedSampler(
                dataset=train_dataset,
                batch_size=geneformer_batch_size,
                lengths=example_lengths,
                generator=generator,
            )

train_dataloader = DataLoader(train_dataset,
                        shuffle=False, 
                        drop_last=False, 
                        sampler=sampler)

#dataset_list = train_dataset.to_pandas().to_dict('records')

# Save the samples as shards using MDSWriter
#with MDSWriter(out=streaming_dataset_location, columns=columns, compression='zstd') as out:
##    for x in dataset_list:
#        out.write({
#            "input_ids" : x["input_ids"],
#            "length" : x["length"]
#        })

with MDSWriter(out=f"{streaming_dataset_location}_batched", columns=columns, compression='zstd') as out:
    batch_iter = iter(train_dataloader)
    for i in range(10): # int( len(train_dataset)/geneformer_batch_size ) + 1 ):
        batch = next(batch_iter)
        print(batch)
#        sample = next(sample_iter)
#        print(sample)
#        for x in sample:
#            out.write({
#                "input_ids" : x["input_ids"],
#                "length" : x["length"]
#            })


