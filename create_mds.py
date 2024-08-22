from datasets import load_from_disk
from streaming import MDSWriter

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

train_dataset_list = train_dataset.to_pandas().to_dict('records')
test_dataset_list = test_dataset.to_pandas().to_dict('records')

# Save the samples as shards using MDSWriter
with MDSWriter(out=f"{streaming_dataset_location}/train", columns=columns, compression='zstd') as out:
    for x in train_dataset_list:
        out.write({
            "input_ids" : x["input_ids"],
            "length" : x["length"]
        })

with MDSWriter(out=f"{streaming_dataset_location}/test", columns=columns, compression='zstd') as out:
    for x in test_dataset_list:
        out.write({
            "input_ids" : x["input_ids"],
            "length" : x["length"]
        })

