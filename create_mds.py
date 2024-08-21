from datasets import load_from_disk
from streaming import MDSWriter, StreamingDataset

datadir = "/Geneformer/data"
dataset_file = f"{datadir}/dataset/genecorpus_30M_2048.dataset"
streaming_dataset_location = f"{datadir}/streaming/genecorpus_30M_2048.dataset"

columns = {
    'input_ids': "ndarray",
    'lengths': 'int'
}

##3 Prepare dataset
dataset = load_from_disk(dataset_file)
print(dataset)


###*******************************************************************************************************************
# Split dataset into train and validation sets
train_test_split = dataset.train_test_split(test_size=0.1)
train_dataset = train_test_split["train"]
test_dataset = train_test_split["test"]

dataset_list = train_dataset.select(range(1000)).to_pandas().to_dict('records')

# Save the samples as shards using MDSWriter
with MDSWriter(out=streaming_dataset_location, columns=columns, compression='zstd') as out:
    for x in dataset_list:
        out.write({
            "input_ids" : x["input_ids"],
            "lengths" : x["lengths"]
        })

