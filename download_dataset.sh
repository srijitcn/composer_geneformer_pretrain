mkdir /Geneformer -p
cd /Geneformer

mkdir data
cd data

curl https://huggingface.co/datasets/ctheodoris/Genecorpus-30M/resolve/main/token_dictionary.pkl\?download\=true -o token_dictionary.pkl  

mkdir dataset
cd dataset
mkdir genecorpus_30M_2048.dataset

curl https://huggingface.co/datasets/ctheodoris/Genecorpus-30M/resolve/main/genecorpus_30M_2048.dataset/dataset.arrow -L -o genecorpus_30M_2048.dataset/dataset.arrow  
curl https://huggingface.co/datasets/ctheodoris/Genecorpus-30M/resolve/main/genecorpus_30M_2048.dataset/dataset_info.json -L -o genecorpus_30M_2048.dataset/dataset_info.json
curl https://huggingface.co/datasets/ctheodoris/Genecorpus-30M/resolve/main/genecorpus_30M_2048.dataset/state.json -L -o genecorpus_30M_2048.dataset/state.json
curl https://huggingface.co/datasets/ctheodoris/Genecorpus-30M/resolve/main/genecorpus_30M_2048_lengths.pkl -L -o genecorpus_30M_2048_lengths.pkl

cd /
