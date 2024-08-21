curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
apt-get install git-lfs
git lfs install


cd ..
git clone https://huggingface.co/ctheodoris/Geneformer
cd Geneformer
git checkout b07f4b1e8893a0923a8fde223fe3b5a60b976d99
pip install .

cd composer_geneformer_pretrain
mkdir output
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

ls -al genecorpus_30M_2048.dataset/

cd ../..
pip install -r requirements.txt