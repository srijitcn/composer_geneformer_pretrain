#install git-lfs , pre-req for geneformer clone
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
apt-get install git-lfs
git lfs install

#install geneformer
cd /
git clone https://huggingface.co/ctheodoris/Geneformer
cd Geneformer
git checkout b07f4b1e8893a0923a8fde223fe3b5a60b976d99
pip install -q .

#Download training data and converting to streaming dataset
#commenting since we already have it in s3
#sh ./download_dataset.sh 
#python  create_mds.py
