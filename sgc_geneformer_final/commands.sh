#install git-lfs , pre-req for geneformer clone
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
apt-get install git-lfs
git lfs install

#install geneformer
cd /
git clone https://huggingface.co/ctheodoris/Geneformer
cd Geneformer
git checkout b07f4b1e8893a0923a8fde223fe3b5a60b976d99
pip install .

#Download training data and converting to streaming dataset
#commenting since we already have it in s3
#sh ./download_dataset.sh 
#python  create_mds.py


# echo ">>> Installing Geneformer"
# cd /composer_geneformer_pretrain
# sh geneformer_prep.sh 

# echo ">>> Installing dependencies"
# pip install -r requirements.txt

# # Create working directory
# mkdir -p /pretrain/temp

#sh download_dataset.sh
##################################################
##the following code is only needed to copy the dataset from s3 to locally create streaming dataset
#echo ">>> Configuring aws"
#cd /  
#apt update
#apt install unzip
#curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
#unzip awscliv2.zip
#sudo ./aws/install

#echo ">>> Copying data from s3.. might take few mins"
#cd /composer_geneformer_pretrain
#mkdir /Geneformer/data/dataset -p 
#aws s3 cp s3://srijit-nair-sandbox-bucket/geneformer/data/token_dictionary.pkl /Geneformer/data/token_dictionary.pkl
#aws s3 cp --recursive s3://srijit-nair-sandbox-bucket/geneformer/data/dataset /Geneformer/data/dataset
#mkdir /Geneformer/data -p 
#aws s3 cp --quiet --recursive s3://srijit-nair-sandbox-bucket/geneformer/data /Geneformer/data
#echo "done"
#python create_mds.py


