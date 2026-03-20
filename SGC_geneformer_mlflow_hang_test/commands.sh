echo ">>> Installing Geneformer"
cd /composer_geneformer_pretrain
sh geneformer_prep.sh 

# echo ">>> Installing dependencies"
# pip install -r requirements.txt

# Uninstall existing torch (e.g., 2.7.0) and install torch 2.8
pip uninstall -y torch torchvision torchaudio
pip install --no-cache-dir torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu126

# Verify versions
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import mlflow; print(f'MLflow: {mlflow.__version__}')"

# Create working directory
mkdir -p /pretrain/temp

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


