echo ">>> Configuring aws"

cd /  
apt update
apt install unzip
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

mkdir ~/.aws
echo "[default]" >> ~/.aws/credentials
echo "AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID" >> ~/.aws/credentials
echo "AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY" >> ~/.aws/credentials

echo "[default]" >> ~/.aws/config
echo "region=us-west-2" >> ~/.aws/config
echo "output=json" >> ~/.aws/config

echo ">>> Installing Geneformer"
cd /composer_geneformer_pretrain
sh geneformer_prep.sh 

echo ">>> Installing dependencies"
pip install -q -r requirements.txt

mkdir /Geneformer/data -p 

echo ">>> Copying data from s3.. might take few mins"
#aws s3 cp s3://srijit-nair-sandbox-bucket/geneformer/data/token_dictionary.pkl /Geneformer/data/token_dictionary.pkl
aws s3 cp --quiet --recursive s3://srijit-nair-sandbox-bucket/geneformer/data /Geneformer/data
echo "done"
