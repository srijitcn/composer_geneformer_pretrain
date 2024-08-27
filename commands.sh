echo ">>> Configuring aws"

echo ">>> Installing Geneformer"
cd /composer_geneformer_pretrain
sh geneformer_prep.sh 

echo ">>> Installing dependencies"
pip install -q -r requirements.txt

#echo ">>> Copying data from s3.. might take few mins"
mkdir /Geneformer/data/dataset -p 
aws s3 cp s3://srijit-nair-sandbox-bucket/geneformer/data/token_dictionary.pkl /Geneformer/data/token_dictionary.pkl
aws s3 cp --recursive s3://srijit-nair-sandbox-bucket/geneformer/data/dataset /Geneformer/data/dataset
#mkdir /Geneformer/data -p 
#aws s3 cp --quiet --recursive s3://srijit-nair-sandbox-bucket/geneformer/data /Geneformer/data
#echo "done"

python create_mds.py

#create working dirrectory
mkdir -p /pretrain/temp

