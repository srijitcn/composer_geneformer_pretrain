import boto3
import pickle 

s3 = boto3.resource('s3')
token_dictionary = pickle.loads(s3.Bucket("srijit-nair-sandbox-bucket").Object("geneformer/data/token_dictionary.pkl").get()['Body'].read())
print(token_dictionary)