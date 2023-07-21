import boto3
s3client = boto3.client('s3')
# s3client.download_file(Bucket, Key, Filename)
bucket = "mlflow-consumer-complaints-case-study"
# key = "credentials.txt"
# filename = "doc/credentials.txt"
key = "dataset.dedup.resampled.csv"
filename = "data/dataset.dedup.resampled.csv"
s3client.download_file(bucket, key, filename)