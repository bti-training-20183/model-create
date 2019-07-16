import io
import os
import sys
import config
from minio import Minio
from minio.error import (ResponseError, BucketAlreadyOwnedByYou,
                         BucketAlreadyExists)
sys.path.append(os.getcwd())


class DataStoreHandler:
    def __init__(self, endpoint, access_key, secret_key, bucket_name):
        self.bucket_name = bucket_name
        self.minioClient = Minio(
            endpoint=endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=False)
        print('Connected to Minio')
        try:
            self.minioClient.make_bucket(bucket_name)
        except BucketAlreadyOwnedByYou as err:
            print('BucketAlreadyOwnedByYou')
            pass
        except BucketAlreadyExists as err:
            print('BucketAlreadyExists')
            pass
        except ResponseError as err:
            print('ResponseError')
            pass

    def upload(self, from_path, to_path):
        try:
            self.minioClient.fput_object(self.bucket_name, to_path, from_path)
        except ResponseError as err:
            return err

    def download(self, from_path, to_path):
        try:
            f = self.minioClient.fget_object(
                self.bucket_name, from_path, to_path)
            return f
        except ResponseError as err:
            print(err)


DataStore_Handler = DataStoreHandler(
    config.MINIO_URL, config.MINIO_ACCESS_KEY, config.MINIO_SECRET_KEY, config.MINIO_BUCKET)
