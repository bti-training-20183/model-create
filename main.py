from utils.datastore_handler import DataStore_Handler
from utils.message_handler import MessageHandler
from models.timeseries_models import LSTMModel
import pandas
import os
import sys
import config
import time
import json
sys.path.append(os.getcwd())


def callback(channel, method, properties, body):
    print(f'[x] Received {body} from {properties}')

    '''
	LOAD DATA FROM MINIO --> CREATE - TRAIN - SAVE MODEL --> UPLOAD MODEL TO MINIO
	'''
    msg = json.loads(body)
    to_path = 'tmp/' + msg['name'] + '.csv'
    from_path = msg['file_uri']

    # download data from minio
    DataStore_Handler.download(from_path, to_path)

    # read data from downloaded file
    data = pandas.read_csv(to_path, header=None)

    # create model
    model = LSTMModel(data)
    model.compile()

    # train model
    model.train()

    # save model
    model.save()

    # upload model to minio
    fullname = msg['name']
    from_path = 'tmp/model.h5'
    filename, file_extension = os.path.splitext(fullname)
    to_path = filename + '/model/' + filename + '.h5'
    DataStore_Handler.upload(from_path, to_path)

	# TODO: Save logs to Mongo

    # send notification
    MessageHdlr.sendMessage(
        'from_creator', 'Model training done! Send to deployer')


class Creator:
    def __init__(self):
        pass

    def listen(self, queue, MessageHandler):
        MessageHandler.consumeMessage(queue, callback)


if __name__ == "__main__":
    if not os.path.exists('tmp'):
        os.makedirs('tmp')
    MessageHdlr = MessageHandler(config.RABBITMQ_CONNECTION)
    model_creator = Creator()
    model_creator.listen(config.QUEUE["from_preprocessor"], MessageHdlr)
