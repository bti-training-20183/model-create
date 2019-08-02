from utils.datastore_handler import DataStore_Handler
from utils.database_handler import Database_Handler
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
    to_path = 'tmp/' + msg['name'] + msg['type']
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
    filename = msg['name']
    from_path = 'tmp/model.h5'
    file_extension = '.' + from_path.split('.')[-1]
    to_path = filename + '/model/' + filename + file_extension
    DataStore_Handler.upload(from_path, to_path)
    os.remove(from_path)
    # SAVE LOGS TO MONGO
    logs = {
        "name": filename,
        "type": file_extension,
        'date': time.strftime("%Y-%m-%d %H:%M:%S"),
        "file_uri": to_path,
        # TODO add algorithm
        'algorithm': '',
        'preprocessor_id': received_msg.get('preprocessor_id', '')
    }
    logged_info = Database_Handler.insert(logs)
    # send notification
    msg = {
        "name": filename,
        "type": file_extension,
        'date': time.strftime("%Y-%m-%d %H:%M:%S"),
        "file_uri": to_path,
        # TODO add algorithm
        'algorithm': '',
        'creator_id': logged_info.inserted_id
    }
    MessageHdlr.sendMessage(
        'from_creator', json.dumps(msg))


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
