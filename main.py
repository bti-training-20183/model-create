from utils.datastore_handler import DataStore_Handler
from utils.database_handler import Database_Handler
from utils.message_handler import MessageHandler
from models.timeseries_models import LSTMModel, ARIMAModel
from sklearn.model_selection import train_test_split
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
    received_msg = json.loads(body)
    to_path = 'tmp/' + received_msg['name'] + received_msg['type']
    from_path = received_msg['file_uri']

    # download data from minio
    DataStore_Handler.download(from_path, to_path)

    
    # read data from downloaded file
    data = pandas.read_csv(to_path, header=None)
    data = data.to_numpy()

    # split data to train set and test set
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=3, shuffle=True)


    '''
    train models
    '''
    # 1. lstm
    model_lstm = LSTMModel(train_data, test_data)
    model_lstm.compile()
    model_lstm.train()

    # 2. arima
    model_arima = ARIMAModel(train_data, test_data)


    '''
    find the best model
    '''
    final_scores = dict()
    rmse_loss_lstm = model_lstm.rmse_loss()
    final_scores['lstm'] = rmse_loss_lstm

    rmse_loss_arima = model_arima.rmse_loss()
    final_scores['arima'] = rmse_loss_arima

    best_alg = min(final_scores.keys(), key=(lambda k: final_scores[k]))

    
    '''
    save the best model
    '''
    if best_alg == 'lstm':
        model_lstm.save()
    elif best_alg == 'arima':
        model_arima.save()  
    else:       # set lstm as the default model
        model_lstm.save()


    # upload model to minio
    filename = received_msg['name']
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
        'preprocessor_id': received_msg.get('preprocessor_id', '')
    }
    logged_info = Database_Handler.insert(logs)
    # send notification
    msg = {
        "name": filename,
        "type": file_extension,
        'date': time.strftime("%Y-%m-%d %H:%M:%S"),
        "file_uri": to_path,
        'creator_id': str(logged_info.inserted_id)
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
