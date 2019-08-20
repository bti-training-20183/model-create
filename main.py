from utils.datastore_handler import DataStore_Handler
from utils.database_handler import Database_Handler
from utils.message_handler import Message_Handler
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
    # Clear file in tmp/ folder
    for f in os.listdir('/tmp'):
        os.remove('tmp/'+ f)
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
    train_data, test_data = train_test_split(data, test_size=0.2)


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
    model_file = 'model.h5'
    scaler_file = 'scaler.pkl'
    if best_alg == 'lstm':
        model_lstm.save()
    elif best_alg == 'arima':
        model_file = 'model.pkl'
        model_arima.save()  
    else:       # set lstm as the default model
        model_lstm.save()


    # upload model and necessary files to minio
    files = [model_file, scaler_file] # filelist for forwarding to edge-server
    filename = received_msg['name']
    file_extension = '.' + model_file.split('.')[-1]
    dest = filename + '/model/'
    for fname in files:
        if os.path.isfile('tmp'+fname):         # some models don't have scaler.pkl, etc.
            DataStore_Handler.upload('tmp/'+fname, dest + fname)
            os.remove('tmp/'+fname)

    # SAVE LOGS TO MONGO
    logs = {
        "name": filename,
        "type": file_extension,
        'date': time.strftime("%Y-%m-%d %H:%M:%S"),
        "file_uri": dest,
        'preprocessor_id': received_msg.get('preprocessor_id', '')
    }
    logged_info = Database_Handler.insert(config.MONGO_COLLECTION, logs)
    
    # send notification
    msg = {
        "name": filename,
        "type": file_extension,
        'date': time.strftime("%Y-%m-%d %H:%M:%S"),
        "file_uri": dest,
        'files': files,
        'creator_id': str(logged_info.inserted_id)
    }
    Message_Handler.sendMessage(
        'from_creator', json.dumps(msg))


class Creator:
    def __init__(self):
        pass

    def listen(self, queue, MessageHandler):
        MessageHandler.consumeMessage(queue, callback)


if __name__ == "__main__":
    if not os.path.exists('tmp'):
        os.makedirs('tmp')
    # MessageHdlr = MessageHandler(config.RABBITMQ_CONNECTION)
    model_creator = Creator()
    model_creator.listen(config.QUEUE["from_preprocessor"], Message_Handler)
