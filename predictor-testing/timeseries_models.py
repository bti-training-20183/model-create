import tensorflow as tf
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
from keras.layers import LSTM, Dense, Dropout, Flatten
from keras.models import Sequential
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_squared_error
# from fbprophet import Prophet
import math

class LSTMModel:
	def __init__(self, train_data, test_data):
		self.TIMESTEPS = 5
		self.BATCH_SIZE = 32
		# self.EPOCHS = 50
		self.EPOCHS = 1 # For testing only

		# normalize data
		self.scaler = MinMaxScaler(feature_range=(0,1))
		scaled_train_data = self.scaler.fit_transform(train_data)
		scaled_test_data = self.scaler.transform(test_data)

		self.x_train, self.y_train = self.create_timeseries(scaled_train_data)
		self.x_test, self.y_test = self.create_timeseries(scaled_test_data)

	def create_timeseries(self, data):
		dim_0 = data.shape[0] - self.TIMESTEPS
		dim_1 = data.shape[1]
		x = np.zeros((dim_0, self.TIMESTEPS, dim_1))
		y = np.zeros((dim_0))

		for i in range(dim_0):
			x[i] = data[i:self.TIMESTEPS+i]
			y[i] = data[self.TIMESTEPS+1, dim_1 - 3]
		
		return x, y

	def compile(self):
		self.model = Sequential()
		self.model.add(LSTM(units=50, return_sequences=True, input_shape=(self.TIMESTEPS, self.x_train.shape[2])))
		self.model.add(Dropout(0.5))
		self.model.add(Dense(20,activation='relu'))
		self.model.add(Flatten())
		self.model.add(Dense(1,activation='sigmoid'))
		self.model.compile(loss='mean_squared_error', optimizer='adam')

	def train(self):
		self.model.fit(self.x_train, self.y_train, epochs=self.EPOCHS, batch_size=self.BATCH_SIZE, verbose=1)

	def save(self):
		self.model.save("tmp/model.h5")

	def rmse_loss(self):
		preds = self.model.predict(self.x_test)

		transformable_preds = np.zeros(shape=(len(preds), self.x_test.shape[2]))
		transformable_preds[:,0] = preds[:, 0]

		preds = self.scaler.inverse_transform(transformable_preds)[:,0]	

		rmse = np.sqrt(mean_squared_error(self.y_test, preds))
		return rmse

	def predict(self, data):
		scaled_data = self.scaler.transform(data.reshape(data.shape[0]*data.shape[1], data.shape[2]))
		scaled_data = scaled_data.reshape(data.shape[0], data.shape[1], data.shape[2])
		preds = self.model.predict(scaled_data)
		broadcastable_preds = np.zeros(shape=(len(preds), data.shape[2]))
		broadcastable_preds[:,0] = preds[:,0]
		preds = self.scaler.inverse_transform(broadcastable_preds)[:,0]
		return preds


class ARIMAModel:
	def __init__(self, train_data, test_data):
		# get 'Close' feature
		self.train_data = train_data[:, train_data.shape[1] - 3]
		self.test_data = test_data[:, test_data.shape[1] - 3]

		self.model_for_evaluating = auto_arima(self.train_data, start_p=1, start_q=1,max_p=1, max_q=1, m=1,start_P=0, 
			seasonal=True,d=1, D=1, trace=True,error_action='ignore',suppress_warnings=True)

	def save(self):
		data = np.concatenate((self.train_data, self.test_data), axis=0)
		self.model_for_predicting = auto_arima(data, start_p=1, start_q=1,max_p=1, max_q=1, m=1,start_P=0, 
			seasonal=True,d=1, D=1, trace=True,error_action='ignore',suppress_warnings=True)
	
		with open("tmp/model.pkl", "wb") as pkl:
			pickle.dump(self.model_for_predicting, pkl)

	def rmse_loss(self):
		preds = self.model_for_evaluating.predict(n_periods=self.test_data.shape[0], return_conf_int=False)
		rmse = np.sqrt(mean_squared_error(self.test_data, preds))
		return rmse

	def predict(self, n_periods):
		preds = self.model_for_predicting.predict(n_periods=n_periods)
		return preds

# class PROPHETModel:
#     def __init__(self, data_frame):
#         self.data_frame = data_frame
#         self.data = self.data_frame[['Date', 'Close']]
#         self.data.colums['ds', 'y']
#         self.train_size = int(len(self.data)*0.8)
#         self.val_size = len(self.data) - self.train_size
#         self.train_data = self.data[0:self.train_size]
#         self.val_data= self.data[self.train_size:len(self.data)]
#         self.prophet_model = Prophet()

#     def rmse_loss(self):
#         y_true = list(self.val_data['y'])
#         future_date = self.prophet_model.make_future_dataframe(self.val_size, freq='d')
#         preds_future_date = self.prophet_model.predict(future_date)
#         y_hat = preds_future_date['yhat'][len(preds_future_date)-self.val_size:len(preds_future_date)]
#         y_hat = list(y_hat)
#         rmse = np.sqrt(np.mean((np.array(y_true)-np.array(y_hat))**2))
#         return rmse
    
#     def save(self):
#         self.prophet_model.fit(self.train_data)
#         with open("tmp/prophet_model.pkl", "wb") as pkl:
#             pickle.dump(self.prophet_model, pkl)

#     def predict(self, n_periods):
#         future_date_frame = self.prophet_model.make_future_dataframe(self.val_size + self.n_preiods, freq='d')
#         preds = self.prophet_model.predict(future_date_frame)
#         return preds
