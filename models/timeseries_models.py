import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.layers import LSTM, Dense, Dropout, Flatten
from keras.models import Sequential
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_squared_error
import os
import sys
import joblib
import pickle
sys.path.append(os.getcwd())


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

		self.x_train, self.y_train = self.create_timeseries_train(scaled_train_data)
		self.x_test, self.y_test = self.create_timeseries_test(test_data, scaled_test_data)

	def create_timeseries_train(self, data):		# input are all 6 feature values, output is just 'Close' value;
		dim_0 = data.shape[0] - self.TIMESTEPS
		dim_1 = data.shape[1]
		x = np.zeros((dim_0, self.TIMESTEPS, dim_1))
		y = np.zeros((dim_0))
		y_test = np.zeros((dim_0))

		for i in range(dim_0):
			x[i] = data[i:self.TIMESTEPS+i]
			y[i] = data[self.TIMESTEPS+1, dim_1 - 3]

		return x, y

	def create_timeseries_test(self, data_origin, data_scaled):		# input are all 6 feature values, output is just 'Close' value
		dim_0 = data_origin.shape[0] - self.TIMESTEPS
		dim_1 = data_origin.shape[1]
		x = np.zeros((dim_0, self.TIMESTEPS, dim_1))
		y = np.zeros((dim_0))

		for i in range(dim_0):
			x[i] = data_scaled[i:self.TIMESTEPS+i]
			y[i] = data_origin[self.TIMESTEPS+1, dim_1 - 3]

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
		with open('tmp/scaler.pkl', 'wb') as pkl:
			pickle.dump(self.scaler, pkl)

	def rmse_loss(self):
		preds = self.model.predict(self.x_test)

		transformable_preds = np.zeros(shape=(len(preds), self.x_test.shape[2]))
		transformable_preds[:,0] = preds[:, 0]

		preds = self.scaler.inverse_transform(transformable_preds)[:,0]	

		rmse = np.sqrt(mean_squared_error(self.y_test, preds))
		return rmse


class ARIMAModel:
	def __init__(self, train_data, test_data):
		# get 'Close' feature vlue
		self.train_data = train_data[:, train_data.shape[1] - 3]
		self.test_data = test_data[:, test_data.shape[1] - 3]

		# create model
		self.model = auto_arima(self.train_data, start_p=1, start_q=1,max_p=2, max_q=2, m=5,start_P=0, 
			seasonal=True,d=1, D=1, trace=True,error_action='ignore',suppress_warnings=True)
	
	def save(self):
		# Serialize with Pickle
		with open('tmp/model.pkl', 'wb') as pkl:
			pickle.dump(self.model, pkl)

	def rmse_loss(self):
		preds, _ = self.model.predict(n_periods=self.test_data.shape[0], return_conf_int=True)
		rmse = np.sqrt(mean_squared_error(self.test_data, preds))
		return rmse