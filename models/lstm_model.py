import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.layers import LSTM, Dense, Dropout, Flatten
from keras.models import Sequential, load_model
from sklearn.metrics import mean_squared_error
from keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import sys
import joblib
import pickle
sys.path.append(os.getcwd())

class LSTMModel:
	def __init__(self, train_data, test_data):
		self.INPUT_DAYS = 90			# get input of 3 months to train
		self.OUTPUT_DAYS = 90			# predict 3 months
		self.BATCH_SIZE = 32
		self.EPOCHS = 100

		# normalize data
		self.scaler = MinMaxScaler(feature_range=(0,1))
		scaled_train_data = self.scaler.fit_transform(train_data)
		scaled_test_data = self.scaler.transform(test_data)

		self.x_train, self.y_train = self.create_timeseries_train(scaled_train_data)
		self.x_test, self.y_test = self.create_timeseries_test(test_data, scaled_test_data)

	def create_timeseries_train(self, data):		# input are all 6 feature values, output is just 'Close' value;
		X, y = [], []
		for i in range(len(data) - self.INPUT_DAYS - self.OUTPUT_DAYS):
			X.append(data[i:i+self.INPUT_DAYS])
			y.append(data[i+self.INPUT_DAYS : i+self.INPUT_DAYS+self.OUTPUT_DAYS, 3])  # 3 is the index of 'Close' colum in data
		return np.array(X), np.array(y)

	def create_timeseries_test(self, data, data_scaled):		# input are all 6 feature values, output is just 'Close' value;
		X, y = [], []
		for i in range(len(data) - self.INPUT_DAYS - self.OUTPUT_DAYS):
			X.append(data_scaled[i:i+self.INPUT_DAYS])
			y.append(data[i+self.INPUT_DAYS:i+self.INPUT_DAYS+self.OUTPUT_DAYS, 3])  # 3 is the index of 'Close' colum in data
		return np.array(X), np.array(y)

	def compile(self):
		self.model = Sequential()
		self.model.add(LSTM(units=50, return_sequences=True, input_shape=(self.x_train.shape[1], self.x_train.shape[2])))
		self.model.add(Dropout(0.2))
		self.model.add(LSTM(units = 40, return_sequences = True))
		self.model.add(Dropout(0.2))
		self.model.add(LSTM(units = 50, return_sequences = True))
		self.model.add(Dropout(0.2))
		self.model.add(LSTM(units = 40, return_sequences=True))
		self.model.add(Dropout(0.2))
		self.model.add(LSTM(units = 30))
		self.model.add(Dropout(0.2))
		self.model.add(Dense(self.OUTPUT_DAYS))
		self.model.compile(loss='mean_squared_error', optimizer='adam')

	def train(self):
		es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
		mc = ModelCheckpoint('final_model.h5', monitor='val_loss', mode='min', save_best_only=True, verbose=1)
		
		self.model.fit(self.x_train, self.y_train, epochs=self.EPOCHS, batch_size=self.BATCH_SIZE, verbose=1, validation_split=0.15, callbacks=[es, mc])

	def save(self):
		model = load_model("final_model.h5")
		model.save("tmp/model.h5")
		with open('tmp/scaler.pkl', 'wb') as pkl:
			pickle.dump(self.scaler, pkl)

	def rmse_loss(self):
		preds = self.model.predict(self.x_test)

		tmp = np.zeros(shape=(preds.shape[0]*preds.shape[1], self.x_test.shape[2]))
		tmp[:,3] = preds.reshape(preds.shape[0]*preds.shape[1])

		preds = self.scaler.inverse_transform(tmp)[:,3]	

		rmse = np.sqrt(mean_squared_error(self.y_test, preds))
		return rmse
