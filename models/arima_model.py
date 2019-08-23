import tensorflow as tf
import numpy as np
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_squared_error
import os
import sys
import joblib
import pickle
sys.path.append(os.getcwd())

class ARIMAModel:
	def __init__(self, train_data, test_data):
		# get 'Close' feature vlue
		self.train_data = train_data[:, train_data.shape[1] - 3]
		self.test_data = test_data[:, test_data.shape[1] - 3]

		# create model
		self.model = auto_arima(self.train_data, start_p=1, start_q=1,max_p=1, max_q=1, m=1,start_P=0, 
			seasonal=True,d=1, D=1, trace=True,error_action='ignore',suppress_warnings=True)
	
	def save(self):
		# Serialize with Pickle
		with open('tmp/model.pkl', 'wb') as pkl:
			pickle.dump(self.model, pkl)

	def rmse_loss(self):
		preds, _ = self.model.predict(n_periods=self.test_data.shape[0], return_conf_int=True)
		rmse = np.sqrt(mean_squared_error(self.test_data, preds))
		return rmse