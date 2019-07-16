from sklearn.preprocessing import MinMaxScaler
from keras.layers import LSTM, Dense, Dropout, Flatten
from keras.models import Sequential
import numpy as np

class LSTMModel:
	'''LSTM model for data. Input must be a numpy'''
	def __init__(self, data):
		self.TIMESTEPS = 5
		self.BATCH_SIZE = 32
		self.EPOCHS = 50

		data = numpy.array(data)

		# normalize data
		scaler = MinMaxScaler(feature_range=(0,1))
		scaled_data = scaler.fit_transform(data)

		# create timeseries
		dim_0 = scaled_data.shape[0] - self.TIMESTEPS
		dim_1 = scaled_data.shape[1]
		x = np.zeros((dim_0, self.TIMESTEPS, dim_1))
		y = np.zeros((dim_0))

		for i in range(dim_0):
			x[i] = scaled_data[i:self.TIMESTEPS+i]
			y[i] = scaled_data[self.TIMESTEPS+1, dim_1 - 2]
		
		self.x_train = x
		self.y_train = y

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