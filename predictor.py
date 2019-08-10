def lstm_predict(model, data, scaler):
	scaled_data = scaler.transform(data.reshape(data.shape[0]*data.shape[1], data.shape[2]))
	scaled_data = scaled_data.reshape(data.shape[0], data.shape[1], data.shape[2])

	preds = model.predict(scaled_data)
	
	broadcastable_preds = np.zeros(shape=(len(preds), data.shape[2]))
	broadcastable_preds[:,0] = preds[:,0]
	
	preds = self.scaler.inverse_transform(broadcastable_preds)[:,0]
	
	return preds

def arima_predict(model, n_periods):
	preds = model.predict(n_periods=n_periods)
	return preds