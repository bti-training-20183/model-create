import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from timeseries_models import LSTMModel, ARIMAModel


data = pd.read_csv("aame.us.txt").fillna(0)

data = data[['Open', 'High', 'Low', 'Close', 'Volume', 'OpenInt']].to_numpy()


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

print("Models evaluation:")
print(final_scores.values())

best_alg = min(final_scores.keys(), key=(lambda k: final_scores[k]))

model_lstm.save()
model_arima.save()


pred_df = pd.read_csv("aame.us.txt")
pred_data = pred_df.iloc[0:5][['Open', 'High', 'Low', 'Close', 'Volume', 'OpenInt']].to_numpy()
pred_data = np.expand_dims(pred_data, 0)

print("Input shape: ")
print(pred_data.shape)

lstm_pred = model_lstm.predict(pred_data)
arima_pred = model_arima.predict(pred_data.shape[0])

# print("lstm:")
# print(lstm_pred)
# print(lstm_pred.shape)

# print("arima:")
# print(arima_pred)
# print(arima_pred.shape)