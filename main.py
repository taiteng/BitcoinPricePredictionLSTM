from math import sqrt

from flask import Flask, jsonify
from flask_cors import CORS
import datetime as dt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import pandas as pd
import requests


app = Flask(__name__)


CORS(app)


def get_data():
    url = 'https://min-api.cryptocompare.com/data/histoday'
    params = {'fsym': 'BTC', 'tsym': 'USD', 'limit': 2000}

    response = requests.get(url, params=params)
    data = response.json()['Data']
    df = pd.DataFrame(data)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)

    start = dt.datetime(2011, 1, 1)
    end = dt.datetime.now()
    delta = end - start

    while delta.days > 0:
        if delta.days >= 2000:
            date = end - dt.timedelta(days=2000)
            params = {'fsym': 'BTC', 'tsym': 'USD', 'limit': 2000,
                      'toTs': date.timestamp()}
        else:
            params = {'fsym': 'BTC', 'tsym': 'USD', 'limit': delta.days,
                      'toTs': df.index[0].timestamp()}
        response = requests.get(url, params=params)
        data = response.json()['Data']
        df1 = pd.DataFrame(data)
        df1['time'] = pd.to_datetime(df1['time'], unit='s')
        df1.set_index('time', inplace=True)
        delta -= dt.timedelta(days=2000)
        df = pd.concat([df1, df], axis=0)
    btcData = pd.DataFrame(df)
    btcData = btcData.dropna()
    cols = ["conversionType", "conversionSymbol", "volumefrom"]
    btcData = btcData.drop(columns=cols, axis=1)

    return btcData


def build_lstm_model(btcData):
    scaler = MinMaxScaler()
    data_sc = scaler.fit_transform(btcData['close'].values.reshape(-1, 1))

    prediction_days = 60
    future_days = 30
    x_train, y_train = [], []

    for x in range(prediction_days, len(data_sc) - future_days):
        x_train.append(data_sc[x - prediction_days:x, 0])
        y_train.append(data_sc[x + future_days, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    model = Sequential()

    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=30))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=25, batch_size=32)

    return model


def test_predict(model, btcData):
    scaler = MinMaxScaler()
    prediction_days = 60
    test_start = '2020-01-01'
    test_data = btcData.iloc[btcData.index >= test_start].copy()

    total_data = pd.concat([btcData['close'], test_data['close']])

    model_inputs = total_data[len(total_data) - len(test_data) - prediction_days:].values
    model_inputs = model_inputs.reshape(-1, 1)
    model_inputs = scaler.fit_transform(model_inputs)

    x_test = []

    for x in range(prediction_days, len(model_inputs)):
        x_test.append(model_inputs[x - prediction_days:x, 0])
    x_test = np.array(x_test, dtype=np.float32)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    prediction_price = model.predict(x_test)
    prediction_price = scaler.inverse_transform(prediction_price)

    return prediction_price


def calculate_accuracy(btcData, test_predictions):
    test_start = '2020-01-01'
    test_data = btcData.iloc[btcData.index >= test_start].copy()
    actual_price = test_data['close'].values

    if len(actual_price) != len(test_predictions):
        raise ValueError("Lengths of actual and predicted values must be the same.")

    absolute_errors = [abs(actual - predicted) for actual, predicted in zip(actual_price, test_predictions)]
    mean_absolute_error = np.mean(absolute_errors)
    mean_actual_value_MAE = np.mean(actual_price)
    mae_accuracy = (1 - (mean_absolute_error / mean_actual_value_MAE)) * 100
    print('MAE: ', mean_absolute_error)

    root_mean_square_error = np.sqrt(np.mean((actual_price - test_predictions.mean(axis=1)) ** 2))
    mean_actual_value_RMSE = np.mean(actual_price)
    rmse_accuracy = (1 - root_mean_square_error / mean_actual_value_RMSE) * 100
    print('RMSE: ', root_mean_square_error)

    mean_accuracy = (mae_accuracy + rmse_accuracy) / 2

    return mean_accuracy


def make_predictions(model, btcData):
    scaler = MinMaxScaler()
    prediction_days = 60
    future_days = 30

    test_start = '2020-01-01'
    test_data = btcData.iloc[btcData.index >= test_start].copy()

    total_data = pd.concat([btcData['close'], test_data['close']])

    model_inputs = total_data[len(total_data) - len(test_data) - prediction_days:].values
    model_inputs = model_inputs.reshape(-1, 1)
    model_inputs = scaler.fit_transform(model_inputs)

    real_data = [model_inputs[len(model_inputs) + 1 - prediction_days:len(model_inputs) + 1, 0]]
    real_data = np.array(real_data)
    real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

    prediction = model.predict(real_data)
    prediction = scaler.inverse_transform(prediction)

    prediction = np.reshape(prediction, future_days)

    return prediction


@app.route('/')
def predict():
    try:
        btcData = get_data()

        model = build_lstm_model(btcData)

        test_predictions = test_predict(model, btcData)

        accuracy_percentage = calculate_accuracy(btcData, test_predictions)

        predictions = make_predictions(model, btcData)

        prediction = predictions.tolist()

        return jsonify({'Prediction': str(prediction), 'Accuracy': str(f"{accuracy_percentage:.2f}%")})
    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
