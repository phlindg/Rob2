import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def normalize_window(window_data):
    normalized_data = []
    for window in window_data:
        normalized_window = [((float(p) / float(window[0]))-1) for p in window]
        normalized_data.append(normalized_window)
    return normalized_data

def preprocess_ts(data, seq_len, test_size=0.9,normalize=True):
    """
    data : np.array av data
    seq_len : hur lång varje sekvens är. alltså varje rad i matrisen
    test_size : hur stor del av data är train o test
    normalize : om den ska normaliseras eller ej
    """
    sequence_len = seq_len+1
    result = []
    for index in range(len(data) - sequence_len):
        res = data[index: index+sequence_len]
        result.append(res)
        
    if normalize:
        result = normalize_window(result)
    result = np.array(result)
    
    row = round(test_size*result.shape[0])
    train = result[:int(row), :]
    np.random.shuffle(train)
    x_train = train[:, :-1]
    y_train = train[:, -1]
    x_test = result[int(row):, :-1]
    y_test = result[int(row):, -1]
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    y_train = np.reshape(y_train, (len(y_train), 1, 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))  

    return x_train, y_train, x_test, y_test

def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data', alpha=0.2)
    #Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        data = [i[0] for i in data]
        plt.plot(padding + data)
        plt.legend()
    plt.show()

def series_to_supervised(data, n_in =1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = [], []
    #input sequence
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [("var%d(t-%d)" % (j+1, i)) for j in range(n_vars)]
    #forecast sequence
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [("var%d(t)" % (j+1)) for j in range(n_vars)]
        else:
            names += [("var%d(t-%d)" % (j+1, i)) for j in range(n_vars)]
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)
    return agg
def prepare_data(data, n_test, n_lag, n_seq, scaled = False, drop_cols=None):
    """
    data : np array
    n_test : test size
    n_lag : hur många dagar bakåt
    n_seq : hur många dagar framåt
    """
    if scaled:
        scaler = MinMaxScaler(feature_range=(0,1))
        data = scaler.fit_transform(data)
    reframed = series_to_supervised(data, n_lag, n_seq)
    if drop_cols != None:
        reframed.drop(reframed.columns[[12,13,14,15,16,17,18,19,20,21,22]], axis=1, inplace=True)
    values = reframed.values

    test_size = int(n_test*values.shape[0])

    train = values[:test_size, :]
    test = values[test_size:, :]
    train_x, train_y = train[:, :-1], train[:, -1]
    test_x, test_y = test[:, :-1], test[:, -1]
    train_x = train_x.reshape((train_x.shape[0], 1, train_x.shape[1]))
    test_x = test_x.reshape((test_x.shape[0], 1, test_x.shape[1]))
    if scaled:
        return train_x, train_y, test_x, test_y, scaler
    else:
        return train_x, train_y, test_x, test_y
def inverse_transform(test_x, preds, test_y, scaled=False, scaler=None):
    test_x = test_x.reshape((test_x.shape[0], test_x.shape[2]))
    inv_preds = np.concatenate((test_x[:, :-1], preds), axis=1)
    if scaled:
        inv_preds = scaler.inverse_transform(inv_preds)
    inv_preds = inv_preds[:,-1]

    test_y = test_y.reshape((len(test_y), 1))
    inv_y = np.concatenate((test_x[:, :-1], test_y), axis=1)
    if scaled:
        inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:,-1]
    return inv_preds, inv_y

