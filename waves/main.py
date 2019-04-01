from kek import *
from mywaves import *
from prepros import *

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

data = pd.read_csv("C:/Users/Phili/Desktop/fond/data/SAND.csv", sep=";", header = 0, index_col = 0, parse_dates = True).iloc[::-1]
data = data["Closing price"]
mw = MyWavelet(data, "db4")
v, w_j = mw.mra()

rets = data.pct_change().dropna()
input_dim = (None, 1)
output_dim = 1
print("SHAPE: ", w_j.shape)
print(input_dim, output_dim)



data = np.c_[w_j[1:, :],rets]

train_x, train_y, test_x, test_y = prepare_data(data, 0.1   , 1,1, scaled=False)

input_dim = (train_x.shape[1], train_x.shape[2])
output_dim = 1


period = 1
lstm = MyLSTM(input_dim, output_dim, n_hidden=0, n_neurons_per_layer=10, dropout_percentage=0.1)
model = lstm.create_regression_model()
lstm.fit_model(train_x, train_y, epochs=1, batch_size=period, validation_split=0.05)
preds, real = lstm.adaptive_pred(test_x, test_y, period)
print(preds[0], test_y.shape)

plt.subplot(1,2,1)
plt.plot(preds.squeeze(), label="preds", color="black")
plt.plot(test_y, label="real", color="gray", alpha=0.5)
plt.legend()
signals = []
for i in range(len(preds)):
    s = np.sign(preds[i])
    if s == 1:
        signals.append(1)
    elif s == -1:
        signals.append(-1)
    else:
        signals.append(0)
signals = np.array(signals)
#pos = np.diff(signals)
r = real*signals
ec = (1+r).cumprod()
plt.subplot(1,2,2)
plt.plot(ec)
plt.show()
