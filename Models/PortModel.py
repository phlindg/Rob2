

from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Dropout, Input, Conv2D, Flatten, concatenate, Activation
import keras
import numpy as np
from prepros import *

#https://datascience.stackexchange.com/questions/33389/how-can-this-cnn-for-the-portfolio-management-problem-be-implemented-in-keras
class PortModel:
    def __init__(self, input_dim, weights_dim):
        self.input_dim = input_dim
        self.weights_dim = weights_dim
        
    def create_model_conv(self, loss="mse", optimizer="rmsprop"):
        """
        Den ska ha LSTM layers för returnsen
        Men också ta in tidigare weights som input
        """
        price_history = Input(shape=self.input_dim)
        feature_maps = Conv2D(2, (1,3), activation="relu")(price_history)
        feature_maps = Conv2D(20, (1, 48), activation="relu")(feature_maps)

        w_last = Input(shape=self.weights_dim)
        feature_maps = concatenate([feature_maps, w_last], axis=-1)
        feature_map = Conv2D(1, (1,1), activation="relu")(feature_maps)
        feature_map = Flatten()(feature_map)
        w = Activation("softmax")(feature_map)

        model = Model(inputs=[price_history, w_last], outputs=w)
        model.compile(loss=loss, optimizer = optimizer)
        print(model.summary())
        self.model = model
        return model
    def create_model_lstm(self, loss="mse", optimizer = "rmsprop"):
        """
        FEL PÅ INPUT SHAPE LSTM
        """
        price_history = Input(shape=self.input_dim)
        feature_maps = LSTM(units=20)(price_history)

        w_last = Input(shape=self.weights_dim)
        feature_maps = concatenate([feature_maps, w_last], axis=-1)
        feature_map = Conv2D(1, (1,1), activation="relu")(feature_maps)
        feature_map = Flatten()(feature_map)
        w = Activation("softmax")(feature_map)

        model = Model(inputs=[price_history, w_last], outputs = w)
        print(model.summary())
        self.model = model
    def fit_model(self, x_train, y_train, epochs, batch_size):
        self.model.fit(x_train, y_train, epochs=epochs, batch_size = batch_size, validation_split=0.05)

