from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Flatten, Reshape
from keras.optimizers import Adam
import numpy as np

class MyLSTM:
    def __init__(self, input_dim, output_dim, n_hidden, n_neurons_per_layer, dropout_percentage):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_hidden = n_hidden
        self.n_neurons_per_layer = n_neurons_per_layer
        self.dropout_percentage = dropout_percentage
    def create_regression_model(self, loss="mse", optimizer="rmsprop"):
        model = Sequential()
        model.add(LSTM(
            input_shape = self.input_dim,
            units = self.n_neurons_per_layer,
            return_sequences = True
        ))
        for _ in range(self.n_hidden):
            model.add(LSTM(
                units = self.n_neurons_per_layer,
                return_sequences = True #Om denna e false squishas den till en array.
            ))
            model.add(Dropout(self.dropout_percentage))
        
        model.add(LSTM(
                units = self.n_neurons_per_layer,
                return_sequences = False #Om denna e false squishas den till en array.
        ))
        model.add(Dropout(self.dropout_percentage))
        
        model.add(Dense(1))
        model.compile(loss=loss, optimizer = optimizer)
        self.model = model
        print(model.summary())
        return model
    def create_classification_model(self, loss="categorical_crossentropy", optimizer="rmsprop"):
        model = Sequential()
        model.add(LSTM(
            input_shape = self.input_dim,
            units = self.n_neurons_per_layer,
            return_sequences = True
        ))
        for _ in range(self.n_hidden):
            model.add(LSTM(
                units = self.n_neurons_per_layer,
                return_sequences = True #Om denna e false squishas den till en array.
            ))
            model.add(Dropout(self.dropout_percentage))
        
        model.add(LSTM(
                units = self.n_neurons_per_layer,
                return_sequences = False #Om denna e false squishas den till en array.
        ))
        model.add(Dropout(self.dropout_percentage))
        
        model.add(Dense(self.output_dim, activation="sigmoid"))
        model.compile(loss=loss, optimizer = optimizer, metrics=["accuracy"])
        self.model = model
        print(model.summary())
        return model
    def fit_model(self, x_train, y_train, batch_size, epochs, validation_split=0.05):
        self.model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs, validation_split=validation_split)
    def pred_seq_full(self, x_test, window_size):
        curr_frame = x_test[0]
        predicted = []
        for i in range(len(x_test)):
            pred = self.model.predict(curr_frame[np.newaxis, :,:])
            print(pred)
            predicted.append(pred[0,0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
        return predicted
    def pred_seq_multiple(self, x_test, window_size, prediction_len):
        prediction_seqs = []
        print(len(x_test))
        for i in range(int(len(x_test)/prediction_len)):
            curr_frame = x_test[i*prediction_len]
            predicted = []
            for j in range(prediction_len):
                pred = self.model.predict(curr_frame[np.newaxis, :,:])
                predicted.append(pred[0,0])
                curr_frame = curr_frame[1:]
                curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
            prediction_seqs.append(curr_frame)
        return prediction_seqs
    def adaptive_pred(self, x_test,y_test,period):
        preds = []
        real = []
        for i in range(1, len(y_test)+1):
            if i % period == 0:
                curr_data = x_test[(i-period):i, :,:]
                curr_real = y_test[(i-period):i]
                pred = self.model.predict(curr_data)
                #if period > 1:
                real.append(curr_real[0])
                preds.append(pred[0])
               # else:
               #     real.append(curr_real)
               #     preds.append(pred)
                self.model.fit(curr_data, curr_real, batch_size=period, epochs=1, verbose=0)
        return np.array(preds), np.array(real)






    
