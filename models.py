#---------------------------------------------------------------------------------------------------#
# File name: models.py                                                                              #
# Autor: Chrissi2802                                                                                #
# Created on: 08.09.2022                                                                            #
#---------------------------------------------------------------------------------------------------#
# Google Brain - Ventilator Pressure Prediction (VPP)
# Exact description in the functions.
# This file provides the models.


from tensorflow.keras.models import Model
import tensorflow.keras.layers as layer


def gru_net(data):

    x_input = layer.Input(shape = (data.shape[-2:]))

    x = layer.Bidirectional(layer.GRU(256, return_sequences = True))(x_input)
    x = layer.BatchNormalization(axis = -1)(x)
    x = layer.Bidirectional(layer.GRU(128, return_sequences = True))(x)
    x = layer.BatchNormalization(axis = -1)(x)

    x = layer.Dense(128, activation = "relu")(x)
    x = layer.BatchNormalization(axis = -1)(x)
    x = layer.Dropout(0.5)(x)
    x_output = layer.Dense(1)(x)

    model = Model(inputs = x_input, outputs = x_output, name = "GRU_NET")

    return model


def gru_net_big(data):

    x_input = layer.Input(shape = (data.shape[-2:]))

    x = layer.Bidirectional(layer.GRU(1024, return_sequences = True))(x_input)
    x = layer.BatchNormalization(axis = -1)(x)
    x = layer.Bidirectional(layer.GRU(512, return_sequences = True))(x)
    x = layer.BatchNormalization(axis = -1)(x)

    x = layer.Bidirectional(layer.GRU(256, return_sequences = True))(x)
    x = layer.BatchNormalization(axis = -1)(x)
    x = layer.Bidirectional(layer.GRU(128, return_sequences = True))(x)
    x = layer.BatchNormalization(axis = -1)(x)

    x = layer.Dense(128, activation = "relu")(x)
    x = layer.BatchNormalization(axis = -1)(x)
    x = layer.Dropout(0.5)(x)
    x_output = layer.Dense(1)(x)

    model = Model(inputs = x_input, outputs = x_output, name = "GRU_NET_BIG")

    return model


def lstm_net_big(data):

    x_input = layer.Input(shape = (data.shape[-2:]))

    x = layer.Bidirectional(layer.LSTM(1024, return_sequences = True))(x_input)
    x = layer.BatchNormalization(axis = -1)(x)
    x = layer.Bidirectional(layer.LSTM(512, return_sequences = True))(x)
    x = layer.BatchNormalization(axis = -1)(x)

    x = layer.Bidirectional(layer.LSTM(256, return_sequences = True))(x)
    x = layer.BatchNormalization(axis = -1)(x)
    x = layer.Bidirectional(layer.LSTM(128, return_sequences = True))(x)
    x = layer.BatchNormalization(axis = -1)(x)

    x = layer.Dense(128, activation = "relu")(x)
    x = layer.BatchNormalization(axis = -1)(x)
    x = layer.Dropout(0.5)(x)
    x_output = layer.Dense(1)(x)

    model = Model(inputs = x_input, outputs = x_output, name = "LSTM_NET_BIG")

    return model

    
if (__name__ == "__main__"):

    import dataset

    # Dataset
    CVPP_Datasets = dataset.VPP_Datasets(create_new = False, many_features = True)
    train, target, test = CVPP_Datasets.get_datasets()

    gru_small = gru_net(train)
    gru_big = gru_net_big(train)
    lstm_big = lstm_net_big(train)
