from keras.models import Model, load_model
from keras.layers import Input, Reshape, TimeDistributed
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering, AgglomerativeClustering
import ffmpeg
import numpy as np
from data import Data
import random
from tabulate import tabulate
from matplotlib import pyplot as plt
from generator import MyGenerator


input_dims = (20, 400, 400, 1)
inputs = Input(shape=input_dims)


def load_concat_model(inputs, number=0, cnn_decoder_length=10):
    cnn_model = load_model("model_cnn_run" + str(number) + ".h5")
    cnn_encoder = Model(cnn_model.layers[0].input, cnn_model.layers[-1].output)
    
    print(cnn_model.layers[1-cnn_decoder_length].input)

    #cnn_decoder = Model(cnn_model.layers[1 - cnn_decoder_length].input, cnn_model.layers[-1].output)
    
    time_cnn = TimeDistributed(cnn_encoder)(inputs)


    decode_layer = cnn_model.layers[1-cnn_decoder_length]

    decode_output = cnn_model.layers[-1]

    
    Model(decode_layer, decode_output)


    reshape1 =  Reshape((20, 4))(time_cnn)

    lstm_model = load_model("model_lstm_run" + str(number) + ".h5")

    lstm_ae = lstm_model(reshape1)

    reshape2 = Reshape((20, 2, 2, 1))(lstm_ae)

    bla = TimeDistributed(decode_layer)(reshape2)

    #decoder = TimeDistributed(cnn_decoder)(reshape2)

    return Model(inputs, bla)


autoencoder = load_concat_model(inputs, 0, 10)
autoencoder.summary()
