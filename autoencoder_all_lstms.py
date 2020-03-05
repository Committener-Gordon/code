from numpy import array
from keras.models import Sequential
from keras.models import Model, load_model
from keras.layers import LSTM, Input, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Dense, RepeatVector, TimeDistributed, Reshape
from keras.utils import plot_model
from keras.optimizers import RMSprop
import ffmpeg
import numpy as np
from generator_lstm import MyGenerator
from data_fixed import Data
from matplotlib import pyplot as plt 
import tensorflow as tf

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def get_autoencoder(inputs):                             
     
    #encoder
    lstm1 = LSTM(16, return_sequences=True)(inputs)
    #decoder
    lstm2 = LSTM(int(inputs.shape[-1]), return_sequences=True)(lstm1)
    return lstm2


video_input_dims = (20, 400, 400, 1)
video_inputs = Input(shape=video_input_dims)

batch_size = 8
shuffle = True

#data = Data.get_data_paths("./calving", "./rcnn", percentage_a=1,  percentage_b=1)
data = np.load("prepared_ids.npy")
data_count = len(data)
separator = int(data_count*0.8)
train_data = data[:separator]
validation_data = data[separator:]

print("Data count: " + str(data_count))
print("Train Data: " + str(len(train_data)))
print("Validation Data: " + str(len(validation_data)))


for i in range(8):

    if i == 0:
        cnn_model = loaded_model = load_model("model_cnn_run0.h5")
        cnn_encoder = Model(cnn_model.inputs, cnn_model.layers[-10].output)
    elif i == 1:
        cnn_model = loaded_model = load_model("model_cnn_run1.h5")
        cnn_encoder = Model(cnn_model.inputs, cnn_model.layers[-10].output)
    elif i == 2:
        cnn_model = loaded_model = load_model("model_cnn_run2.h5")
        cnn_encoder = Model(cnn_model.inputs, cnn_model.layers[-8].output)
    elif i == 3:
        cnn_model = loaded_model = load_model("model_cnn_run3.h5")
        cnn_encoder = Model(cnn_model.inputs, cnn_model.layers[-6].output)
    elif i == 4:
        cnn_model = loaded_model = load_model("model_cnn_run4.h5")
        cnn_encoder = Model(cnn_model.inputs, cnn_model.layers[-10].output)
    elif i == 5:
        cnn_model = loaded_model = load_model("model_cnn_run5.h5")
        cnn_encoder = Model(cnn_model.inputs, cnn_model.layers[-10].output)
    elif i == 6:
        cnn_model = loaded_model = load_model("model_cnn_run6.h5")
        cnn_encoder = Model(cnn_model.inputs, cnn_model.layers[-6].output)
    elif i == 7:
        cnn_model = loaded_model = load_model("model_cnn_run7.h5")
        cnn_encoder = Model(cnn_model.inputs, cnn_model.layers[-5].output)
    
    #calculate the amount of lstm cells based on the output layer shape of the cnn_encoder
    cnn_output_1 = int(cnn_encoder.output.shape[1])
    cnn_output_2 = int(cnn_encoder.output.shape[2])
    lstm_cells = cnn_output_1 * cnn_output_2
    lstm_input_dims = (20, lstm_cells)
    lstm_inputs = Input(shape=lstm_input_dims)
    print(lstm_inputs.shape)
    
    cnn_encoder.summary()
    
    data_generator = MyGenerator(train_data, batch_size=batch_size, model=cnn_encoder, 
    video_dim=video_input_dims, lstm_dim=lstm_input_dims, shuffle=shuffle)
    validation_generator = MyGenerator(validation_data, batch_size=batch_size, model=cnn_encoder,
    video_dim=video_input_dims, lstm_dim=lstm_input_dims, shuffle=shuffle)
    print("###\n###")
    print("Data amount: " + str(len(data)))
    print("###\n###")

    autoencoder = Model(lstm_inputs, get_autoencoder(lstm_inputs))
    autoencoder.compile(loss='mean_squared_error', optimizer = RMSprop()) 
    autoencoder.summary()


    history = autoencoder.fit_generator(generator=data_generator, use_multiprocessing=False, workers=1, max_queue_size=1, epochs=20)

    autoencoder.save("model_lstm_run" + str(i) + ".h5")

    print("\n\n\n\nLoss")
    print(history.history['loss'])
    print("\n\n\n\nValidation Loss")
    print(history.history['val_loss'])


    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig("lstm_loss_run" + str(i) + ".png")

