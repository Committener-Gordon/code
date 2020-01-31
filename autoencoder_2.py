from numpy import array
from keras.models import Sequential
from keras.models import Model
from keras.layers import LSTM, Input, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Dense, RepeatVector, TimeDistributed, Reshape
from keras.utils import plot_model
from keras.optimizers import RMSprop
import ffmpeg
import numpy as np
from generator import MyGenerator
from data import Data

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def get_autoencoder(inputs):                             

        
    #encoder
    conv1 = TimeDistributed(Conv2D(16, (9, 9), strides=(3,3), activation='relu', padding='same'))(inputs)
    print(conv1.shape)
    pool1 = TimeDistributed(MaxPooling2D(pool_size=(2, 2), padding='same'))(conv1) #25 x 25 x 16

    conv2 = TimeDistributed(Conv2D(16, (9, 9), strides=(3,3), activation='relu', padding='same'))(pool1)
    print(conv2.shape)
    #pool2 = TimeDistributed(MaxPooling2D(pool_size=(2, 2), padding='same'))(conv2) #25 x 25 x 16
    
    #print(pool2.shape)
    
    reshape = Reshape((inputs.shape[1], 8464))(conv2)
    dense1 = TimeDistributed(Dense(100))(reshape)
    lstm1 = LSTM(100, return_sequences=True)(dense1)
    lstmviz = LSTM(2, return_sequences=True)(lstm1)
    
    lstm2 = LSTM(100, return_sequences=True)(lstmviz)
    dense2 = TimeDistributed(Dense(8464))(lstm2)
    reshape2 = Reshape((-1, 23, 23, 16))(dense2)
     
        
    #up2 = TimeDistributed(UpSampling2D((2,2)))(reshape2) # 400 x 400 x 8
    conv5 = TimeDistributed(Conv2DTranspose(16, (9, 9), strides=(3,3), activation='relu', padding='same', output_padding=(0, 0)))(reshape2) # 100 x 100 x 8
    print(conv5.shape)

    up3 = TimeDistributed(UpSampling2D((2,2)))(conv5) # 400 x 400 x 8
    conv6 = TimeDistributed(Conv2DTranspose(16, (9, 9), strides=(3,3), activation='relu', padding='same', output_padding=(0,0)))(up3) # 100 x 100 x 8
    print(conv6.shape)
    decoded = TimeDistributed(Conv2D(1, (3, 3), strides=(1,1), activation='relu', padding='same'))(conv6)
    
    #conv1 = TimeDistributed(Conv2D(32, (3, 3), strides=(1,1), activation='relu', padding='same'))(inputs)
    #pool1 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv1) #200 x 200 x 32
    #conv2 = TimeDistributed(Conv2D(16, (3, 3), strides=(1,1), activation='relu', padding='same'))(pool1) #200 x 200 x 16
    #pool2 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv2) #100 x 100 x 16
    #conv3 = TimeDistributed(Conv2D(8, (3, 3), strides=(1,1), activation='relu', padding='same'))(pool2) #100 x 100 x 8    
    
    #reshape = Reshape((inputs.shape[1], 80000))(conv3)
    #dense1 = TimeDistributed(Dense(100))(reshape)
    #lstm1 = LSTM(100, return_sequences=True)(dense1)
   
    #decoder
    #lstm2 = LSTM(100, return_sequences=True)(lstm1)
    #dense2 = TimeDistributed(Dense(80000))(lstm2)
    #reshape2 = Reshape((-1, 100, 100, 8))(dense2)
    #conv4 = TimeDistributed(Conv2D(8, (3, 3), strides=(1,1), activation='relu', padding='same'))(reshape2) #100 x 100 x 8
    #up1 = TimeDistributed(UpSampling2D((2,2)))(conv4) # 200 x 200 x 128
    #conv5 = TimeDistributed(Conv2D(16, (3, 3), strides=(1,1), activation='relu', padding='same'))(up1) # 200 x 200 x 16
    #up2 = TimeDistributed(UpSampling2D((2,2)))(conv5) # 400 x 400 x 16
    #decoded = TimeDistributed(Conv2D(1, (3, 3), strides=(1,1), activation='relu', padding='same'))(up2) # 400 x 400 x 1
    return decoded




input_dims = (20, 400, 400, 1)
inputs = Input(shape=input_dims)
inputs2 = Input(shape=input_dims)
batch_size = 8
shuffle = True

data = Data.get_data_paths("./calving", "./rcnn", percentage_a=1,  percentage_b=0.1145)
data_generator = MyGenerator(data, batch_size=batch_size, dim=input_dims, shuffle=shuffle)
print("###\n###")
print("Data amount: " + str(len(data)))
print("###\n###")

autoencoder = Model(inputs, get_autoencoder(inputs))
autoencoder.compile(loss='mean_squared_error', optimizer = RMSprop()) 
autoencoder.summary()

autoencoder.fit_generator(generator=data_generator, use_multiprocessing=True, workers=4, max_queue_size=4, epochs=5)

autoencoder.save("model_2conv_viz.h5")
