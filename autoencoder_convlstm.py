from numpy import array
from keras.models import Sequential
from keras.models import Model
from keras.layers import LSTM, Input, Conv3D, Conv3DTranspose, MaxPooling3D, UpSampling3D, Dense, RepeatVector, TimeDistributed, Reshape, ConvLSTM2D, BatchNormalization
from keras.utils import plot_model
from keras.optimizers import RMSprop
import ffmpeg
import numpy as np
from generator import MyGenerator
from data import Data
from matplotlib import pyplot as plt 

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def get_autoencoder(inputs):                             

        
    #encoder
    conv1 = ConvLSTM2D(16, (4, 4), strides=(2,2), recurrent_activation='hard_sigmoid', activation='tanh', padding='same', return_sequences=True)(inputs)
    print(conv1.shape)
    norm1 = BatchNormalization()(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(norm1) #25 x 25 x 16
    conv2 = ConvLSTM2D(8, (4, 4), strides=(2,2), recurrent_activation='hard_sigmoid', activation='tanh', padding='same', return_sequences=True)(pool1)
    print(conv2.shape)
    norm2 = BatchNormalization()(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(norm2) #25 x 25 x 16
    
    print(pool2.shape)
    
    conv3 = ConvLSTM2D(1, (5, 5), strides=(5,5), recurrent_activation='hard_sigmoid', activation='tanh', padding='same', return_sequences=True)(pool2)
    print(conv3.shape)
    norm3 = BatchNormalization()(conv3)
    up2 = UpSampling3D((2,5,5))(norm3)
    conv4 = ConvLSTM2D(8, (4, 4), strides=(1,1), recurrent_activation='hard_sigmoid', activation='tanh', padding='same', return_sequences=True)(up2)
    up3 = UpSampling3D((2,4,4))(conv4)
    conv5 = ConvLSTM2D(16, (4, 4), strides=(1,1), recurrent_activation='hard_sigmoid', activation='tanh', padding='same', return_sequences=True)(up3)
    up4 = UpSampling3D((1,4,4))(conv5)
    conv6 = ConvLSTM2D(1, (4, 4), strides=(1,1), recurrent_activation='hard_sigmoid', activation='tanh', padding='same', return_sequences=True)(up4)
    return conv6

    
    #reshape = Reshape((1, 25000))(pool2)
    #dense1 = Dense(100)(reshape)
    #dense2 = Dense(3125)(dense1)
    #reshape2 = Reshape((-1, 25, 25, 1))(dense2)
        
    #conv5 = Conv3DTranspose(8, (4, 4, 4), strides=(2,2,2), activation='relu', padding='same')(reshape2) # 100 x 100 x 8
    #up2 = UpSampling3D((1,2,2))(conv5) # 400 x 400 x 8
    #conv6 = Conv3DTranspose(16, (4, 4, 4), strides=(2,2,2), activation='relu', padding='same')(up2) # 100 x 100 x 8
    #up3 = UpSampling3D((1,2,2))(conv6) # 400 x 400 x 8
    #decoded = Conv3D(1, (3, 3, 3), strides=(1,1, 1), activation='relu', padding='same')(up3)
    
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
    #return decoded




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
plot_model(autoencoder, to_file="convlstm_model.png")

history = autoencoder.fit_generator(generator=data_generator, use_multiprocessing=True, workers=4, max_queue_size=4, epochs=5)

autoencoder.save("model_convlstm.h5")

print("Accuracy")
print(history.history['acc'])
print("\n\n\n\nValidation Accuracy")
print(history.history['val_acc'])
print("\n\n\n\nLoss")
print(history.history['loss'])
print("\n\n\n\nValidation Loss")
print(history.history['val_loss'])

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig("convlstm_acc.png")

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig("convlstm_loss.png")

