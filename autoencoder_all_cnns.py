from numpy import array
from keras.models import Sequential
from keras.models import Model
from keras.layers import LSTM, Input, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Dense, RepeatVector, TimeDistributed, Reshape
from keras.utils import plot_model
from keras.optimizers import RMSprop
import ffmpeg
import numpy as np
from generator_frame import MyGenerator
from data_fixed import Data
from matplotlib import pyplot as plt 


def get_ae_2x2(inputs):

    #encoder
    conv1 = Conv2D(16, (5, 5), strides=(1,1), activation='relu', padding='same')(inputs)
    pool1 = MaxPooling2D(pool_size=(5, 5))(conv1) #25 x 25 x 16
    conv2 = Conv2D(16, (5, 5), strides=(1,1), activation='relu', padding='same')(pool1)
    pool2 = MaxPooling2D(pool_size=(5, 5))(conv2) #25 x 25 x 16
    conv3 = Conv2D(8, (3, 3), strides=(1,1), activation='relu', padding='same')(pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3) #25 x 25 x 16
    conv4 = Conv2D(8, (3, 3), strides=(1,1), activation='relu', padding='same')(pool3)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4) #25 x 25 x 16
    conv5 = Conv2D(1, (3, 3), strides=(2,2), activation='relu', padding='same')(pool4)
     
    conv6 = Conv2DTranspose(8, (3, 3), strides=(2,2), activation='relu', padding='same')(conv5) # 100 x 100 x 8
    up2 = UpSampling2D((2,2))(conv6) # 400 x 400 x 8
    conv7 = Conv2DTranspose(8, (3, 3), strides=(1,1), activation='relu', padding='same')(up2) # 100 x 100 x 8
    up3 = UpSampling2D((2,2))(conv7) # 400 x 400 x 8
    conv8 = Conv2DTranspose(16, (5, 5), strides=(1,1), activation='relu', padding='same')(up3) # 100 x 100 x 8
    up4 = UpSampling2D((5,5))(conv8) # 400 x 400 x 8
    conv9 = Conv2DTranspose(16, (5, 5), strides=(1,1), activation='relu', padding='same')(up4) # 100 x 100 x 8
    up5 = UpSampling2D((5,5))(conv9) # 400 x 400 x 8
    decoded = Conv2DTranspose(1, (3, 3), strides=(1, 1), activation='relu', padding='same')(up5)
    return decoded

def get_ae_4x4(inputs):

    #encoder
    conv1 = Conv2D(16, (5, 5), strides=(1,1), activation='relu', padding='same')(inputs)
    pool1 = MaxPooling2D(pool_size=(5, 5))(conv1) #25 x 25 x 16
    conv2 = Conv2D(16, (5, 5), strides=(1,1), activation='relu', padding='same')(pool1)
    pool2 = MaxPooling2D(pool_size=(5, 5))(conv2) #25 x 25 x 16
    conv3 = Conv2D(8, (3, 3), strides=(1,1), activation='relu', padding='same')(pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3) #25 x 25 x 16
    conv4 = Conv2D(8, (3, 3), strides=(1,1), activation='relu', padding='same')(pool3)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4) #25 x 25 x 16
    conv5 = Conv2D(1, (3, 3), strides=(1,1), activation='relu', padding='same')(pool4)
      
    conv6 = Conv2DTranspose(8, (3, 3), strides=(1,1), activation='relu', padding='same')(conv5) # 100 x 100 x 8
    up2 = UpSampling2D((2,2))(conv6) # 400 x 400 x 8
    conv7 = Conv2DTranspose(8, (3, 3), strides=(1,1), activation='relu', padding='same')(up2) # 100 x 100 x 8
    up3 = UpSampling2D((2,2))(conv7) # 400 x 400 x 8
    conv8 = Conv2DTranspose(16, (5, 5), strides=(1,1), activation='relu', padding='same')(up3) # 100 x 100 x 8
    up4 = UpSampling2D((5,5))(conv8) # 400 x 400 x 8
    conv9 = Conv2DTranspose(16, (5, 5), strides=(1,1), activation='relu', padding='same')(up4) # 100 x 100 x 8
    up5 = UpSampling2D((5,5))(conv9) # 400 x 400 x 8
    decoded = Conv2DTranspose(1, (3, 3), strides=(1, 1), activation='relu', padding='same')(up5)
    return decoded

def get_ae_8x8(inputs):
    
    #encoder
    conv1 = Conv2D(16, (5, 5), strides=(1,1), activation='relu', padding='same')(inputs)
    pool1 = MaxPooling2D(pool_size=(5, 5))(conv1) #25 x 25 x 16
    conv2 = Conv2D(16, (5, 5), strides=(1,1), activation='relu', padding='same')(pool1)
    pool2 = MaxPooling2D(pool_size=(5, 5))(conv2) #25 x 25 x 16
    conv3 = Conv2D(8, (3, 3), strides=(1,1), activation='relu', padding='same')(pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3) #25 x 25 x 16
    conv4 = Conv2D(1, (3, 3), strides=(1,1), activation='relu', padding='same')(pool3)
    
    conv7 = Conv2DTranspose(8, (3, 3), strides=(1,1), activation='relu', padding='same')(conv4) # 100 x 100 x 8
    up3 = UpSampling2D((2,2))(conv7) # 400 x 400 x 8
    conv8 = Conv2DTranspose(16, (5, 5), strides=(1,1), activation='relu', padding='same')(up3) # 100 x 100 x 8
    up4 = UpSampling2D((5,5))(conv8) # 400 x 400 x 8
    conv9 = Conv2DTranspose(16, (5, 5), strides=(1,1), activation='relu', padding='same')(up4) # 100 x 100 x 8
    up5 = UpSampling2D((5,5))(conv9) # 400 x 400 x 8
    decoded = Conv2DTranspose(1, (3, 3), strides=(1, 1), activation='relu', padding='same')(up5)
    return decoded

def get_ae_16x16(inputs):
    
    #encoder
    conv1 = Conv2D(16, (5, 5), strides=(1,1), activation='relu', padding='same')(inputs)
    pool1 = MaxPooling2D(pool_size=(5, 5))(conv1) #25 x 25 x 16
    conv2 = Conv2D(16, (5, 5), strides=(1,1), activation='relu', padding='same')(pool1)
    pool2 = MaxPooling2D(pool_size=(5, 5))(conv2) #25 x 25 x 16
    conv3 = Conv2D(1, (3, 3), strides=(1,1), activation='relu', padding='same')(pool2)
    
    conv8 = Conv2DTranspose(16, (5, 5), strides=(1,1), activation='relu', padding='same')(conv3) # 100 x 100 x 8
    up4 = UpSampling2D((5,5))(conv8) # 400 x 400 x 8
    conv9 = Conv2DTranspose(16, (5, 5), strides=(1,1), activation='relu', padding='same')(up4) # 100 x 100 x 8
    up5 = UpSampling2D((5,5))(conv9) # 400 x 400 x 8
    decoded = Conv2DTranspose(1, (3, 3), strides=(1, 1), activation='relu', padding='same')(up5)
    return decoded

def get_ae_5x5(inputs):
    
    #encoder
    conv1 = Conv2D(16, (3, 3), strides=(1,1), activation='relu', padding='same')(inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) #25 x 25 x 16
    conv2 = Conv2D(16, (3, 3), strides=(1,1), activation='relu', padding='same')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) #25 x 25 x 16
    conv3 = Conv2D(8, (3, 3), strides=(1,1), activation='relu', padding='same')(pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3) #25 x 25 x 16
    conv4 = Conv2D(8, (3, 3), strides=(1,1), activation='relu', padding='same')(pool3)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4) #25 x 25 x 16   
    conv5 = Conv2D(1, (5, 5), strides=(5,5), activation='relu', padding='same')(pool4)
    
    conv6 = Conv2DTranspose(8, (5, 5), strides=(5,5), activation='relu', padding='same')(conv5) # 100 x 100 x 8
    up1 = UpSampling2D((2,2))(conv6) # 400 x 400 x 8
    conv7 = Conv2DTranspose(8, (3, 3), strides=(1,1), activation='relu', padding='same')(up1) # 100 x 100 x 8
    up2 = UpSampling2D((2,2))(conv7) # 400 x 400 x 8
    conv8 = Conv2DTranspose(16, (3, 3), strides=(1,1), activation='relu', padding='same')(up2) # 100 x 100 x 8
    up3 = UpSampling2D((2,2))(conv8) # 400 x 400 x 8
    conv9 = Conv2DTranspose(16, (3, 3), strides=(1,1), activation='relu', padding='same')(up3) # 100 x 100 x 8
    up4 = UpSampling2D((2,2))(conv9) # 400 x 400 x 8
    decoded = Conv2DTranspose(1, (3, 3), strides=(1, 1), activation='relu', padding='same')(up4)
    return decoded

def get_ae_25x25(inputs):
    
    #encoder
    conv1 = Conv2D(16, (3, 3), strides=(1,1), activation='relu', padding='same')(inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) #25 x 25 x 16
    conv2 = Conv2D(16, (3, 3), strides=(1,1), activation='relu', padding='same')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) #25 x 25 x 16
    conv3 = Conv2D(8, (3, 3), strides=(1,1), activation='relu', padding='same')(pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3) #25 x 25 x 16
    conv4 = Conv2D(8, (3, 3), strides=(1,1), activation='relu', padding='same')(pool3)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4) #25 x 25 x 16   
    conv5 = Conv2D(1, (3, 3), strides=(1,1), activation='relu', padding='same')(pool4)
    
    conv6 = Conv2DTranspose(8, (3, 3), strides=(1,1), activation='relu', padding='same')(conv5) # 100 x 100 x 8
    up1 = UpSampling2D((2,2))(conv6) # 400 x 400 x 8
    conv7 = Conv2DTranspose(8, (3, 3), strides=(1,1), activation='relu', padding='same')(up1) # 100 x 100 x 8
    up2 = UpSampling2D((2,2))(conv7) # 400 x 400 x 8
    conv8 = Conv2DTranspose(16, (3, 3), strides=(1,1), activation='relu', padding='same')(up2) # 100 x 100 x 8
    up3 = UpSampling2D((2,2))(conv8) # 400 x 400 x 8
    conv9 = Conv2DTranspose(16, (3, 3), strides=(1,1), activation='relu', padding='same')(up3) # 100 x 100 x 8
    up4 = UpSampling2D((2,2))(conv9) # 400 x 400 x 8
    decoded = Conv2DTranspose(1, (3, 3), strides=(1, 1), activation='relu', padding='same')(up4)
    return decoded



input_dims = (400, 400, 1)
inputs = Input(shape=input_dims)
batch_size = 16
shuffle = True

data = np.load("prepared_ids.npy")
data_count = len(data)
separator = int(data_count*0.8)
train_data = data[:separator]
validation_data = data[separator:]

data_generator = MyGenerator(train_data, batch_size=batch_size, dim=input_dims, shuffle=shuffle)
validation_generator = MyGenerator(validation_data, batch_size=batch_size, dim=input_dims, shuffle=shuffle)

all_histories = []
for i in range(6):
    if i==0:
        autoencoder = Model(inputs, get_ae_2x2(inputs))
    elif i==1:
        autoencoder = Model(inputs, get_ae_4x4(inputs))
    elif i==2:
        autoencoder = Model(inputs, get_ae_8x8(inputs))
    elif i==3:
        autoencoder = Model(inputs, get_ae_16x16(inputs))
    elif i==4:
        autoencoder = Model(inputs, get_ae_5x5(inputs))
    elif i==5:
        autoencoder = Model(inputs, get_ae_25x25(inputs))
    autoencoder.compile(loss='mean_squared_error', optimizer = RMSprop()) 
    autoencoder.summary()

    history = autoencoder.fit_generator(generator=data_generator, validation_data=validation_generator, use_multiprocessing=True, workers=16, max_queue_size=16, epochs=20)
    
    autoencoder.save("model_cnn_run" + str(i) + ".h5")

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
    plt.ylim(top=0.25)
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig("cnn_loss_run" + str(i) + ".png")
    plt.close()

    all_histories.append(history)

print("#####\n#####\n#####ALL HISTORIES HERE!!!\n#####\n#####\n#####")
for history in all_histories:
    print(history.history)

