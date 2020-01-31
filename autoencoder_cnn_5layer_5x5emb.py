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

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def get_autoencoder(inputs):                             

        
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
    print(conv3.shape)
    
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




input_dims = (400, 400, 1)
inputs = Input(shape=input_dims)
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

data_generator = MyGenerator(train_data, batch_size=batch_size, dim=input_dims, shuffle=shuffle)
validation_generator = MyGenerator(validation_data, batch_size=batch_size, dim=input_dims, shuffle=shuffle)
print("###\n###")
print("Data amount: " + str(len(data)))
print("###\n###")

autoencoder = Model(inputs, get_autoencoder(inputs))
autoencoder.compile(loss='mean_squared_error', optimizer = RMSprop()) 
autoencoder.summary()

history = autoencoder.fit_generator(generator=data_generator, validation_data=validation_generator, use_multiprocessing=False, workers=1, max_queue_size=1, epochs=20)

autoencoder.save("model_cnn_3layers.h5")

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
plt.savefig("conv3d_acc.png")

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig("conv3d_loss.png")

