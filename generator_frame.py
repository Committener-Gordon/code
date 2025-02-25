import numpy as np
import ffmpeg
import keras

class MyGenerator(keras.utils.Sequence):
    #Generates data for Keras
    def __init__(self, list_IDs, batch_size=64, dim=(400,400,1), shuffle=True):
        #nitialization
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        #enotes the number of batches per epoch
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        #enerate one batch of data
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        #pdates indexes after each epoch
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        #enerates data containing batch_size samples # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        #y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            #X[i,] = np.load('data/' + ID + '.npy')

            # Store class
            #y[i] = self.labels[ID]
            out, _ = (
                ffmpeg
                .input(ID)
                .output('pipe:', format='rawvideo', pix_fmt='gray')
                .run(quiet=True)
            )
            video = (
                np
                .frombuffer(out, np.uint8)
                .reshape([-1, self.dim[0], self.dim[1], self.dim[2]])
            )
            
            frame = video[0] / 255
            X[i] = frame
        return X, X
