import numpy as np
import ffmpeg
import keras
from keras.backend import clear_session

class MyGenerator(keras.utils.Sequence):
    #Generates data for Keras
    def __init__(self, list_IDs, model, batch_size=64, lstm_dim=(20,100), video_dim=(20,400,400,1), shuffle=True):
        #nitialization
        self.video_dim = video_dim
        self.lstm_dim = lstm_dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.model = model
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
        videos = np.empty((self.batch_size, *self.video_dim))
        X = np.empty((self.batch_size, self.lstm_dim[0] - 1, self.lstm_dim[1]))
        y = np.empty((self.batch_size, 1, self.lstm_dim[1]))

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
                .reshape([self.video_dim[0], self.video_dim[1], self.video_dim[2], self.video_dim[3]])
            )   
            

            video = video/255
            video = video.reshape([-1, 400, 400, 1])
            self.model._make_predict_function()
            features = self.model.predict(video)
            base_features = features[:-1]
            predict_features = features[-1]            

            X[i] = np.reshape(base_features, (-1, self.lstm_dim[1]))
            y[i] = np.reshape(predict_features, (1, self.lstm_dim[1]))
        print(X.shape())
        print(Y.shape())
        return X, y
