from keras.models import Model, load_model
from keras.layers import Input, Reshape, TimeDistributed
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import ffmpeg
import numpy as np
from data import Data
import random
from tabulate import tabulate
from matplotlib import pyplot as plt
from generator import MyGenerator

CHUNK_SIZE = 20

def load_model_encoder(inputs, number=0, cnn_decoder_length=10):
    cnn_model = load_model("model_cnn_run" + str(number) + ".h5")
    cnn_encoder = Model(cnn_model.inputs, cnn_model.layers[-cnn_decoder_length].output)
    
    time_cnn = TimeDistributed(cnn_encoder)(inputs)

    reshape =  Reshape((19, 4))(time_cnn)

    lstm_model = load_model("model_lstm_predict_run" + str(number) + ".h5")
    lstm_encoder = Model(lstm_model.inputs, lstm_model.layers[-3].output)

    embedding = lstm_encoder(reshape)
    return Model(inputs, embedding)
    


input_dims = (19, 400, 400, 1)
inputs = Input(shape=input_dims)

my_model = load_model_encoder(inputs, 0, 10)
my_model.summary()

data_set = Data.get_data_paths("./calving_samples", "./random_samples", percentage_a=0.1, percentage_b=0.8)

chunks = int(len(data_set) / CHUNK_SIZE)
print("Datensatz: " + str(len(data_set)))
print("Chunks: " + str(chunks))

data_set = data_set[: chunks * CHUNK_SIZE]

print("Getrimmter Datensatz: " + str(len(data_set)))

big_X = []
big_y = []

for i in range(chunks):
    X = []
    current_data = data_set[i*CHUNK_SIZE:(i+1)*CHUNK_SIZE]
    for file in current_data:
        out, _ = (
	        ffmpeg
	        .input(file)
	        .output('pipe:', format='rawvideo', pix_fmt='gray')
	        .run(quiet=True)
        )
        video = (
	        np
	        .frombuffer(out, np.uint8)
	        .reshape([20, 400, 400, 1])
        )
        X.append((video/255)[:19])	
        if "calving" in file:
            #calved_count += 1
            big_y.append(0)
        else:
            #random_count += 1
            big_y.append(1)
        print(file)

    
    print(np.shape(X))
    video_origin = np.array(X)

    embedding = my_model.predict(video_origin)
    embedding = embedding.reshape(-1, 16)
    print(np.shape(embedding))

    big_X.extend(embedding)
    print("Shape of big_X: " + str(np.shape(big_X)))
print(np.shape(big_y))
print("Amount zeros: " + str(big_y.count(0)))
print("Amount ones: " + str(big_y.count(1)))

test_X = [*big_X[0:10], *big_X[len(big_X)-10 : len(big_X)]]
big_X = big_X[10:len(big_X)-10]

test_y = [*big_y[0:10], *big_y[len(big_y)-10 : len(big_y)]]
big_y = big_y[10:len(big_y)-10]

print("Shape of big_X: " + str(np.shape(big_X)))
print("Shape of test_X: " + str(np.shape(test_X)))
print("Shape of big_y: " + str(np.shape(big_y)))
print("Shape of test_y: " + str(np.shape(test_y)))


clf = SVC(kernel="linear")
clf.fit(big_X, big_y)

pred_y = clf.predict(test_X)
print(accuracy_score(test_y,pred_y))


