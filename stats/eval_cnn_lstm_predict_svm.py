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
import os

CHUNK_SIZE = 20
MODEL_NO = 5
DECODER_LENGTH = 10
EMBEDDING_SIZE = 625

FILE_APPENDIX = ""

def load_model_encoder(inputs, number=0, cnn_decoder_length=10):
    cnn_model = load_model("models/model_cnn_run" + str(number) + ".h5")
    cnn_encoder = Model(cnn_model.inputs, cnn_model.layers[-cnn_decoder_length].output)
    
    time_cnn = TimeDistributed(cnn_encoder)(inputs)

    reshape =  Reshape((19, EMBEDDING_SIZE))(time_cnn)

    lstm_model = load_model("models/model_lstm_predict_run" + str(number) + ".h5")
    lstm_encoder = Model(lstm_model.inputs, lstm_model.layers[-3].output)

    embedding = lstm_encoder(reshape)
    return Model(inputs, embedding)
    

big_X = []
big_y = []

# if there exists preprocessed data for the specified model, we don't have to process it again
if (os.path.isfile("preprocessed/embedding" + FILE_APPENDIX + str(MODEL_NO) + ".npy") 
    and os.path.isfile("preprocessed/labels" + FILE_APPENDIX + str(MODEL_NO) + ".npy")):

    big_X = np.load("preprocessed/embedding" + FILE_APPENDIX + str(MODEL_NO) + ".npy")
    big_y = np.load("preprocessed/labels" + FILE_APPENDIX + str(MODEL_NO) + ".npy")
    print("Loaded preprocessed file.")
else:
    # preprocess the data by running it through our models
    input_dims = (19, 400, 400, 1)
    inputs = Input(shape=input_dims)

    my_model = load_model_encoder(inputs, MODEL_NO, DECODER_LENGTH)
    my_model.summary()

    data_set = np.load("prepared_ids.npy")

    chunks = int(len(data_set) / CHUNK_SIZE)
    print("Datensatz: " + str(len(data_set)))
    print("Chunks: " + str(chunks))

    data_set = data_set[: chunks * CHUNK_SIZE]

    print("Getrimmter Datensatz: " + str(len(data_set)))

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

    np.save("preprocessed/embedding" + FILE_APPENDIX + str(MODEL_NO), big_X)
    np.save("preprocessed/labels" + FILE_APPENDIX + str(MODEL_NO), big_y)

    print(np.shape(big_y))
    print("Amount zeros: " + str(big_y.count(0)))
    print("Amount ones: " + str(big_y.count(1)))

testsize = int(len(big_X) / 10)

accuracy_scores = np.zeros(10)

for i in range(10):
    X_train = [*big_X[: i * testsize], *big_X[(i+1) * testsize :]]
    X_test = big_X[i * testsize : (i+1) * testsize]

    y_train = [*big_y[: i * testsize], *big_y[(i+1) * testsize :]]
    y_test = big_y[i * testsize : (i+1) * testsize]

    clf = SVC(kernel="rbf")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy_scores[i] = accuracy_score(y_test, y_pred)

print("Accuracy Scores: " + str(accuracy_scores))
print("Average: " + str(np.average(accuracy_scores)))

