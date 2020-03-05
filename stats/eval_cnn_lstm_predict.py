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

def tabulateLabels(files, kmeans, spectral, complete, average, ward):
    table = []
    headers = ["File", "K-Means", "Spectral", "Agg-Avg", "Agg-Complete", "Agg-Ward"]
    for key, file in enumerate(files):
        line = []
        line.append(file)
        line.append(kmeans[key])
        line.append(spectral[key])
        line.append(average[key])
        line.append(complete[key])
        line.append(ward[key])
        table.append(line)
    print(tabulate(table, headers=headers))

def tabulate2Clusters(labels, separator):
    c_labels = [0, 0]
    r_labels = [0, 0]

    for i in range(len(labels)):
	    if i < separator:
		    c_labels[labels[i]] += 1
	    else: 
		    r_labels[labels[i]] += 1
    print(tabulate([["Calving cows", c_labels[0], c_labels[1]], ["Random cows", r_labels[0], r_labels[1]]], headers=["Ground Truth/Cluster", "0", "1"]))

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

data_set = Data.get_data_paths("./calving_samples", "./random_samples", percentage_a=0.02, percentage_b=0.16)
used_data = []

X = []
calved_count = 0
random_count = 0
ground_truth = []

for file in data_set:
    if random.random() < 0.5:
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
        used_data.append(file)
        if "calving" in file:
            calved_count += 1
            ground_truth.append(0)
        else:
            random_count += 1
            ground_truth.append(1)
        print(file)


print(np.shape(X))
X = np.array(X)

y = my_model.predict(X)
#y = y[:,19]
y = y.reshape(-1, 16)

print(np.shape(y))

kmeans = KMeans(n_clusters=2, random_state=0).fit(y)
#print(used_data)

print("Calving count: " + str(calved_count))
print("Random count: " + str(random_count))

labels = kmeans.labels_

print("KMeans")
tabulate2Clusters(labels, calved_count)

spectral = SpectralClustering(n_clusters=2, assign_labels="kmeans").fit(y)
print("\n\nSpectral Clustering")
tabulate2Clusters(spectral.labels_, calved_count)


agg_avg = AgglomerativeClustering(n_clusters=2, linkage="average").fit(y)
print("\n\nAgglomerative Clustering AVG")
tabulate2Clusters(agg_avg.labels_, calved_count)

agg_complete = AgglomerativeClustering(n_clusters=2, linkage="complete").fit(y)
print("\n\nAgglomerative Clustering COMPLETE")
tabulate2Clusters(agg_complete.labels_, calved_count)

agg_ward = AgglomerativeClustering(n_clusters=2, linkage="ward").fit(y)
print("\n\nAgglomerative Clustering WARD")
tabulate2Clusters(agg_ward.labels_, calved_count)

tabulateLabels(used_data, kmeans.labels_, spectral.labels_, agg_avg.labels_, agg_complete.labels_, agg_ward.labels_)

