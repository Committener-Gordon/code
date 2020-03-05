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

def tabulateLabels(files, kmeans, spectral, average, complete, ward):
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
    cnn_model = load_model("models/model_cnn_run" + str(number) + ".h5")
    cnn_encoder = Model(cnn_model.inputs, cnn_model.layers[-cnn_decoder_length].output)
    
    time_cnn = TimeDistributed(cnn_encoder)(inputs)

    reshape =  Reshape((19, 625))(time_cnn)

    lstm_model = load_model("models/model_lstm_predict_run" + str(number) + ".h5")
    lstm_encoder = Model(lstm_model.inputs, lstm_model.layers[-3].output)

    embedding = lstm_encoder(reshape)
    return Model(inputs, embedding)
    


input_dims = (19, 400, 400, 1)
inputs = Input(shape=input_dims)

my_model = load_model_encoder(inputs, 5, 10)
my_model.summary()

data_set = [
    "./calving_samples/135/0257.mp4",
    "./calving_samples/486/0246.mp4",
    "./calving_samples/780/0238.mp4",
    "./calving_samples/780/0270.mp4",
    "./calving_samples/780/0210.mp4",
    "./calving_samples/545/0305.mp4",
    "./calving_samples/244/0117.mp4",
    "./calving_samples/244/0120.mp4",
    "./calving_samples/376/0275.mp4",
    "./calving_samples/22/0282.mp4",
    "./calving_samples/22/0291.mp4",
    "./calving_samples/154/0204.mp4",
    "./calving_samples/128/0180.mp4",
    "./calving_samples/128/0192.mp4",
    "./calving_samples/128/0100.mp4",
    "./random_samples/144/900/M00001/c0imgcut.mp4",
    "./random_samples/144/000/M00001/c2imgcut.mp4",
    "./random_samples/116/900/M00001/c2imgcut.mp4",
    "./random_samples/116/800/M00001/c1imgcut.mp4",
    "./random_samples/143/200/M00001/c1imgcut.mp4",
    "./random_samples/143/999/M00004/c2imgcut.mp4",
    "./random_samples/115/000/M00001/c0imgcut.mp4",
    "./random_samples/120/200/M00001/c0imgcut.mp4",
    "./random_samples/120/800/M00001/c0imgcut.mp4",
    "./random_samples/147/400/M00004/c3imgcut.mp4",
    "./random_samples/147/400/M00004/c0imgcut.mp4",
    "./random_samples/147/100/M00002/c1imgcut.mp4",
    "./random_samples/147/000/M00005/c1imgcut.mp4",
    "./random_samples/147/000/M00003/c0imgcut.mp4",
    "./random_samples/118/801/M00004/c0imgcut.mp4",
    "./random_samples/118/801/M00005/c0imgcut.mp4",
    "./random_samples/122/300/M00001/c2imgcut.mp4",
    "./random_samples/122/300/M00001/c0imgcut.mp4",
    "./random_samples/122/300/M00002/c2imgcut.mp4",
    "./random_samples/122/000/M00004/c0imgcut.mp4",
    "./random_samples/122/000/M00001/c2img.mp4"
]




used_data = []

X = []
calved_count = 0
random_count = 0
ground_truth = []

for file in data_set:
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

