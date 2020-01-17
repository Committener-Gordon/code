from numpy import array
from keras.models import Sequential
from keras.models import Model, load_model
from keras.layers import Input
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering, AgglomerativeClustering
import ffmpeg
import numpy as np
from data import Data
import random
from tabulate import tabulate


def tabulate2Clusters(labels, separator):
    c_labels = [0, 0]
    r_labels = [0, 0]

    for i in range(len(labels)):
	    if i < separator:
		    c_labels[labels[i]] += 1
	    else: 
		    r_labels[labels[i]] += 1
    print(tabulate([["Calving cows", c_labels[0], c_labels[1]], ["Random cows", r_labels[0], r_labels[1]]], headers=["Ground Truth/Cluster", "0", "1"]))



input_dims = (20, 400, 400, 1)
inputs = Input(shape=input_dims)


loaded_model = load_model("model.h5")
#loaded_model.summary()

#loaded_model.layers.pop()
#loaded_model.layers.pop()
#loaded_model.layers.pop()
#loaded_model.layers.pop()
#loaded_model.layers.pop()
#loaded_model.layers.pop()


new_model = Model(loaded_model.inputs, loaded_model.layers[-7].output)
#new_model.summary()

data_set = Data.get_data_paths("./calving_samples", "./random_samples", percentage_b=0.82)
used_data = []

X = []
calved_count = 0
random_count = 0

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
		X.append(video/255)	
		used_data.append(file)
		if "calving" in file:
			calved_count += 1
		else:
			random_count += 1
		print(file)

#filename = "./vid/0301.mp4"

print(np.shape(X))
X = np.array(X)

y = new_model.predict(X)
y = y[:,19]
y = y.reshape(-1, 100)

print(np.shape(y))
kmeans = KMeans(n_clusters=2, random_state=0).fit(y)
#print(used_data)

print("Calving count: " + str(calved_count))
print("Random count: " + str(random_count))

labels = kmeans.labels_
c_labels = [0, 0]
r_labels = [0, 0]

for i in range(len(labels)):
	if i < calved_count:
		c_labels[labels[i]] += 1
	else: 
		r_labels[labels[i]] += 1

print("KMeans")
tabulate2Clusters(labels, calved_count)


#for i in range(1, 10):
    #for j in range(1, 10):
        #dbscan = DBSCAN(eps=i, min_samples=j, metric="euclidean").fit(y)
        #labels = dbscan.labels_
        #print("\n\nDBSCAN: eps: " + str(i) + ", min_samples: " + str(j))
        #print("Labels: " + str(labels))



spectral = SpectralClustering(n_clusters=2, assign_labels="kmeans").fit(y)
print("\n\nSpectral Clustering")
tabulate2Clusters(spectral.labels_, calved_count)


agg = AgglomerativeClustering(n_clusters=2, linkage="average").fit(y)
print("\n\nAgglomerative Clustering AVG")
tabulate2Clusters(agg.labels_, calved_count)

agg = AgglomerativeClustering(n_clusters=2, linkage="complete").fit(y)
print("\n\nAgglomerative Clustering COMPLETE")
tabulate2Clusters(agg.labels_, calved_count)

agg = AgglomerativeClustering(n_clusters=2, linkage="ward").fit(y)
print("\n\nAgglomerative Clustering WARD")
tabulate2Clusters(agg.labels_, calved_count)



