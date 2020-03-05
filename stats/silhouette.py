#from keras.models import Model, load_model
#from keras.layers import Input, Reshape, TimeDistributed
from sklearn.svm import SVC
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering, AgglomerativeClustering
#import ffmpeg
import numpy as np
#from data import Data
import random
#from tabulate import tabulate
from matplotlib import pyplot as plt
#from generator import MyGenerator
import os
from sklearn.metrics import silhouette_score, davies_bouldin_score, v_measure_score

all_embeddings = []
file_names = ["2x2", "2x2_8"]

all_embeddings.append(np.load("preprocessed/embedding0.npy"))
all_embeddings.append(np.load("preprocessed/embedding_f8e1_0.npy"))

all_scores = []

for i in range(len(all_embeddings)):
    scores = []
    normalizer = 0
    for j in range(2,12):
        km = KMeans(n_clusters=j, random_state=0).fit(all_embeddings[i])
        preds = km.predict(all_embeddings[i])

        silhouette = silhouette_score(all_embeddings[i],preds)
        scores.append(silhouette)
        print("Silhouette score for {} with number of cluster(s) {}: {}".format(file_names[i], j,silhouette))
    all_scores.append(scores)
    


style_list = [
   "--bo", "--ro", "--go", "--yo", "--b^", "--r^", "--g^", "--y^", "--bv", "--rv", "--gv", "--yv"
]



for i in range(len(all_scores)):
    plt.plot([j for j in range(2,12)], all_scores[i], style_list[i])
    #plt.plot([i for i in range(2,12)], sc_scores, '--ro')
    #plt.plot([i for i in range(2,12)], aggc_scores, '--go')
    #plt.plot([i for i in range(2,12)], agga_scores, '--yo')
    #plt.plot([i for i in range(2,12)], aggw_scores, '--co')
plt.grid(True)
plt.xlabel("Number of clusters")
plt.ylabel("K-Means score")
plt.legend(file_names, loc="upper right")
plt.show()



