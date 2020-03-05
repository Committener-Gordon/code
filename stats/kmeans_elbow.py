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

big_X = []
big_y = []

all_embeddings = []
file_names = []

for r, d, f in os.walk("preprocessed"):
   for file in f:
        if "embedding" in file:
            embeddings = np.load(os.path.join(r, file))
            all_embeddings.append(embeddings)
            file_names.append(file)

print(np.shape(all_embeddings))

all_scores = []

for i in range(len(all_embeddings)):
    scores = []
    normalizer = 0
    for j in range(2,12):
        km = KMeans(n_clusters=j, random_state=0).fit(all_embeddings[i])
        preds = km.predict(all_embeddings[i])

        
        print("Score for {} with number of cluster(s) {}: {}".format(file_names[i],j,km.score(all_embeddings[i])))
        score = -km.score(all_embeddings[i])
        if j == 2:
            normalizer = score
        score = score / normalizer        
        scores.append(score)
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



