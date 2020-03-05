from keras.models import Model, load_model
from keras.layers import Input, Reshape, TimeDistributed
from sklearn.svm import SVC
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering, AgglomerativeClustering
import ffmpeg
import numpy as np
from data import Data
import random
from tabulate import tabulate
from matplotlib import pyplot as plt
from generator import MyGenerator
import os
from sklearn.metrics import silhouette_score, davies_bouldin_score, v_measure_score

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

def tabulate2Clusters(labels, ground_truth):
    c_labels = [0, 0]
    r_labels = [0, 0]

    for i in range(len(labels)):
        if ground_truth[i] == 0:
            c_labels[labels[i]] += 1
        elif ground_truth[i] == 1: 
            r_labels[labels[i]] += 1
        else:
            print("Oopsie")
    print(tabulate([["Calving cows", c_labels[0], c_labels[1]], ["Random cows", r_labels[0], r_labels[1]]], headers=["Ground Truth/Cluster", "0", "1"]))



CHUNK_SIZE = 20
MODEL_NO = 0
DECODER_LENGTH = 10
EMBEDDING_SIZE = 64

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

#km = KMeans(n_clusters=2, random_state=0).fit(big_X)
#print(used_data)

#labels = km.labels_

#print("KMeans")
#tabulate2Clusters(labels, big_y)


km_scores= []
km_silhouette = []
km_vmeasure_score =[]
km_db_score = []

sc_scores= []
sc_silhouette = []
sc_vmeasure_score =[]
sc_db_score = []

aggc_scores= []
aggc_silhouette = []
aggc_vmeasure_score =[]
aggc_db_score = []

agga_scores= []
agga_silhouette = []
agga_vmeasure_score =[]
agga_db_score = []

aggw_scores= []
aggw_silhouette = []
aggw_vmeasure_score =[]
aggw_db_score = []
for i in range(2,12):
    km = KMeans(n_clusters=i, random_state=0).fit(big_X)
    preds = km.predict(big_X)

    
    print("Score for number of cluster(s) {}: {}".format(i,km.score(big_X)))
    km_scores.append(-km.score(big_X))
    
    silhouette = silhouette_score(big_X,preds)
    km_silhouette.append(silhouette)
    print("Silhouette score for number of cluster(s) {}: {}".format(i,silhouette))
    
    db = davies_bouldin_score(big_X,preds)
    km_db_score.append(db)
    print("Davies Bouldin score for number of cluster(s) {}: {}".format(i,db))
    
    v_measure = v_measure_score(big_y,preds)
    km_vmeasure_score.append(v_measure)
    print("V-measure score for number of cluster(s) {}: {}".format(i,v_measure))
    print("-"*100)


    sc = SpectralClustering(n_clusters=i, assign_labels="kmeans").fit(big_X)
    preds = sc.labels_

    #print("Score for number of cluster(s) {}: {}".format(i,sc.score(big_X)))
    #sc_scores.append(-sc.score(big_X))

    silhouette = silhouette_score(big_X,preds)
    sc_silhouette.append(silhouette)
    print("Silhouette score for number of cluster(s) {}: {}".format(i,silhouette))
    
    db = davies_bouldin_score(big_X,preds)
    sc_db_score.append(db)
    print("Davies Bouldin score for number of cluster(s) {}: {}".format(i,db))
    
    v_measure = v_measure_score(big_y,preds)
    sc_vmeasure_score.append(v_measure)
    print("V-measure score for number of cluster(s) {}: {}".format(i,v_measure))
    print("-"*100)


    aggc = AgglomerativeClustering(n_clusters=i, linkage="complete").fit(big_X)
    preds = aggc.labels_

    #print("Score for number of cluster(s) {}: {}".format(i,aggc.score(big_X)))
    #aggc_scores.append(-aggc.score(big_X))

    silhouette = silhouette_score(big_X,preds)
    aggc_silhouette.append(silhouette)
    print("Silhouette score for number of cluster(s) {}: {}".format(i,silhouette))
    
    db = davies_bouldin_score(big_X,preds)
    aggc_db_score.append(db)
    print("Davies Bouldin score for number of cluster(s) {}: {}".format(i,db))
    
    v_measure = v_measure_score(big_y,preds)
    aggc_vmeasure_score.append(v_measure)
    print("V-measure score for number of cluster(s) {}: {}".format(i,v_measure))
    print("-"*100)


    agga = AgglomerativeClustering(n_clusters=i, linkage="average").fit(big_X)
    preds = agga.labels_

    #print("Score for number of cluster(s) {}: {}".format(i,agga.score(big_X)))
    #agga_scores.append(-agga.score(big_X))

    silhouette = silhouette_score(big_X,preds)
    agga_silhouette.append(silhouette)
    print("Silhouette score for number of cluster(s) {}: {}".format(i,silhouette))
    
    db = davies_bouldin_score(big_X,preds)
    agga_db_score.append(db)
    print("Davies Bouldin score for number of cluster(s) {}: {}".format(i,db))
    
    v_measure = v_measure_score(big_y,preds)
    agga_vmeasure_score.append(v_measure)
    print("V-measure score for number of cluster(s) {}: {}".format(i,v_measure))
    print("-"*100)


    aggw = AgglomerativeClustering(n_clusters=i, linkage="ward").fit(big_X)
    preds = aggw.labels_

    #print("Score for number of cluster(s) {}: {}".format(i,aggw.score(big_X)))
    #aggw_scores.append(-aggw.score(big_X))

    silhouette = silhouette_score(big_X,preds)
    aggw_silhouette.append(silhouette)
    print("Silhouette score for number of cluster(s) {}: {}".format(i,silhouette))
    
    db = davies_bouldin_score(big_X,preds)
    aggw_db_score.append(db)
    print("Davies Bouldin score for number of cluster(s) {}: {}".format(i,db))
    
    v_measure = v_measure_score(big_y,preds)
    aggw_vmeasure_score.append(v_measure)
    print("V-measure score for number of cluster(s) {}: {}".format(i,v_measure))
    print("-"*100)
    

xs = []
plotlabels = np.zeros((5,10))
for i in range(5):
    xs.append([])   
    for j in range(2,12):
        xs[i].append(j)
        plotlabels[i][j-2] = i



plt.plot([i for i in range(2,12)], km_scores, '--bo')
#plt.plot([i for i in range(2,12)], sc_scores, '--ro')
#plt.plot([i for i in range(2,12)], aggc_scores, '--go')
#plt.plot([i for i in range(2,12)], agga_scores, '--yo')
#plt.plot([i for i in range(2,12)], aggw_scores, '--co')
plt.grid(True)
plt.xlabel("Number of clusters")
plt.ylabel("K-Means score")
plt.show()

plt.plot([i for i in range(2,12)], km_silhouette, '--bo')
plt.plot([i for i in range(2,12)], sc_silhouette, '--ro')
plt.plot([i for i in range(2,12)], aggc_silhouette, '--go')
plt.plot([i for i in range(2,12)], agga_silhouette, '--yo')
plt.plot([i for i in range(2,12)], aggw_silhouette, '--co')
plt.grid(True)
plt.xlabel("Number of clusters")
plt.ylabel("Silhouette score")
plt.show()

plt.plot([i for i in range(2,12)], km_vmeasure_score, '--bo')
plt.plot([i for i in range(2,12)], sc_vmeasure_score, '--ro')
plt.plot([i for i in range(2,12)], aggc_vmeasure_score, '--go')
plt.plot([i for i in range(2,12)], agga_vmeasure_score, '--yo')
plt.plot([i for i in range(2,12)], aggw_vmeasure_score, '--co')
plt.grid(True)
plt.xlabel("Number of clusters")
plt.ylabel("V-Measure score")
plt.show()

plt.plot([i for i in range(2,12)], km_db_score, '--bo')
plt.plot([i for i in range(2,12)], sc_db_score, '--ro')
plt.plot([i for i in range(2,12)], aggc_db_score, '--go')
plt.plot([i for i in range(2,12)], agga_db_score, '--yo')
plt.plot([i for i in range(2,12)], aggw_db_score, '--co')
plt.grid(True)
plt.xlabel("Number of clusters")
plt.ylabel("Davies-Bouldin score")
plt.show()




spectral = SpectralClustering(n_clusters=2, assign_labels="kmeans").fit(big_X)
print("\n\nSpectral Clustering")
tabulate2Clusters(spectral.labels_, big_y)


agg_avg = AgglomerativeClustering(n_clusters=2, linkage="average").fit(big_X)
print("\n\nAgglomerative Clustering AVG")
tabulate2Clusters(agg_avg.labels_, big_y)

agg_complete = AgglomerativeClustering(n_clusters=2, linkage="complete").fit(big_X)
print("\n\nAgglomerative Clustering COMPLETE")
tabulate2Clusters(agg_complete.labels_, big_y)

agg_ward = AgglomerativeClustering(n_clusters=2, linkage="ward").fit(big_X)
print("\n\nAgglomerative Clustering WARD")
tabulate2Clusters(agg_ward.labels_, big_y)

#tabulateLabels(used_data, km.labels_, spectral.labels_, agg_avg.labels_, agg_complete.labels_, agg_ward.labels_)

