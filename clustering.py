from panda_useFunction import DataMinipulation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GMM
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.decomposition import PCA
import itertools
import warnings
#Disable warning
warnings.filterwarnings("ignore")
#Color label
color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold','darkorange'])
def column(matrix, i):
    return [row[i] for row in matrix]
#Clean up data before clustering
data = DataMinipulation()
data.read_csv("../Data/customer_usage_hidden.csv")
data.covert_cart_asCode(["Voice_Type_Usage","SMS_Type_Usage","Data_Type_Usage","Time_Slot"])
data.del_col("ID")
datainlist = data.get_data().values.tolist()
#print (datainlist)
#Clustering method using Guassian Mixture Model
kmeans = KMeans(n_clusters=3,init="k-means++",n_init=35,tol=1e-5)
labels = kmeans.fit(datainlist).labels_
#print(labels)
#gmm = GMM(n_components=3,tol=1e-5).fit(datainlist)
#labels = gmm.predict(datainlist)
#dbscan = DBSCAN(eps=1,min_samples=60,metric='euclidean')
#dbscan.fit(datainlist)
#labels = dbscan.fit_predict(datainlist)
#PCA for ploting
#pca = PCA(n_components=3)
#pca.fit(datainlist)
#datainlist = pca.transform(datainlist)
#print(datainlist)
#plt.scatter(datainlist[:,0], datainlist[:,1], c=labels, cmap='viridis');
#plt.show()
#fig = plt.figure()
#ax = fig.gca(projection='3d')
#ax.scatter(datainlist[:,0][0:3000], datainlist[:,1][0:3000],datainlist[:,2][0:3000], c=labels[0:3000])
#plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(column(datainlist,0)[0:3000]
	,column(datainlist,4)[0:3000]
	,column(datainlist,6)[0:3000]
	,c=labels[0:3000])
plt.show()
