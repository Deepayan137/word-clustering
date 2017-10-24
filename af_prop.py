# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import numpy as np
import sklearn.cluster
from Levenshtein import distance
import argparse
import pdb
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from mst_clustering import MSTClustering
from sklearn.decomposition import TruncatedSVD
import seaborn as sns; sns.set()
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--filename", required=True, help="Path to the image")
args = vars(ap.parse_args())

def jaccard(w1, w2):
	w1 = set(w1)
	w2 = set(w2)
	return float(len(w1 & w2)) / len(w1 | w2)

def word_clusters(filename):
	with open (filename, 'r') as f:
		words = f.readlines()
	words= [word.rstrip() for word in words if len(word)>4]
	words = np.asarray(words) #So that indexing with a list will work
	lev_similarity = -1*np.array([[distance(w1,w2) for w1 in words[:500]] for w2 in words[:500]])
	#jac_similarity = np.array([[jaccard(w1,w2) for w1 in words[:500]] for w2 in words[:500]])
	affprop = sklearn.cluster.AffinityPropagation(affinity="precomputed", damping=0.5)
	affprop.fit(lev_similarity)
	aff_matrix = affprop.affinity_matrix_
	X_tsne = TSNE(learning_rate=100).fit_transform(aff_matrix) # takes the affinity matrix and projects it to 2 dimensional space
	labels= affprop.labels_
	centers = affprop.cluster_centers_indices_
	#
	plt.scatter(X_tsne[:, 0], X_tsne[:, 1],c=labels) # plots a scatter plot
	center_tsne = X_tsne[centers]
	
	k=0
	
	for i,j in center_tsne:
		txt = words[centers[k]]
		k+=1
		plt.scatter(i,j,s=50,c='red',marker='+') #plots the cluster center

	plt.show()
	with open ('temp_lev.txt', 'w+', encoding="utf-8") as out_f:

		for cluster_id in np.unique(affprop.labels_):
			exemplar = words[affprop.cluster_centers_indices_[cluster_id]]
			cluster = np.unique(words[np.nonzero(affprop.labels_==cluster_id)])
			cluster_str = ", ".join(cluster)
			out_f.write(" - *%s:* %s \n" % (exemplar, cluster_str))


def MST_clustering(filename):
	with open (filename, 'r') as f:
		words = f.readlines()
	words= [word.rstrip() for word in words if len(word)>4]
	words = np.asarray(words)
	jac_similarity = np.array([[jaccard(w1,w2) for w1 in words[:500]] for w2 in words[:500]])
	
	#pdb.set_trace()
	mst = MSTClustering(min_cluster_size=10, cutoff_scale=1) # cut-off scale ??
	mst.fit(jac_similarity)
	mst_matrix = mst.full_tree_

	X_tsne = TSNE(learning_rate=100).fit_transform(mst_matrix.todense())
	labels = mst.labels_
	pdb.set_trace()
	plt.scatter(X_tsne[:, 0], X_tsne[:, 1],c=labels)
	#plot_mst(mst)
	plt.show()

if __name__ == '__main__':
	#word_clusters(args['filename'])
	MST_clustering(args['filename'])