# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import numpy as np
import sklearn.cluster
from Levenshtein import distance
import argparse
import pdb
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--filename", required=True, help="Path to the image")
args = vars(ap.parse_args())

def word_clusters(filename):
	with open (filename, 'r') as f:
		words = f.readlines()
	words= [word.rstrip() for word in words if len(word)>4]
	words = np.asarray(words) #So that indexing with a list will work
	lev_similarity = -1*np.array([[distance(w1,w2) for w1 in words[:500]] for w2 in words[:500]])

	affprop = sklearn.cluster.AffinityPropagation(affinity="precomputed", damping=0.5)
	affprop.fit(lev_similarity)
	aff_matrix = affprop.affinity_matrix_
	X_tsne = TSNE(learning_rate=100).fit_transform(aff_matrix)
	labels= affprop.labels_
	centers = affprop.cluster_centers_indices_
	#
	plt.scatter(X_tsne[:, 0], X_tsne[:, 1],c=labels)
	center_tsne = X_tsne[centers]
	
	k=0
	
	for i,j in center_tsne:
		txt = words[centers[k]]
		k+=1
		plt.scatter(i,j,s=50,c='red',marker='+')

		# plt.annotate(txt,(i,j))
		# plt.annotate('$%s$'%txt,(i,j))

	# for i,j in X_tsne:
	# 	txt = words[labels[k]]
	# 	k+=1
	# 	plt.annotate(txt,(i,j))
	plt.show()
	with open ('temp.txt', 'w+', encoding="utf-8") as out_f:

		for cluster_id in np.unique(affprop.labels_):
			exemplar = words[affprop.cluster_centers_indices_[cluster_id]]
			cluster = np.unique(words[np.nonzero(affprop.labels_==cluster_id)])

			cluster_str = ", ".join(cluster)
			#exemplar = exemplar.encode('utf-8')
			#cluster_str = cluster_str.encode('utf-8')
			#print(" - *%s:* %s" % (exemplar, cluster_str))
			out_f.write(" - *%s:* %s \n" % (exemplar, cluster_str))



if __name__ == '__main__':
	word_clusters(args['filename'])