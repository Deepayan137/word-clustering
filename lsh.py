from __future__ import division
import os
import re
import random
import time
import binascii
import pdb
import numpy as np
import sys
import pandas as pd
def jaccard(w1, w2):
	w1 = set(w1)
	w2 = set(w2)
	return float(len(w1 & w2)) / len(w1 | w2)

def pickRandomCoeffs(k):
	randList = []
	maxShingleID = 2**32-1
	while k > 0:
		randIndex = random.randint(0, maxShingleID) 
		while randIndex in randList:
			randIndex = random.randint(0, maxShingleID) 

		randList.append(randIndex)
		k = k - 1
	return randList



def get_shingles(words):
	totalShingles = 0
	wordNames = []
	wordAsShingleSets= {}

	numWords = len(words)
	for i,word in enumerate(words):
		chars = [ch for ch in word]
		wordID =  word
		wordNames.append(wordID)
		shinglesInWord = set()
		for i in range(0, len(chars) - 2):
			shingle = chars[i]+" "+chars[i+1]+" "+chars[i+2]
			crc = binascii.crc32(shingle.encode()) & 0xffffffff
			shinglesInWord.add(crc)
		wordAsShingleSets[wordID] = shinglesInWord
	return wordNames, wordAsShingleSets

def get_signatures(words):
	wordNames, wordAsShingleSets = get_shingles(words) 
	maxShingleID = 2**32-1
	nextPrime = 4294967311
	numHashes = 10
	coeffA = pickRandomCoeffs(numHashes)
	coeffB = pickRandomCoeffs(numHashes)
	signatures = []

	for wordID in wordNames:
		shingleIDSet = wordAsShingleSets[wordID]
		signature = []
		h=[]
		for i in range(0, numHashes):
			minHashCode = nextPrime + 1	
			for shingleID in shingleIDSet:
				hashCode = (coeffA[i] * shingleID + coeffB[i]) % nextPrime 
				h.append(hashCode)

				if hashCode < minHashCode:
					minHashCode = hashCode
			signature.append(minHashCode)
		signatures.append(signature)
	print ('\nComparing all signatures...') 
	return wordNames, signatures

def get_similarity(words):
	numWords = len(words)
	wordNames, signatures = get_signatures(words)
	numHashes =len(signatures[0])
	#pdb.set_trace()
	estJSim=np.zeros((numWords, numWords))
	threshold = 0.5
	for i in range(0, numWords):
		signature1 = signatures[i]
		for j in range(i + 1, numWords):
			signature2 = signatures[j]
			count = 0
			for k in range(0, numHashes):
				count = count + (signature1[k] == signature2[k])
				estJSim[i, j] = (count / numHashes)
			
	
	r=[]
	for i in range(0, numWords): 
		row = []
		for j in range(i + 1, numWords):
	  		estJ = estJSim[i, j]
	  		if estJ >0.5:
	  			row.append((wordNames[j]))
		r.append(row)
		if row:
			with open('test.txt', 'a') as in_file:
				in_file.write((wordNames[i])+'====>'+",".join(row)+'\n')
	#pdb.set_trace()
if __name__ == '__main__':
	data_file = sys.argv[1]
	with open(data_file, 'r') as in_file:
		words = in_file.readlines()
	words= [word.rstrip() for word in words[:500]]
	get_similarity(words)

	