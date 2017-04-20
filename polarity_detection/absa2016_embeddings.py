#author Panagiotis Theodorakakos

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import xml.etree.ElementTree as ET
import numpy as np
from numpy import linalg as la
from sklearn import linear_model
from nltk import word_tokenize

class features():
    
	def __init__(self, train_path, test_path, model): 
		self.train_path = train_path
		self.test_path = test_path
		self.model = model
		self.dictionary = self.readIDF('idf_amazon.txt') #reads the idf of the keywords
	
		
	#reads the idf lexicon created from the amazon corpus
	def readIDF(self,path):
		f = open(path, "r", encoding='ascii')
		values = []
		idfs = []
		for word in f:
			word = word.split()
			values.append(word[0]) #store the word
			idfs.append(float(word[1])) #store the idf

		dictionary = dict(zip(values, idfs)) #store them in a dictionary for easy access 
		
		f.close()
		return dictionary
		
	#performs horizontal normalization of the embedding vector 
	def normalize(self,emb):
		
		n = la.norm(emb, axis=1) #finding the norm of each row of the embedding vector
		
		m = []
		for i in range(len(emb)):
			if n[i] == 0:
				m.append(np.zeros(200))
			else:
				m.append(emb[i]/n[i])
			
		return m
		
	#calculate the centroid of each sentence
	def calcCentroid(self,model,sentences):
		chars_to_remove = ['=', '!', '?', ',', '<', '.', '>', '/', ';' ,':', ']', '}', '[', '{', '|', '@', '$', '%', '^', '&', '*', '(', ')', '_', '-', '+', '+', '"','1', '2', '3','4','5','6','7','8','9' ] #removing characters
		centroids = []
		
		counter = 0
		for sentence in sentences: #for each sentence of the coprus

			embeddings = []
			idf = []
			
			for word0 in sentence: #for each word of the sentence
				word0 = word0.split('-') #handle cases such as good-job --> good, job
				for i in range(len(word0)):
					word = word0[i]
					word = ''.join([c for c in word if c not in chars_to_remove]) #handles cases like '1.Food' --> 'Food'
					if word not in chars_to_remove and word != '': #removing stopwords, symbols and blanks
						
						#---------- use the helpingEmbeddings.txt instead of the embeddings model ----------
						flag = False				
						for i in range(len(model)):
							if word == model[i][0]:
								flag = True
								index = i
								break
						if flag:
						#---------- use the helpingEmbeddings.txt instead of the embeddings model ----------
						
						#if word in model.vocab: #we have an embedding of the word #UNCOMMENT TO USE EMBEDDINGS MODEL
							#idf from amazon corpus
							if word in self.dictionary:
								if self.dictionary[word] < 0.5: #use only the usefull (based on IDF) words to create the centroid
									usefull = False
								else:
									idf.append( self.dictionary[word] ) #getting the idf of the word
									usefull = True
							else:
								usefull = False
							
							if usefull:
								#---------- use the helpingEmbeddings.txt instead of the embeddings model ----------
								embeddings.append(model[index][1])
								#---------- use the helpingEmbeddings.txt instead of the embeddings model ---------- 
							
								#tmp = model[word] #find the embeddings #UNCOMMENT TO USE EMBEDDINGS MODEL
								#embeddings.append(tmp) #store the 200-dimensional embedding from each word of the sentence #UNCOMMENT TO USE EMBEDDINGS MODEL
			counter = counter + 1
										
			l = len(embeddings) #number of words in the sentence
				
			if l == 0: #if no word of the sentence has an embedding
				centroids.append(np.zeros(200))
			else:
			
				embeddings = self.normalize(embeddings) #normalize the embedding vector before calculating the centroid of the sentence
				
				embeddings_new = []
				for i in range(l): #multiply each word embedding with each idf
					embeddings_tmp = []
					for j in range(200):
						embeddings_tmp.append(embeddings[i][j]*idf[i])
					embeddings_new.append(embeddings_tmp)
				
				centroid = np.sum(embeddings_new,axis=0) #sum each row
				
				centroid = centroid/l #caclulating the average
					
				centroids.append(centroid) #store each centroid
		
		return centroids
		
	def train(self):
        
		train_tags = []
		train_vector = []
		train_emb = []
        
		reviews = ET.parse(self.train_path).getroot().findall('Review')
		for review in reviews:
			sentences = review[0] #get the sentences
			for sentence in sentences:
				if (len(sentence) > 1):
					opinions = sentence[1] #getting the opinions field
					if ( len(opinions) > 0): #check if there are aspects 
						
						t = sentence[0].text #getting the text
						
						text = word_tokenize(t.lower()) #tokenize, convert to lower case
						
						for opinion in opinions: 
						
							category = opinion.attrib['polarity']    
							train_tags.append(category) #store the category

							train_emb.append(text) #store the tokenized words for the embedding's calculation
		
		centroid = self.calcCentroid(self.model,train_emb) #caclulate the centroid for each sentence
		
		for i in range(len(centroid)): #join the matrices
			tmp = centroid[i].tolist()
			train_vector.append(tmp)
		print('---- End of train ----')

		return train_vector,train_tags
    
	def test(self):
	
		test_vector = []
		test_emb = []
		cList = []
		
		reviews = ET.parse(self.test_path).getroot().findall('Review')
		for review in reviews:
			sentences = review[0] #get the sentences
			for sentence in sentences:
				if (len(sentence) > 1):
					opinions = sentence[1]
					
					if ( len(opinions) > 0): #check if there are aspects 
						t = sentence[0].text
						
						text = word_tokenize(t.lower())
						textC = word_tokenize(t) #tokenize, check for caps
						
						for opinion in opinions:
							
							test_emb.append(text) #store the tokenized words for the embedding's calculation
		
		centroid = self.calcCentroid(self.model,test_emb) #caclulate the centroid for each sentence	
		
		for i in range(len(centroid)): #join the matrices
			tmp = centroid[i].tolist()
			test_vector.append(tmp)

		print('---- End of Test ----')

		return test_vector
    
	def results(self, train_vector, train_tags, test_vector, dom):

		if dom == 'lap':
			logistic = linear_model.LogisticRegression(C=1.5) #fit logistic
			logistic.fit(train_vector,train_tags)
			resLogistic = logistic.predict_proba(test_vector)
			return resLogistic
			
		else:
			logistic = linear_model.LogisticRegression(C=0.01) #fit logistic
			logistic.fit(train_vector,train_tags)
			resLogistic = logistic.predict_proba(test_vector)
			return resLogistic
