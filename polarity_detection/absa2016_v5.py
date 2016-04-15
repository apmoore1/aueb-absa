#author Panagiotis Theodorakakos

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import xml.etree.ElementTree as ET
from sklearn import linear_model
from nltk import word_tokenize
from postaggers import arktagger 
import numpy as np

class features():

	def __init__(self, train_path, test_path, dom): 
		self.train_path = train_path
		self.test_path = test_path

		#read the lexicon
		self.huLiu_lexicon = self.readLex('HuLiu_lexicon.txt')
		self.AFINN_lexicon = self.readLex('AFINN-111.txt')
		self.nrc_lexicon = self.readLexNRC('nrc_lexicon.txt')

		self.stopwords = self.readStopW('sentiment_stopwords.txt')
		self.negation = self.readNeg('negation.txt')

		#unique lexicon for each domain, created from the train data
		if dom == 'rest':
			self.train_unigram_lexicon = self.readLex('unigram_score_rest.txt')
			self.train_posBigram_lexicon = self.readLexBigram('posBigrams_score_rest.txt')
			
			self.entities = ['RESTAURANT', 'FOOD', 'DRINKS', 'AMBIENCE', 'SERVICE', 'LOCATION']
			self.attributes = ['GENERAL', 'PRICES', 'QUALITY', 'STYLE_OPTIONS', 'MISCELLANEOUS']
		elif dom == 'lap':       
			self.train_unigram_lexicon = self.readLex('unigram_score_lap.txt')
			self.train_posBigram_lexicon = self.readLexBigram('posBigrams_score_lap.txt')

			self.entities = [ 'LAPTOP', 'DISPLAY', 'KEYBOARD', 'MOUSE', 'MOTHERBOARD', 'CPU', 'FANS_COOLING', 'PORTS', 'MEMORY', 'POWER_SUPPLY', 'OPTICAL_DRIVES', 'BATTERY',
							'GRAPHICS', 'HARD_DISK', 'MULTIMEDIA_DEVICES', 'HARDWARE', 'SOFTWARE', 'OS', 'WARRANTY', 'SHIPPING', 'SUPPORT', 'COMPANY' ]
			self.attributes = [ 'GENERAL', 'PRICE', 'QUALITY', 'OPERATION_PERFORMANCE', 'USABILITY', 'DESIGN_FEATURES', 'PORTABILITY', 'CONNECTIVITY', 'MISCELLANEOUS' ]
			
		
	#reads the stopwords
	def readStopW(self,path):
		f = open(path, "r")
		stopwords=[]
		for word in f:
			tmp = word.replace('\n','').lower()
			tmp = tmp.replace('\r','')
			stopwords.append(tmp)

		f.close()
		return stopwords

	#reads the negation lexicon
	def readNeg(self,path):
		f = open(path, "r")
		negation=[]
		for word in f:
			tmp = word.replace('\n','').lower()
			tmp = tmp.replace('\r','')
			negation.append(tmp)

		f.close()
		return negation

	#reads the nrc lexicon
	def readLexNRC(self,path):
		f = open(path, "r")
		lexicon=[]
		counter,s = 0,0
		positive = ['joy','surprise','anticipation','trust', 'positive']
		for word in f:
			if (counter == 10):
				counter = 0
				s = 0
				
			word = word_tokenize(word)
			unigram = word[0]
			if word[1] in positive:
				sc = int(word[2])
			else:
				sc = -int(word[2])
				
			s = s + sc
			if counter == 9:
				if s != 0:
					tmp = [unigram, [s]]
					lexicon.append(tmp)
				
			counter = counter + 1
		f.close()
		return lexicon

	#reads the lexicon, AFINN, Liu
	def readLex(self,path):
		f = open(path, "r")
		lexicon=[]
		for word in f:
			score=[]
			word = word_tokenize(word)
			for i in range(len(word)):
				if i == 0:
					unigram = str(word[0].lower())
				else:
					if len(word) > 5:
						score.append(float(word[i]))
					else:
						score.append(int(word[i]))
			
			tmp = [unigram, score]
			lexicon.append(tmp)
		f.close()
		return lexicon

	#reads the lexicon of bigrams
	def readLexBigram(self,path):
		f = open(path, "r")
		lexicon=[]
		for word in f:
			score=[]
			unigram=[]
			word = word_tokenize(word)
			for i in range(len(word)):
				if i == 0 or i == 1:
					unigram.append(str(word[i].lower()))
				else:
					score.append(float(word[i]))
			
			tmp = [unigram, score]
			lexicon.append(tmp)
		f.close()
		return lexicon

	#returns the normalized vector [0,1]
	def normalize(self,vector):
		ma = np.amax(vector, axis=0)
		mi = np.amin(vector, axis=0)
		for i in range(len(vector)):
			for j in range(len(vector[i])):
				if (ma[j] - mi[j]) != 0:
					vector[i][j] = float((vector[i][j] - mi[j]))/(ma[j] - mi[j])
				#if the numbers exceed the limits, pin them to the limits
				if vector[i][j] > 1:
					vector[i][j] = 1
				elif vector[i][j] < 0:
					vector[i][j] = 0        
		return vector


	#returns a list with the capitalized words, returns the number of capitalized letters
	def listCaps(self,text):
		ctr = [ ',', '<', '.', '>', '/', ';' ,':', ']', '}', '[', '{', '|', '@', '$', '%', '^', '&', '*', '(', ')', '_', '-', '+', '+', '"','1', '2', '3','4','5','6','7','8','9' ] #removing characters
		l = []
		counter = 0
		for word in text:
			if (word == word.upper()) and (word not in ctr):
				counter = counter + 1
				l.append(word.lower())
		return l, counter

	#finding the last word of the text
	def lastWord(self,text):
		chars_to_remove = [ ',', '<', '.', '>', '/', ';' ,':', ']', '}', '[', '{', '|', '@', '$', '%', '^', '&', '*', '(', ')', '_', '-', '+', '+', '"','1', '2', '3','4','5','6','7','8','9' ] #removing characters
		for i in range(len(text)):
			if (text[-(i+1)] not in chars_to_remove):
				return (text[-(i+1)])
			
	#finding the number of appearances of a symbol
	def howMany(self,sentence,symbol):
		c = 0
		for word in sentence:
			if (word == symbol):
				c = c + 1       
		return c

	#returns the number of words with their first letter capilatized
	def howManyUpperFirst(self,text):
		chars_to_remove = ['=', '!', '?', ',', '<', '.', '>', '/', ';' ,':', ']', '}', '[', '{', '|', '@', '$', '%', '^', '&', '*', '(', ')', '_', '-', '+', '+', '"','1', '2', '3','4','5','6','7','8','9' ] #removing characters
		counter = 0
		for word in text:
			for letter in word:
				if letter == letter.upper() and (letter not in chars_to_remove):
					counter = counter + 1
				break
		return [counter]

	#returns the existances or not of a question mark or exclamation mark as the message's end
	def lastSymbol(self,text):
		excl,ques = 0,0
		for i in range(len(text)):
			if i+1 == len(text):
				for j in range(len(text[i])):
					if j+1 == len(text[i]):
						if text[i][j] == '!':
							excl = 1
						elif text[i][j] == '?':
							ques = 1
							
		if excl == 1 or ques == 1:
			so = 1
		else:
			so = 0
		return [excl, ques, so]

	#returns a list with the number of adjectives, adverbs, nouns and verbs for each text
	def howManyPos(self, pos):
		p = []
		for text in pos: #checking the pos tags
			adjectives, adverbs, verbs, nouns = 0,0,0,0
			
			for i in range(len(text)): 
				if (text[i] == 'A'):
					adjectives = adjectives + 1
				elif(text[i] == 'R'):
					adverbs = adverbs + 1
				elif(text[i] == 'N'):
					nouns = nouns + 1
				elif(text[i] =='V'):
					verbs = verbs + 1 

			tmp = [adjectives, adverbs, nouns, verbs]
			p.append(tmp)
		return p

	#calculate the scores of lexicons with scores
	def checkLexiconReady(self,text,stopwords,capsList,negation,lexicon):
		chars_to_remove = ['=', '!', '?', ',', '<', '.', '>', '/', ';' ,':', ']', '}', '[', '{', '|', '@', '$', '%', '^', '&', '*', '(', ')', '_', '-', '+', '+', '"','1', '2', '3','4','5','6','7','8','9' ] #removing characters
		score_sum, maximum, minimum, positives, negatives, score_avg, score_last, positive_sum, negative_sum = 0,0,0,0,0,0,0,0,0
		word_last = self.lastWord(text) #getting the last word

		prev_word = text[0]
		for word in text:
			word = ''.join([c for c in word if c not in chars_to_remove]) #handles cases like '1.Food' --> 'Food'
			if (word not in stopwords) and (word not in chars_to_remove): #not checking stopwords and symbols
				for i in range(len(lexicon)):
					if word == lexicon[i][0]: #if it exists in the lexicon
						word_score = int(lexicon[i][1][0]) #store the score
						if (word in capsList): #if it is capitalized
							word_score = word_score*3 #we triple the score of the word                            
						if (prev_word in negation): #if the previous word is not, don't etc (negation words) 
							word_score = - word_score #we change the sign

						if word_score > maximum: #finding max value
							maximum = word_score
						if word_score < minimum: #finding min value
							minimum = word_score
						if word_score > 0: #finding sum of positive keywords
							positives = positives + 1
							positive_sum = positive_sum + word_score
						if word_score < 0: #finding sum of negative keywords
							negatives = negatives + 1
							negative_sum = negative_sum + word_score
						if word_last == word: #finding the score of the last word
							score_last = word_score 
							
						score_sum = score_sum + word_score #finding sum
						
				prev_word = word 
					  
		if (positives + negatives != 0): #calculating the avg scores 
			score_avg = score_sum/(positives + negatives) #score/#of words of the text in lexicon 
		else:
			score_avg = 0
			
		if (negatives != 0):
			d = float(positives)/negatives
		else:
			d = 1
			
		return [d, score_sum, maximum, minimum, positives, negatives, score_avg, score_last]

	#returns the scores from the train data unigram's lexicons
	def checkLexiconUni(self,text,chars_to_remove,stopwords,capsList,negation,lexicon):
		
		pre_pos,pre_neg,pre_neu,f1_pos,f1_neg,f1_neu,avg_pre_pos,avg_pre_neg,avg_pre_neu,avg_f1_pos,avg_f1_neg,avg_f1_neu = 0,0,0,0,0,0,0,0,0,0,0,0
		pre_max_pos, pre_max_neg, pre_max_neu, f1_max_pos, f1_max_neg, f1_max_neu = -1,-1,-1,-1,-1,-1
		pre_min_pos, pre_min_neg, pre_min_neu, f1_min_pos, f1_min_neg, f1_min_neu  = 1000000,1000000,1000000,1000000,1000000,1000000
		c = 0

		prev_word = text[0] #getting the previous word
		for word in text:
			word = ''.join([k for k in word if k not in chars_to_remove]) #handles cases like '1.Food' --> 'Food'
			if (word not in stopwords) and (word not in chars_to_remove): #not checking stopwords and symbols

				for i in range(len(lexicon)):
					if word == lexicon[i][0]: #if it exists in the lexicon
												
						if prev_word in negation: #if the previous word is a negation word, we change the positive with the negative precision-F1
							p_pos = lexicon[i][1][1]
							p_neg = lexicon[i][1][0]
							p_neu = lexicon[i][1][2] #neutral stays the same
							f_pos = lexicon[i][1][4]
							f_neg = lexicon[i][1][3]
							f_neu = lexicon[i][1][5] #neutral stays the same
						else:
							p_pos = lexicon[i][1][0]
							p_neg = lexicon[i][1][1]
							p_neu = lexicon[i][1][2]
							f_pos = lexicon[i][1][3]
							f_neg = lexicon[i][1][4]
							f_neu = lexicon[i][1][5]
						
						if word in capsList: #if the word is capitalized, triple the each score
							p_pos = p_pos*3
							p_neg = p_neg*3
							p_neu = p_neu*3
							f_pos = f_pos*3
							f_neg = f_neg*3
							f_neu = f_neu*3
							
						pre_pos = pre_pos + p_pos #sum of precision for positive,negative,neutral and f1 positive,negative,neutral
						pre_neg = pre_neg + p_neg
						pre_neu = pre_neu + p_neu
						f1_pos = f1_pos + f_pos
						f1_neg = f1_neg + f_neg
						f1_neu = f1_neu + f_neu
						c = c + 1
						
						if (pre_max_pos < p_pos): #calculating the maximum precision for the positive
							pre_max_pos = p_pos
						if (pre_min_pos > p_pos): #calculating the minimum precision for the positive
							pre_min_pos = p_pos
							
						if (pre_max_neg < p_neg): #calculating the maximum precision for the negative
							pre_max_neg = p_neg
						if (pre_min_neg > p_neg): #calculating the minimum precision for the negative
							pre_min_neg = p_neg
							
						if (pre_max_neu < p_neu): #calculating the maximum precision for the neutral
							pre_max_neu = p_neu
						if (pre_min_neu > p_neu): #calculating the minimum precision for the neutral
							pre_min_neu = p_neu
							
						if (f1_max_pos < f_pos): #calculating the maximum f1 for the positive
							f1_max_pos = f_pos
						if (f1_min_pos > f_pos): #calculating the minimum f1 for the positive
							f1_min_pos = f_pos
							
						if (f1_max_neg < f_neg): #calculating the maximum f1 for the negative
							f1_max_neg = f_neg
						if (f1_min_neg > f_neg): #calculating the minimum f1 for the negative
							f1_min_neg = f_neg
							
						if (f1_max_neu < f_neu): #calculating the maximum f1 for the neutral
							f1_max_neu = f_neu
						if (f1_min_neu > f_neu): #calculating the minimum f1 for the neutral
							f1_min_neu = f_neu
			
			prev_word = word     
			
		avg_pre_pos = float(pre_pos)/c if c>0 else 0. #calculating the average precision
		avg_pre_neg = float(pre_neg)/c if c>0 else 0.
		avg_pre_neu = float(pre_neu)/c if c>0 else 0.

		avg_f1_pos = float(f1_pos)/c if c>0 else 0.
		avg_f1_neg = float(f1_neg)/c if c>0 else 0.
		avg_f1_neu = float(f1_neu)/c if c>0 else 0.    
				   
		return( [avg_pre_pos, avg_pre_neg, avg_pre_neu,
				   avg_f1_pos, avg_f1_neg, avg_f1_neu] )
		
	#returns pos tag gbigrams
	def calculateBigrams(self,postags):
		pos_bigrams = []
		for pos in postags:
			tmp_bigrams = []
			for i in range(len(pos)):
				if ( (i+1) == len(pos) ):
					break
				else:
					tmp = [pos[i], pos[i+1]]
					tmp_bigrams.append(tmp)
			pos_bigrams.append(tmp_bigrams)
			
		return (pos_bigrams)

	#returns the pos tag bigram score of each text 
	def calcScorePosBi(self, pos):
		bigrams = self.calculateBigrams(pos) #pos tags bigrams
		posBigramFeatures = []

		#caclulate the score for the pos bigrams for train end test data
		for bigram in bigrams:
			pre_pos,pre_neg,pre_neu,f1_pos,f1_neg,f1_neu,avg_pre_pos,avg_pre_neg,avg_pre_neu,avg_f1_pos,avg_f1_neg,avg_f1_neu = 0,0,0,0,0,0,0,0,0,0,0,0
			pre_max_pos, pre_max_neg, pre_max_neu, f1_max_pos, f1_max_neg, f1_max_neu = -1,-1,-1,-1,-1,-1
			pre_min_pos, pre_min_neg, pre_min_neu, f1_min_pos, f1_min_neg, f1_min_neu  = 1000000,1000000,1000000,1000000,1000000,1000000
			c = 0
			for i in range(len(bigram)):
				for j in range(len(self.train_posBigram_lexicon)):
					if (bigram[i][0].lower() == self.train_posBigram_lexicon[j][0][0]) and (bigram[i][1].lower() == self.train_posBigram_lexicon[j][0][1]):
						pre_pos = pre_pos + self.train_posBigram_lexicon[j][1][0]
						pre_neg = pre_neg + self.train_posBigram_lexicon[j][1][1]
						pre_neu = pre_neu + self.train_posBigram_lexicon[j][1][2]
						f1_pos = f1_pos + self.train_posBigram_lexicon[j][1][3]
						f1_neg = f1_neg + self.train_posBigram_lexicon[j][1][4]
						f1_neu = f1_neu + self.train_posBigram_lexicon[j][1][5]
						c = c + 1
						
						if (pre_max_pos < self.train_posBigram_lexicon[j][1][0]): #calculating the maximum precision for the positive
							pre_max_pos = self.train_posBigram_lexicon[j][1][0]
						if (pre_min_pos > self.train_posBigram_lexicon[j][1][0]): #calculating the minimum precision for the positive
							pre_min_pos = self.train_posBigram_lexicon[j][1][0]
							
						if (pre_max_neg < self.train_posBigram_lexicon[j][1][1]): #calculating the maximum precision for the negative
							pre_max_neg = self.train_posBigram_lexicon[j][1][1]
						if (pre_min_neg > self.train_posBigram_lexicon[j][1][1]): #calculating the minimum precision for the negative
							pre_min_neg = self.train_posBigram_lexicon[j][1][1]
							
						if (pre_max_neu < self.train_posBigram_lexicon[j][1][2]): #calculating the maximum precision for the neutral
							pre_max_neu = self.train_posBigram_lexicon[j][1][2]
						if (pre_min_neu > self.train_posBigram_lexicon[j][1][2]): #calculating the minimum precision for the neutral
							pre_min_neu = self.train_posBigram_lexicon[j][1][2]
							
						if (f1_max_pos < self.train_posBigram_lexicon[j][1][3]): #calculating the maximum f1 for the positive
							f1_max_pos = self.train_posBigram_lexicon[j][1][3]
						if (f1_min_pos > self.train_posBigram_lexicon[j][1][3]): #calculating the minimum f1 for the positive
							f1_min_pos = self.train_posBigram_lexicon[j][1][3]
							
						if (f1_max_neg < self.train_posBigram_lexicon[j][1][4]): #calculating the maximum f1 for the negative
							f1_max_neg = self.train_posBigram_lexicon[j][1][4]
						if (f1_min_neg > self.train_posBigram_lexicon[j][1][4]): #calculating the minimum f1 for the negative
							f1_min_neg = self.train_posBigram_lexicon[j][1][4]
							
						if (f1_max_neu < self.train_posBigram_lexicon[j][1][5]): #calculating the maximum f1 for the neutral
							f1_max_neu = self.train_posBigram_lexicon[j][1][5]
						if (f1_min_neu > self.train_posBigram_lexicon[j][1][5]): #calculating the minimum f1 for the neutral
							f1_min_neu = self.train_posBigram_lexicon[j][1][5]
			
			avg_pre_pos = float(pre_pos)/c if c>0 else 0. #calculating the average precision
			avg_pre_neg = float(pre_neg)/c if c>0 else 0.
			avg_pre_neu = float(pre_neu)/c if c>0 else 0.
			
			avg_f1_pos = float(f1_pos)/c if c>0 else 0.
			avg_f1_neg = float(f1_neg)/c if c>0 else 0.
			avg_f1_neu = float(f1_neu)/c if c>0 else 0.
					
			tmp = [avg_pre_pos, avg_pre_neg, avg_pre_neu,
				   avg_f1_pos, avg_f1_neg, avg_f1_neu]
			
			posBigramFeatures.append(tmp)
			
		return posBigramFeatures
		

	#----------------------------------------------------------------------------------------------------------------------------------------------------

	
	def train(self,dom):
		
		temp_vector = []
		train_tags = []
		train_pos = []
		train_vector = []

		chars_to_remove = ['=', '!', '?', ',', '<', '.', '>', '/', ';' ,':', ']', '}', '[', '{', '|', '@', '$', '%', '^', '&', '*', '(', ')', '_', '-', '+', '+', '"','1', '2', '3','4','5','6','7','8','9' ] #removing characters

		reviews = ET.parse(self.train_path).getroot().findall('Review')
		for review in reviews:
			sentences = review[0] #get the sentences
			for sentence in sentences:
				if (len(sentence) > 1):
					opinions = sentence[1] #getting the opinions field
					if ( len(opinions) > 0): #check if there are aspects 
						
						t = sentence[0].text
						t2 = word_tokenize(t) #tokenize, don't convert to lower case, check for caps
						capsList, capsCounter = self.listCaps(t2) #storing the caps words of the text
						text = word_tokenize(t.lower()) #tokenize, convert to lower case
						
						for opinion in opinions: 
							category = opinion.attrib['polarity']    
							train_tags.append(category) #store the category
							train_pos.append(t) #store the text for the pos tagging 

							#caclulate score for each lexicon
							temp0 = self.checkLexiconReady(text, self.stopwords, capsList, self.negation, self.AFINN_lexicon) #afinn lexicon scores
							temp3 = self.checkLexiconReady(text, self.stopwords, capsList, self.negation, self.huLiu_lexicon) #Hu and Liu lexicon scores
							temp4 = self.checkLexiconReady(text, self.stopwords, capsList, self.negation, self.nrc_lexicon) #NRC lexicon scores
							
							temp1 = self.checkLexiconUni(text, chars_to_remove, self.stopwords, capsList, self.negation, self.train_unigram_lexicon) #unigram lexicon scores from the train data of each domain

							temp7 = self.howManyUpperFirst(t2) #num of words starting with capitalized first letter
							temp9 = [self.howMany(text, '?'), self.howMany(text, '!')] #number of question and exclamation marks
							temp11 = self.lastSymbol(t2) #is the last symbol a question or an exclamation mark
							
							cat = opinion.attrib['category'].split('#') #a feature for the entity and the attribute
							cat0 = []
							for ent in self.entities:
								if ent == cat[0]:
									cat0.append(1)
								else:
									cat0.append(0)
							cat1 = []
							for attr in self.attributes:
								if attr == cat[1]:
									cat1.append(1)
								else:
									cat1.append(0)
							temp12 = [len(opinions)] + cat0 + cat1

							temp = temp0 + temp1 + temp3 + temp4 + temp7 + temp9 + temp11 + [capsCounter] + temp12
							temp_vector.append(temp) #creating the features vector

		temp_vector = self.normalize(temp_vector) #normalize the vector

		pos = arktagger.pos_tag_list(train_pos) #getting the pos tags
		train_pos = self.howManyPos(pos) #calculating the number of the pos tags

		train_pos_bi = self.calcScorePosBi(pos) #caclulating the pos tags bigram scores for each text
		train_pos_bi = self.normalize(train_pos_bi)

		for i in range(len(temp_vector)): #join the matrices
			train_vector.append(temp_vector[i] + train_pos[i] + train_pos_bi[i])

		print
		print '---- End of train ----'

		return train_vector,train_tags

	def test(self,dom):

		test_pos = []
		temp_vector = []
		test_vector = []

		chars_to_remove = ['=', '!', '?', ',', '<', '.', '>', '/', ';' ,':', ']', '}', '[', '{', '|', '@', '$', '%', '^', '&', '*', '(', ')', '_', '-', '+', '+', '"','1', '2', '3','4','5','6','7','8','9' ] #removing characters

		reviews = ET.parse(self.test_path).getroot().findall('Review')
		for review in reviews:
			sentences = review[0] #get the sentences
			for sentence in sentences:
				if (len(sentence) > 1):
					opinions = sentence[1]
					
					if ( len(opinions) > 0): #check if there are aspects 
						t = sentence[0].text
						t2 = word_tokenize(t)
						capsList, capsCounter = self.listCaps(t2) #storing the caps words of the text
						text = word_tokenize(t.lower())
						
						for opinion in opinions:
							test_pos.append(t)
							
							#calculate score for each lexicon
							temp0 = self.checkLexiconReady(text, self.stopwords, capsList, self.negation, self.AFINN_lexicon) #Afinn lexicon scores
							temp3 = self.checkLexiconReady(text, self.stopwords, capsList, self.negation, self.huLiu_lexicon) #Hu and Liu lexicon scores
							temp4 = self.checkLexiconReady(text, self.stopwords, capsList, self.negation, self.nrc_lexicon) #NRC lexicon scores
							
							temp1 = self.checkLexiconUni(text, chars_to_remove, self.stopwords, capsList, self.negation, self.train_unigram_lexicon) #unigram lexicon scores from the train data of each domain
							
							temp7 = self.howManyUpperFirst(t2) #num of words starting with capitalized first letter
							temp9 = [self.howMany(text, '?'), self.howMany(text, '!')] #number of question and exclamation marks
							temp11 = self.lastSymbol(t2) #is the last symbol a question or an exclamation mark
													
							cat = opinion.attrib['category'].split('#') #a feature for the entity and the attribute
							cat0 = []
							for ent in self.entities:
								if ent == cat[0]:
									cat0.append(1)
								else:
									cat0.append(0)
							cat1 = []
							for attr in self.attributes:
								if attr == cat[1]:
									cat1.append(1)
								else:
									cat1.append(0)
							temp12 = [len(opinions)] + cat0 + cat1
							
							temp = temp0 + temp1 + temp3 + temp4 + temp7 + temp9 + temp11 + [capsCounter] + temp12
							temp_vector.append(temp) #creating the features vector
		  
		temp_vector = self.normalize(temp_vector) #normalize the vector              
						
		pos = arktagger.pos_tag_list(test_pos) #finding the pos tags            
		test_pos = self.howManyPos(pos)

		test_pos_bi = self.calcScorePosBi(pos)
		test_pos_bi = self.normalize(test_pos_bi)

		for i in range(len(temp_vector)): #join the matrices
			test_vector.append(temp_vector[i] + test_pos[i] + test_pos_bi[i])

		print
		print '---- End of Test ----'
		
		return test_vector

	def results(self, train_vector, train_tags, test_vector, dom):		
			
		if dom == 'lap':
			logistic = linear_model.LogisticRegression(C=0.1) #fit logistic
			logistic.fit(train_vector,train_tags)
			resLogistic = logistic.predict_proba(test_vector)
			return resLogistic		
		else:
			logistic = linear_model.LogisticRegression(C=0.32) #fit logistic
			logistic.fit(train_vector,train_tags)
			resLogistic = logistic.predict_proba(test_vector)
			return resLogistic