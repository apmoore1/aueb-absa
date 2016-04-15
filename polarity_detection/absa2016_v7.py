#author Panagiotis Theodorakakos

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import codecs
import xml.etree.ElementTree as ET
import absa2016_v5
import absa2016_v6

#UNCOMMENT TO USE EMBEDDINGS MODEL
#import gensim
#from gensim.models import Word2Vec
#m = gensim.models.Word2Vec.load('model.bin') #load the model

#---------- use the helpingEmbeddings.txt instead of the embeddings model ----------
e = open("helpingEmbeddings.txt","r")
tmp = []
for a in e:
	tmp.append(a.split(" ")) #store the lexicon in an array
e.close()

m = []
tmpE = []
for line in tmp:
	for i in range(len(line)):
		l = line[i]
		if '[' in l:
			word = line[i-1]
			if len(l) != 1:
				tmpE.append(l[1:]) #handles cases where there is no space in the first cell
		elif ']' in l:
			l = l.replace('\n','').replace('\r','').replace(']','')
			if l !='' and l != ']':
				tmpE.append(l)
			
			tmpModel = []
			for i in range(len(tmpE)):
				if i != 0:
					tmpModel.append(float(tmpE[i])) #converting string to floats
			
			#convert to numpy array
			m.append([word,np.array(tmpModel)]) #append: word word_embeddings, skip the first cell of tempE containig the word
			tmpE = []
		else:
			l = l.replace('\n','').replace('\r','').replace(']','')
			if l !='' and l != ']':
				tmpE.append(l)
#---------- use the helpingEmbeddings.txt instead of the embeddings model ----------


print '---------------- Restaurants ----------------'
print

print('-------- Features Model--------')
fea1 = absa2016_v5.features('restaurants/ABSA16_Restaurants_Train_SB1_v2.xml','restaurants/EN_REST_SB1_TEST.xml','rest')
train_vector,train_tags = absa2016_v5.features.train(fea1,'rest')
test_vector = absa2016_v5.features.test(fea1,'rest')
predictionsRest1 = absa2016_v5.features.results(fea1, train_vector, train_tags, test_vector,'rest')
print 'End version 5'

print('-------- Embeddings Model--------')
fea1 = absa2016_v6.features('restaurants/ABSA16_Restaurants_Train_SB1_v2.xml','restaurants/EN_REST_SB1_TEST.xml',m)
train_vector,train_tags = absa2016_v6.features.train(fea1)
test_vector = absa2016_v6.features.test(fea1)
predictionsRest2 = absa2016_v6.features.results(fea1, train_vector, train_tags, test_vector,'rest') #store probabilities for each of the three class for each sentence
print 'End version 6'

#both methods "vote"
l = len(predictionsRest1)
predictionsRest = []

#weights for each method. each method's vote weights the same
w1 = 0.5
w2 = 0.5

for i in range(l):
	a = float(predictionsRest1[i][0]*w1 + predictionsRest2[i][0]*w2)/2 #number of the methods we are using
	b = float(predictionsRest1[i][1]*w1 + predictionsRest2[i][1]*w2)/2
	c = float(predictionsRest1[i][2]*w1 + predictionsRest2[i][2]*w2)/2
	
	if a > b and a > c:
		predictionsRest.append('negative') #check the probabilities
	elif b > a and b > c:
		predictionsRest.append('neutral')
	elif c > a and c > b:
		predictionsRest.append('positive')

#creating the xml
r = []

reviews = ET.parse('restaurants/EN_REST_SB1_TEST.xml').getroot().findall('Review')
for review in reviews:
	
	flag = False
	rid = review.attrib['rid'] #get the review id
	sentences = review[0] #get the sentences
	
	sid = []
	text = [] #store the text
	cat = []
	fr2 = []
	t2 = []
	target2 = []
	
	for sentence in sentences:
		if (len(sentence) > 1):
			opinions = sentence[1]
			if ( len(opinions) > 0): #check if there are aspects
				flag = True
				sid.append(sentence.attrib['id']) #get the sentence id	
				
				text.append((sentence[0].text)) #get the text
				category = [] #store the category
				target = [] #get the target
				pr_polarity = [] #store the predicted polarity	
				fr = [] #store the from 
				t = [] #store the to
				
				for i in range(len(opinions)):
				
					target.append(opinions[i].attrib['target']) #get the target
					category.append(opinions[i].attrib['category']) #get the category
					fr.append(opinions[i].attrib['from'])
					t.append(opinions[i].attrib['to'])
					
					if i == len(opinions) - 1:
						cat.append(category)
						fr2.append(fr)
						t2.append(t)
						target2.append(target)
	if flag:				
		r.append([rid,sid,text,target2,cat,fr2,t2])

counter = 0
#print the output in an xml file
with codecs.open('AUEB-ABSA_REST_EN_B_SB1_3_1_U.xml', 'w','utf-8') as o:
	o.write('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n') 
	o.write('<Reviews>\n')
	for review in r:
		o.write('\t<Review rid="%s">\n' % (review[0]))
		o.write('\t\t<sentences>\n')
		for i in range(len(review[1])):
			o.write('\t\t\t<sentence id="%s">\n' % (review[1][i]))
			tmp = review[2][i]
			tmp = tmp.replace("&", '&amp;')
			o.write('\t\t\t\t<text>%s</text>\n' % (tmp) )
			o.write('\t\t\t\t<Opinions>\n')
			for j in range(len(review[3][i])):
				o.write('\t\t\t\t\t<Opinion target="%s" category="%s" polarity="%s" from="%s" to="%s"/>\n' %(review[3][i][j], review[4][i][j], predictionsRest[counter], review[5][i][j],  review[6][i][j]) )
				counter = counter + 1
			o.write('\t\t\t\t</Opinions>\n')
			o.write('\t\t\t</sentence>\n')
		o.write('\t\t</sentences>\n')
		o.write('\t</Review>\n')
	o.write('</Reviews>')



#============================================================================================================================



print '---------------- Laptops ----------------'
print

print('-------- Features Model--------')
fea2 = absa2016_v5.features('laptops/ABSA16_Laptops_Train_SB1_v2.xml','laptops/EN_LAPT_SB1_TEST_.xml','lap')
train_vector,train_tags = absa2016_v5.features.train(fea2,'lap')
test_vector = absa2016_v5.features.test(fea2,'lap')
predictionsLap1 = absa2016_v5.features.results(fea2, train_vector, train_tags, test_vector,'lap')
print 'End version 5'

print('-------- Embeddings Model--------')
fea2 = absa2016_v6.features('laptops/ABSA16_Laptops_Train_SB1_v2.xml','laptops/EN_LAPT_SB1_TEST_.xml',m)
train_vector,train_tags = absa2016_v6.features.train(fea2)
test_vector = absa2016_v6.features.test(fea2)
predictionsLap2 = absa2016_v6.features.results(fea2, train_vector, train_tags, test_vector,'lap') #store probabilities for each of the three class for each sentence
print 'End version 6'

#both methods "vote"
l = len(predictionsLap1)
predictionsLap = []
for i in range(l):
	a = float(predictionsLap1[i][0]*w1 + predictionsLap2[i][0]*w2)/2 #number of the methods we are using
	b = float(predictionsLap1[i][1]*w1 + predictionsLap2[i][1]*w2)/2
	c = float(predictionsLap1[i][2]*w1 + predictionsLap2[i][2]*w2)/2
	
	if a > b and a > c:
		predictionsLap.append('negative') #check the probabilities
	elif b > a and b > c:
		predictionsLap.append('neutral')
	elif c > a and c > b:
		predictionsLap.append('positive')

		
#creating the xml
r = [] #store the rid the sid's the text the categories and the polarities

reviews = ET.parse('laptops/EN_LAPT_SB1_TEST_.xml').getroot().findall('Review')

for review in reviews:
	flag = False
	rid = review.attrib['rid'] #get the review id
	sentences = review[0] #get the sentences
	sid = []
	text = [] #store the text
	cat = []
	for sentence in sentences:
		if (len(sentence) > 1):
			opinions = sentence[1]
			if ( len(opinions) > 0): #check if there are aspects
				flag = True
				
				sid.append(sentence.attrib['id']) #get the sentence id
				text.append((sentence[0].text)) #get the text
				category = [] #store the category
				pr_polarity = [] #store the predicted polarity
				for i in range(len(opinions)):
					category.append(opinions[i].attrib['category']) #get the category
					if i == len(opinions) - 1:
						cat.append(category)
	if flag:
		r.append([rid,sid,text,cat])

counter = 0
#print the output in an xml file
with codecs.open('AUEB-ABSA_LAPT_EN_B_SB1_3_1_U.xml', 'w') as o:
	o.write('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n')
	o.write('<Reviews>\n')
	for review in r:
		o.write('\t<Review rid="%s">\n' % (review[0]))
		o.write('\t\t<sentences>\n')
		for i in range(len(review[1])):
			o.write('\t\t\t<sentence id="%s">\n' % (review[1][i]))
			o.write('\t\t\t\t<text>%s</text>\n' % ( review[2][i] ))
			o.write('\t\t\t\t<Opinions>\n')
			for j in range(len(review[3][i])):
				o.write('\t\t\t\t\t<Opinion category="%s" polarity="%s" \t/>\n' %(review[3][i][j], predictionsLap[counter]) )
				counter = counter + 1
			o.write('\t\t\t\t</Opinions>\n')
			o.write('\t\t\t</sentence>\n')
		o.write('\t\t</sentences>\n')
		o.write('\t</Review>\n')
	o.write('</Reviews>')