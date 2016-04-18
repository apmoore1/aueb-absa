#!/usr/bin/env python 

'''
**Aspect Category Detection for the 5th task of SemEval 2016**
Constrained Submission for the Restaurants domain

Run from the terminal:
>>> python acd_constrained_restaurants.py --train train.xml --test test.xml
'''

try:
    import xml.etree.ElementTree as ET, getopt, logging, sys, random, re, copy, os
    import numpy as np
    from collections import Counter
    import operator
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.svm import SVC
    import nltk
    from nltk.stem import PorterStemmer
    from xml.sax.saxutils import escape
except:
    sys.exit('Some package is missing... Perhaps <re>?')

# Stopwords, imported from NLTK (v 2.0.4)
stopwords = set(
    ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves',
     'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
     'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was',
     'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the',
     'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
     'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in',
     'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
     'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
     'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'])

contractions = re.compile(r"'|-|\"")
# all non alphanumeric
symbols = re.compile(r'(\W+)', re.U)
# single character removal
singles = re.compile(r'(\s\S\s)', re.I|re.U)
# separators (any whitespace)
seps = re.compile(r'\s+')

def validate(filename):
    '''Validate an XML file, w.r.t. the format given in the 5th task of **SemEval '16**.'''
    tree = ET.parse(filename)
    root = tree.getroot()

    elements = []	
    aspect_terms = []
    for review in root.findall('Review'):
        for sentences in review.findall('sentences'):
            for sentence in sentences.findall('sentence'):
                elements.append(sentence)
						
    return elements
	
def extract_subjectives(filename, new_filename):
	'''Extract only the subjective sentences and leave out the objective sentences and the sentences with the attribute 'OutOfScope=="TRUE".'''
	tree = ET.parse(filename)
	root = tree.getroot()
		
	with open(new_filename, 'w') as o:
		o.write('<Reviews>\n')
		for review in root.findall('Review'):
			o.write('\t<Review rid="%s">\n' % review.get('rid'))
			for sentences in review.findall('sentences'):
				o.write('\t\t<sentences>\n')
				for sentence in sentences.findall('sentence'):
					if (sentence.get('OutOfScope') != "TRUE"):
						if sentence.find('Opinions') != None:
							o.write('\t\t\t<sentence id="%s">\n' % (sentence.get('id')))
							o.write('\t\t\t\t<text>%s</text>\n' % (fix(sentence.find('text').text)))       
							for opinions in sentence.findall('Opinions'):
								o.write('\t\t\t\t<Opinions>\n')
								for opinion in opinions.findall('Opinion'):
									o.write('\t\t\t\t\t<Opinion category="%s" polarity="%s" from="%s" to="%s"/>\n' % (
										fix(opinion.get('category')), opinion.get('polarity'), opinion.get('from'), opinion.get('to')))
								o.write('\t\t\t\t</Opinions>\n')
							o.write('\t\t\t</sentence>\n')
				o.write('\t\t</sentences>\n')
			o.write('\t</Review>\n')
		o.write('</Reviews>')

def leave_outOfScope(filename, new_filename):
	'''Leave out sentences with the attribute 'OutOfScope=="TRUE".'''
	tree = ET.parse(filename)
	root = tree.getroot()
		
	with open(new_filename, 'w') as o:
		o.write('<Reviews>\n')
		for review in root.findall('Review'):
                        o.write('\t<Review rid="%s">\n' % review.get('rid'))
			for sentences in review.findall('sentences'):
				o.write('\t\t<sentences>\n')
				for sentence in sentences.findall('sentence'):
				    if (sentence.get('OutOfScope') != "TRUE"):						
					    o.write('\t\t\t<sentence id="%s">\n' % (sentence.get('id')))
					    o.write('\t\t\t\t<text>%s</text>\n' % (fix(sentence.find('text').text)))       
					    for opinions in sentence.findall('Opinions'):
						    o.write('\t\t\t\t<Opinions>\n')
						    for opinion in opinions.findall('Opinion'):
							o.write('\t\t\t\t\t<Opinion category="%s" polarity="%s" from="%s" to="%s"/>\n' % (
								fix(opinion.get('category')), opinion.get('polarity'), opinion.get('from'), opinion.get('to')))
						    o.write('\t\t\t\t</Opinions>\n')
					    o.write('\t\t\t</sentence>\n')
				o.write('\t\t</sentences>\n')
			o.write('\t</Review>\n')
		o.write('</Reviews>')
	
	
fix = lambda text: escape(text.encode('utf8')).replace('\"', '&quot;')
'''Simple fix for writing out text.'''


def load_lexicon(lex_type, b):
    '''Create a set (lexicon) containing all words found in the train data.'''

    #entity lexica
    food = []
    drinks = []
    service = []
    ambience = []
    location = []
    restaurant = []

    #attribute lexica
    gen = []
    pr = []
    qual = []
    style = []
    misc = []

    f = open(lex_type+"_food_lexicon.txt", "r")
    for line in f:
        feats = line.split()
        if b is True:
            feats[0] = tuple(feats[0].split(','))
        food.append([feats[0], float(feats[1]), float(feats[2]), float(feats[3]), float(feats[4])])
    
    f = open(lex_type+"_drinks_lexicon.txt", "r")
    for line in f:
        feats = line.split()
        if b is True:
            feats[0] = tuple(feats[0].split(','))
        drinks.append([feats[0], float(feats[1]), float(feats[2]), float(feats[3]), float(feats[4])])

    f = open(lex_type+"_service_lexicon.txt", "r")
    for line in f:
        feats = line.split()
        if b is True:
            feats[0] = tuple(feats[0].split(','))
        service.append([feats[0], float(feats[1]), float(feats[2]), float(feats[3]), float(feats[4])])

    f = open(lex_type+"_ambience_lexicon.txt", "r")
    for line in f:
        feats = line.split()
        if b is True:
            feats[0] = tuple(feats[0].split(','))
        ambience.append([feats[0], float(feats[1]), float(feats[2]), float(feats[3]), float(feats[4])])

    f = open(lex_type+"_location_lexicon.txt", "r")
    for line in f:
        feats = line.split()
        if b is True:
            feats[0] = tuple(feats[0].split(','))
        location.append([feats[0], float(feats[1]), float(feats[2]), float(feats[3]), float(feats[4])])

    f = open(lex_type+"_restaurant_lexicon.txt", "r")
    for line in f:
        feats = line.split()
        if b is True:
            feats[0] = tuple(feats[0].split(','))
        restaurant.append([feats[0], float(feats[1]), float(feats[2]), float(feats[3]), float(feats[4])])

    f = open(lex_type+"_general_lexicon.txt", "r")
    for line in f:
        feats = line.split()
        if b is True:
            feats[0] = tuple(feats[0].split(','))
        gen.append([feats[0], float(feats[1]), float(feats[2]), float(feats[3]), float(feats[4])])

    f = open(lex_type+"_price_lexicon.txt", "r")
    for line in f:
        feats = line.split()
        if b is True:
            feats[0] = tuple(feats[0].split(','))
        pr.append([feats[0], float(feats[1]), float(feats[2]), float(feats[3]), float(feats[4])])

    f = open(lex_type+"_quality_lexicon.txt", "r")
    for line in f:
        feats = line.split()
        if b is True:
            feats[0] = tuple(feats[0].split(','))
        qual.append([feats[0], float(feats[1]), float(feats[2]), float(feats[3]), float(feats[4])])

    f = open(lex_type+"_style_options_lexicon.txt", "r")
    for line in f:
        feats = line.split()
        if b is True:
            feats[0] = tuple(feats[0].split(','))
        style.append([feats[0], float(feats[1]), float(feats[2]), float(feats[3]), float(feats[4])])

    f = open(lex_type+"_miscellaneous_lexicon.txt", "r")
    for line in f:
        feats = line.split()
        if b is True:
            feats[0] = tuple(feats[0].split(','))
        misc.append([feats[0], float(feats[1]), float(feats[2]), float(feats[3]), float(feats[4])])
    
    f.close()
    
    return [food, drinks, service, ambience, location, restaurant, gen, pr, qual, style, misc]


class Category:
    '''Category objects contain the term of the category (e.g., food, price, etc.) of a sentence.'''

    def __init__(self, term=''):
        self.term = term

    def create(self, element):
        self.term = element.attrib['category']
        return self

    def update(self, term=''):
        self.term = term	
		

class Instance:
    '''An instance is a sentence, modeled out of XML (pre-specified format). It contains the text, and any aspect categories.'''

    def __init__(self, element):
        self.text = element.find('text').text
        self.id = element.get('id')
        self.aspect_categories = [Category(term='').create(e) for es in element.findall('Opinions')
                                  for e in es if
                                  es is not None]

    def get_aspect_categories(self):
        return [c.term.lower() for c in self.aspect_categories]

    def add_aspect_category(self, term):
        c = Category(term)
        self.aspect_categories.append(c)


class Corpus:
    '''A corpus contains instances, and is useful for training algorithms or splitting to train/test files.'''

    def __init__(self, elements):
        self.corpus = [Instance(e) for e in elements]
        self.size = len(self.corpus)
        self.texts = [t.text for t in self.corpus]

    def clean_tags(self):
        for i in range(len(self.corpus)):
            self.corpus[i].aspect_terms = []

    def split(self, threshold, shuffle=False):
        '''Split to train/test, based on a threshold. Turn on shuffling for randomizing the elements beforehand.'''
        clone = copy.deepcopy(self.corpus)
        if shuffle: random.shuffle(clone)
        train = clone[:int(threshold * self.size)]
        test = clone[int(threshold * self.size):]
        return train, test

    def write_out(self, filename, instances, short=True):
        with open(filename, 'w') as o:
            o.write('<sentences>\n')
            for i in instances:
                o.write('\t<sentence id="%s">\n' % (i.id))
                o.write('\t\t<text>%s</text>\n' % fix(i.text))
                o.write('\t\t<Opinions>\n')
                if not short:
                    for c in i.aspect_categories:
                        o.write('\t\t\t<Opinion category="%s"/>\n' % (fix(c.term)))
                o.write('\t\t</Opinions>\n')
                o.write('\t</sentence>\n')
            o.write('</sentences>')
			
def calculate_feats(n):
    '''Calculate the max, min, median and avg of a list with floats'''
    
    max_ = max(n) if n else 0.
    min_ = min(n) if n else 0.
    avg = np.average(n) if n else 0.
    median = np.median(n) if n else 0.
    
    return max_, min_, avg, median

def append_feats_to_list(l, feats):
    '''Append the features to the feature list.'''

    for f in feats:
        l.append(f)
    return l

def assign_features(lexica, words, flag):
    feature_list = []
    
    for lex in lexica:
        precision = []
        recall = []
        f1 = []
        for entry in lex:
            for w in words:
                if flag is False: #can't compare tuples (like in bigrams) with the '==' operator
                    if w == entry[0]:
                        precision.append(entry[2])
                        recall.append(entry[3])
                        f1.append(entry[4])
                else: #so you use the 'in' keyword
                    if w in entry[0]:
                        precision.append(entry[2])
                        recall.append(entry[3])
                        f1.append(entry[4])
        pre_max, pre_min, pre_avg, pre_median = calculate_feats(precision)
        re_max, re_min, re_avg, re_median = calculate_feats(recall)
        f1_max, f1_min, f1_avg, f1_median = calculate_feats(f1)
        feature_list = append_feats_to_list(feature_list, [pre_max, pre_min, pre_avg, pre_median, re_max, re_min,
                                                                re_avg, re_median, f1_max, f1_min, f1_avg, f1_median])

    return feature_list

# cleaner (order matters)
def clean(text): 
    text = text.lower()
    text = contractions.sub('', text)
    text = symbols.sub(r' \1 ', text)
    text = singles.sub(' ', text)
    text = seps.sub(' ', text)
    return text

def classify(traincorpus, testcorpus):

    # classifiers with the dicts as features
    food_clf1 = SVC(kernel='rbf', C=152.2185107203483, gamma=0.009290680585958758, probability=True)
    drinks_clf1 = SVC(kernel='sigmoid', C=215.2694823049509, gamma=0.015625, probability=True)
    service_clf1 = SVC(kernel='rbf', C=2.0, gamma=0.21022410381342863, probability=True)
    ambience_clf1 = SVC(kernel='sigmoid', C=107.63474115247546, gamma=0.02209708691207961, probability=True)
    location_clf1 = SVC(kernel='rbf', C=16, gamma=0.07432544468767006, probability=True)
    restaurant_clf1 = SVC(kernel='rbf', C=32, gamma=0.026278012976678578, probability=True)
    general_clf1 = SVC(kernel='rbf', C=4.756828460010884, gamma=0.07432544468767006, probability=True)
    price_clf1 = SVC(kernel='rbf', C=6.727171322029716, gamma=0.14865088937534013, probability=True)
    quality_clf1 = SVC(kernel='poly', C=0.0011613350732448448, gamma=1.4142135623730951, probability=True)
    style_clf1 = SVC(kernel='linear', C=0.42044820762685725, probability=True)
    misc_clf1 = SVC(kernel='rbf', C=256.0, gamma=0.0065695032441696445, probability=True)

    stemmer = PorterStemmer()
    
    unigrams_lexica = load_lexicon("lexica/restaurants/acd/unigrams", False)
    bigrams_lexica = load_lexicon("lexica/restaurants/acd/bigram", True)
    bipos_lexica = load_lexicon("lexica/restaurants/acd/bipos", True)
    stemmed_unigrams_lexica = load_lexicon("lexica/restaurants/acd/stemmed_unigrams", False)
    stemmed_bigrams_lexica = load_lexicon("lexica/restaurants/acd/stemmed_bigrams", True)

    train_sentences1 = [] #the sentences to be used to store our features

    #entity labels
    food_labels = []
    drinks_labels = []
    service_labels = []
    ambience_labels = []
    location_labels = []
    restaurant_labels = []

    #attribute labels
    general_labels = []
    price_labels = []
    quality_labels = []
    style_labels = []
    misc_labels = []

    #to calculate the number of categories
    cats = []

    print('Creating train feature vectors...')

    #extracting sentences and appending them labels
    for instance in traincorpus.corpus:
        words = (re.findall(r"[\w']+", instance.text.lower())) #the unigrams list

        stemmed_words = []
        stemmed_bi_words = []
        for w in words:
            if w not in stopwords:
                stemmed_words.append(stemmer.stem(w)) #the stemmed unigrams list
            stemmed_bi_words.append(stemmer.stem(w))
                
        stemmed_bigrams = nltk.bigrams(stemmed_bi_words)
        stemmed_bigrams_list = []
        for w in stemmed_bigrams:
            stemmed_bigrams_list.append(w) #the stemmed bigrams list
                
        bigram_words = nltk.bigrams(words)
        bigram_list = []
        for w in bigram_words:
            bigram_list.append(w) #the bigram list

        tags = nltk.pos_tag(words)
        tags_set = set() #the pos list
        for _, t in tags:
                tags_set.add(t)

        bitags = nltk.bigrams(list(tags_set))
        bitag_list = []
        for t in bitags:
            bitag_list.append(t) #the pos bigrams list

        unigrams_feats = []
        bigrams_feats = []
        bipos_feats = []
        stemmed_unigrams_feats = []
        stemmed_bigrams_feats = []            

        #unigrams features
        unigrams_feats = assign_features(unigrams_lexica, words, False)

        #bigrams features
        bigrams_feats = assign_features(bigrams_lexica, bigram_list, True)

        #pos bigrams features
        bipos_feats = assign_features(bipos_lexica, bitag_list, True)
                    
        #stemmed_unigram features
        stemmed_unigrams_feats = assign_features(stemmed_unigrams_lexica, stemmed_words, False)

        #stemmed_bigram features
        stemmed_bigrams_feats = assign_features(stemmed_bigrams_lexica, stemmed_bigrams_list, True)

        train_sentences1.append(unigrams_feats + bigrams_feats + bipos_feats + stemmed_unigrams_feats + stemmed_bigrams_feats)

        #to avoid training a sentence more than once to a category, since there are
        #categories like food#quality and food#price assigned to a sentence
        ent_set = set()
        attr_set = set()
        for c in instance.get_aspect_categories():
            ent_attr = c.split('#')
            ent_set.add(ent_attr[0]) #store the entity
            attr_set.add(ent_attr[1]) #store the attribute
            cats.append(c)
            
        #check entity category
        if "food" in ent_set:
            food_labels.append(1)
        else:
            food_labels.append(0)
            
        if "drinks" in ent_set:
            drinks_labels.append(1)
        else:
            drinks_labels.append(0)
            
        if "service" in ent_set:
            service_labels.append(1)
        else:
            service_labels.append(0)
            
        if "ambience" in ent_set:
            ambience_labels.append(1)
        else:
            ambience_labels.append(0)
            
        if "location" in ent_set:
            location_labels.append(1)
        else:
            location_labels.append(0)
            
        if "restaurant" in ent_set:
            restaurant_labels.append(1)
        else:
            restaurant_labels.append(0)
            
        #check attribute category
        if "general" in attr_set:
            general_labels.append(1)
        else:
            general_labels.append(0)
            
        if "prices" in attr_set:
            price_labels.append(1)
        else:
            price_labels.append(0)
            
        if "quality" in attr_set:
            quality_labels.append(1)
        else:
            quality_labels.append(0)
            
        if "style_options" in attr_set:
            style_labels.append(1)
        else:
            style_labels.append(0)
            
        if "miscellaneous" in attr_set:
            misc_labels.append(1)
        else:
            misc_labels.append(0)

    cat_dict = Counter(cats) #the dictionary containing all the categories seen and how many times they are seen

    train_features1 = np.asarray(train_sentences1) #the classifier needs an array, not a list
    
    food_clf1.fit(train_features1, food_labels)
    drinks_clf1.fit(train_features1, drinks_labels)
    service_clf1.fit(train_features1, service_labels)
    ambience_clf1.fit(train_features1, ambience_labels)
    location_clf1.fit(train_features1, location_labels)
    restaurant_clf1.fit(train_features1, restaurant_labels)   
    general_clf1.fit(train_features1, general_labels)
    price_clf1.fit(train_features1, price_labels)
    quality_clf1.fit(train_features1, quality_labels)
    style_clf1.fit(train_features1, style_labels)
    misc_clf1.fit(train_features1, misc_labels)
    
    print('Done!')
    print('Creating test feature vectors...')
    
    test_sentences1 = []
    for instance in testcorpus.corpus:
        words = (re.findall(r"[\w']+", instance.text.lower()))
        
        stemmed_words = []
        stemmed_bi_words = []
        for w in words:
            if w not in stopwords:
                stemmed_words.append(stemmer.stem(w))
            stemmed_bi_words.append(stemmer.stem(w))
            
        stemmed_bigrams = nltk.bigrams(stemmed_bi_words)
        stemmed_bigrams_list = []
        for w in stemmed_bigrams:
            stemmed_bigrams_list.append(w)
                
        bigram_words = nltk.bigrams(words)
        bigram_list = []
        for w in bigram_words:
            bigram_list.append(w)

        tags = nltk.pos_tag(words)
        tags_set = set()
        for _, t in tags:
            tags_set.add(t)

        bitags = nltk.bigrams(list(tags_set))
        bitag_list = []
        for t in bitags:
            bitag_list.append(t)

        unigrams_feats = []
        bigrams_feats = []
        bipos_feats = []
        stemmed_unigrams_feats = []
        stemmed_bigrams_feats = []

        #unigrams features
        unigrams_feats = assign_features(unigrams_lexica, words, False)

        #bigrams features
        bigrams_feats = assign_features(bigrams_lexica, bigram_list, True)

        #pos bigrams features
        bipos_feats = assign_features(bipos_lexica,bitag_list, True)
                
        #stemmed_unigram features
        stemmed_unigrams_feats = assign_features(stemmed_unigrams_lexica, stemmed_words, False)

        #stemmed_bigram features
        stemmed_bigrams_feats = assign_features(stemmed_bigrams_lexica, stemmed_bigrams_list, True) 
                    
        test_sentences1.append(unigrams_feats + bigrams_feats + bipos_feats + stemmed_unigrams_feats + stemmed_bigrams_feats)
        
    test_features1 = np.asarray(test_sentences1)

    print('Done!')
    print('Predicting categories...')
    
    for i, test_fvector1 in enumerate(test_features1):
            #we get the [0,1] index, because on the [0,0] is the prediction for the category '-'
            food_pred1 = food_clf1.predict_proba(test_fvector1.reshape(1,-1))[0,1]
            drinks_pred1 = drinks_clf1.predict_proba(test_fvector1.reshape(1,-1))[0,1]
            service_pred1 = service_clf1.predict_proba(test_fvector1.reshape(1,-1))[0,1]
            ambience_pred1 = ambience_clf1.predict_proba(test_fvector1.reshape(1,-1))[0,1]
            location_pred1 = location_clf1.predict_proba(test_fvector1.reshape(1,-1))[0,1]
            restaurant_pred1 = restaurant_clf1.predict_proba(test_fvector1.reshape(1,-1))[0,1]
            general_pred1 = general_clf1.predict_proba(test_fvector1.reshape(1,-1))[0,1]
            price_pred1 = price_clf1.predict_proba(test_fvector1.reshape(1,-1))[0,1]
            quality_pred1 = quality_clf1.predict_proba(test_fvector1.reshape(1,-1))[0,1]
            style_pred1 = style_clf1.predict_proba(test_fvector1.reshape(1,-1))[0,1]
            misc_pred1 = misc_clf1.predict_proba(test_fvector1.reshape(1,-1))[0,1]

            #dictionaries containing the probabilities for every E and A category
            entity_prob = {"food": food_pred1, "drinks": drinks_pred1,
                           "service": service_pred1, "ambience": ambience_pred1,
                            "location": location_pred1,
                           "restaurant": restaurant_pred1}

            attr_prob = {"general": general_pred1, "prices": price_pred1,
                         "quality": quality_pred1, "style_options": style_pred1,
                        "miscellaneous": misc_pred1}

            sorted_entity_prob = sorted(entity_prob.items(), key=operator.itemgetter(1), reverse=True)
            sorted_attr_prob = sorted(attr_prob.items(), key=operator.itemgetter(1), reverse=True)
            categories = []
            
            for entity in sorted_entity_prob:
                for attr in sorted_attr_prob:
                    if entity[1] > 0.4 and attr[1] > 0.4:
                        category = entity[0]+'#'+attr[0]
                        for c in cat_dict:
			    #if the e#a exists in the category dictionary and has > 0 appearances
                            if category == c and cat_dict[c] > 0:
                                categories.append(category)
                            
            testcorpus.corpus[i].aspect_categories = ([Category(term=c) for c in categories])
            
    print('Done!')
    return testcorpus.corpus

class Evaluate():

    def __init__(self, correct, predicted):
        self.size = len(correct)
        self.correct = correct
        self.predicted = predicted

    # Aspect Category Detection
    def category_detection(self, b=1):
        common, relevant, retrieved = 0., 0., 0.
        for i in range(self.size):
            cor = set(self.correct[i].get_aspect_categories())
            # Use set to avoid duplicates (i.e., two times the same category)
            pre = set(self.predicted[i].get_aspect_categories())
            common += len([c for c in pre if c in cor])
            retrieved += len(pre)
            relevant += len(cor)
        p = common / retrieved if retrieved > 0 else 0.
        r = common / relevant
        f1 = (1 + b ** 2) * p * r / ((p * b ** 2) + r) if p > 0 and r > 0 else 0.
        return p, r, f1, common, retrieved, relevant

def semEval_format(testWithReviews, testWithoutReviewsACD):

    tree1 = ET.parse(testWithReviews)
    root1 = tree1.getroot()

    tree3 = ET.parse(testWithoutReviewsACD)
    root3 = tree3.getroot()

    with open('acd_const_rest_semeval.xml', 'w') as o:
        o.write('<Reviews>\n')
        for review in root1.findall('Review'):
            o.write('\t<Review rid="%s">\n' % review.get('rid'))
            for sentences in review.findall('sentences'):
                o.write('\t\t<sentences>\n')
                for sentence1 in sentences.findall('sentence'):
                    if (sentence1.get('OutOfScope') == "TRUE"):
                            o.write('\t\t\t<sentence id="%s" OutOfScope="TRUE">\n' % (sentence1.get('id')))
                            o.write('\t\t\t\t<text>%s</text>\n' % (fix(sentence1.find('text').text)))
                            o.write('\t\t\t</sentence>\n')
                    else:
                        for sentence3 in root3.findall('sentence'):
                            if sentence1.get('id') == sentence3.get('id'):
                                o.write('\t\t\t<sentence id="%s">\n' % (sentence3.get('id')))
                                o.write('\t\t\t\t<text>%s</text>\n' % (fix(sentence3.find('text').text)))
                                for opinions in sentence3.findall('Opinions'):
                                    if opinions.find('Opinion') != None:
                                        o.write('\t\t\t\t<Opinions>\n')
                                        for opinion in opinions.findall('Opinion'):
                                            o.write('\t\t\t\t\t<Opinion category="%s"/>\n' % (
                                                        fix(opinion.get('category').upper())))
                                        o.write('\t\t\t\t</Opinions>\n')
                                o.write('\t\t\t</sentence>\n')
                o.write('\t\t</sentences>\n')
            o.write('\t</Review>\n')
        o.write('</Reviews>')

			
def main(argv=None):
	# Parse the input
    opts, args = getopt.getopt(argv, "hg:dt:k:", ["help", "grammar", "train=", "test="])
    trainfile, testfile = None, None
    use_msg = 'Use as:\n">>> python acd_constrained_restaurants.py --train train.xml --test test.xml"\n\nThis will parse a train set, examine whether is valid, perform category detection on the test set provided, and write out a file with the predictions.'
    if len(opts) == 0: sys.exit(use_msg)
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            sys.exit(use_msg)
        elif opt in ('-t', "--train"):
            trainfile = arg
        elif opt in ('-k', "--test"):
            testfile = arg
    if trainfile is None or testfile is None: sys.exit(use_msg)

    # Examine if the file is in proper XML format for further use.
    print ('Validating the file...')
    try:
        sentences = validate(trainfile)
        print ('PASSED! This corpus has: %d sentences.' % (len(sentences)))
    except:
        print ("Unexpected error:", sys.exc_info()[0])
        raise

    print('Extracting subjective sentences...')
    trainfile_ = 'train_subjectives.xml'
    testfile_ = 'test_subjectives.xml'   
    extract_subjectives(trainfile, trainfile_)
    leave_outOfScope(testfile, testfile_)
    print('Done!')
		
    # Get the corpus and split into train/test.
    corpus = Corpus(ET.parse(trainfile_).getroot().findall('./Review/sentences/sentence'))
    domain_name = 'restaurants'

    train, seen = corpus.split(threshold=1)
    # Store train/test files and clean up the test files (no aspect terms or categories are present); then, parse back the files back.
    corpus.write_out('%s--train.xml' % domain_name, train, short=False)
    traincorpus = Corpus(ET.parse('%s--train.xml' % domain_name).getroot().findall('sentence'))
    testcorpus = Corpus(ET.parse(testfile_).getroot().findall('./Review/sentences/sentence'))
    corpus.write_out('%s--test.gold.xml' % domain_name, testcorpus.corpus, short=False)
    seen = Corpus(ET.parse('%s--test.gold.xml' % domain_name).getroot().findall('sentence'))

    corpus.write_out('%s--test.xml' % domain_name, seen.corpus)
    unseen = Corpus(ET.parse('%s--test.xml' % domain_name).getroot().findall('sentence'))
                            
    print ('Beginning the ACD task...')
    predicted = classify(traincorpus, unseen)
    print ('P = %f -- R = %f -- F1 = %f (#correct: %d, #retrieved: %d, #relevant: %d)' % Evaluate(seen.corpus, predicted).category_detection())
        
    corpus.write_out('%s--test.predicted-category.xml' % domain_name, predicted, short=False)
    semEval_format(testfile, '%s--test.predicted-category.xml' % domain_name)

    os.remove(testfile_)
    os.remove(trainfile_)

if __name__ == "__main__": main(sys.argv[1:])
