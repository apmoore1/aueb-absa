#!/usr/bin/env python

'''
**Opinion Target Expression for the 5th task of SemEval 2016**
Unconstrained Submission for the Restaurants domain

Run from the terminal:
>>> python ote_unconstrained_restaurants.py --train train.xml --test test.xml
'''

try:
    import xml.etree.ElementTree as ET, getopt, logging, sys, random, re, copy, os
    import numpy as np
    from collections import Counter
    import operator
    from pystruct.models import ChainCRF
    from pystruct.learners import FrankWolfeSSVM
    from sklearn.linear_model import LogisticRegression
    import nltk
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
                for opinions in sentence.findall('Opinions'):
                    for opinion in opinions.findall('Opinion'):
                        aspect_terms.append(opinion.get('target'))
						
    return elements, aspect_terms
	
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
									o.write('\t\t\t\t\t<Opinion target="%s" from="%s" to="%s"/>\n' % (
										fix(opinion.get('target')), opinion.get('from'), opinion.get('to')))
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
							o.write('\t\t\t\t\t<Opinion target="%s" from="%s" to="%s"/>\n' % (
								    fix(opinion.get('target')), opinion.get('from'), opinion.get('to')))
						    o.write('\t\t\t\t</Opinions>\n')
					    o.write('\t\t\t</sentence>\n')
				o.write('\t\t</sentences>\n')
			o.write('\t</Review>\n')
		o.write('</Reviews>')
	
	
fix = lambda text: escape(text.encode('utf8')).replace('\"', '&quot;')
'''Simple fix for writing out text.'''
	
		
class Aspect:
    '''Aspect objects contain the term (e.g., battery life) of an aspect.'''

    def __init__(self, term, offsets):
        self.term = term
        self.offsets = offsets

    def create(self, element):
        self.term = element.attrib['target']
        self.offsets = {'from': str(element.attrib['from']), 'to': str(element.attrib['to'])}
        return self

    def update(self, term=''):
        self.term = term
		
		
class Instance:
    '''An instance is a sentence, modeled out of XML. It contains the text, the aspect terms, and any aspect categories.'''

    def __init__(self, element):
        self.text = element.find('text').text
        self.id = element.get('id')
        self.aspect_terms = [Aspect(term='', offsets={'from': '', 'to': ''}).create(e) for es in element.findall('Opinions') 
							 for e in es if
                                         es is not None]

    def get_aspect_terms(self):
        return [a.term.lower() for a in self.aspect_terms]
		
    def add_aspect_term(self, term, offsets={'from': '', 'to': ''}):
        a = Aspect(term, offsets)
        self.aspect_terms.append(a)


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
                    for a in i.aspect_terms:
                        o.write('\t\t\t<Opinion target="%s" from="%s" to="%s"/>\n' %
                                    (fix(a.term), a.offsets['from'], a.offsets['to']))
                o.write('\t\t</Opinions>\n')
                o.write('\t</sentence>\n')
            o.write('</sentences>')

def load_lexicon(lex_type):
    lex = []

    f = open(lex_type+"_lexicon.txt", "r")
    for line in f:
        tag = line.split()[0]
        lex.append(tag)
        
    return lex

def load_word2vec(path):
    w2v_model = {}
    f = open(path+".txt", "r")
    for line in f:
        vector = []
        fields = line.split()
        name = fields[0]
        for x in fields[1:]:
            vector.append(float(x))
        w2v_model[name] = np.asarray(vector)
        
    return w2v_model

def normalize_horizontal(w2v_vectors):
    '''Normalize the word embeddings horizontally, using the L2-norm.'''
    feature_vectors = []
    norm = np.linalg.norm(w2v_vectors)

    for vec in w2v_vectors:
        feature_vectors.append(vec/norm if norm > 0. else 0.)

    return feature_vectors

# cleaner (order matters)
def clean(text): 
    text = text.lower()
    text = contractions.sub('', text)
    text = singles.sub(' ', text)
    text = seps.sub(' ', text)
    return text
            
def classify(traincorpus, testcorpus):

    model = ChainCRF()
    ssvm = FrankWolfeSSVM(model=model, C=.1, max_iter=10)

    print 'Loading Word2Vec model...'
    w2v_model = load_word2vec("lexica/word_embeds_restaurants_ote")
    print 'Done!'
	
    pos_lexicon = load_lexicon("lexica/restaurants/ote/pos")
    term_lexicon = load_lexicon("lexica/restaurants/ote/term")
    pre1_lexicon = load_lexicon("lexica/restaurants/ote/prefix1")
    pre2_lexicon = load_lexicon("lexica/restaurants/ote/prefix2")
    pre3_lexicon = load_lexicon("lexica/restaurants/ote/prefix3")
    suf1_lexicon = load_lexicon("lexica/restaurants/ote/suffix1")
    suf2_lexicon = load_lexicon("lexica/restaurants/ote/suffix2")
    suf3_lexicon = load_lexicon("lexica/restaurants/ote/suffix3")
    
    train_sentences = [] #the list to be used to store our features for the words    
    sentence_labels = [] #the list to be used for labeling if a word is an aspect term

    print('Creating train feature vectors...')

    #extracting sentences and appending them labels
    for instance in traincorpus.corpus:
        words = nltk.word_tokenize(instance.text)
        
        tags = nltk.pos_tag(words)
        tags_list = [] #the pos list
        for _, t in tags:
                tags_list.append(t)

        last_prediction = ""

        train_words = []
        word_labels = []
        for i, w in enumerate(words):
            word_found = False
            if words[i] == w:
                word_found = True
                
                pos_feats = []
                previous_pos_feats = []
                second_previous_pos_feats = []
                next_pos_feats = []
                second_next_pos_feats = []
                morph_feats = []
                term_feats = []
                pre1_feats = []
                pre2_feats = []
                pre3_feats = []
                suf1_feats = []
                suf2_feats = []
                suf3_feats = []
                vector_feats = []
                previous_vector_feats = []
                second_previous_vector_feats = []
                next_vector_feats = []
                second_next_vector_feats = []
                window_vector_feats = []

                target_labels = []
                train_word_features = []

                #get the embedding vector of the target word and 2 next & previous ones
                #check if the current token is in model
                if words[i].lower() in w2v_model:
                    for vector in w2v_model[words[i].lower()]:
                        vector_feats.append(vector)
                else:
                    counter = 1
                    word_emb_found = False
                    while (i-counter) >= 0 and (word_emb_found is False):
                        if words[i-counter].lower() in w2v_model:
                            for vector in w2v_model[words[i-counter].lower()]:
                                vector_feats.append(vector)
                            word_emb_found = True
                        else:
                            counter = counter + 1
                    if word_emb_found is False:
                        for vector in w2v_model['$start1']:
                            vector_feats.append(vector)

                #check if previous token is in model
                if (i-1) >= 0:
                    if (words[i-1].lower() in w2v_model):
                        for vector in w2v_model[words[i-1].lower()]:
                            previous_vector_feats.append(vector)
                    else:
                        counter = 2 #i-1-1
                        word_emb_found = False
                        while (i-counter) >= 0 and (word_emb_found is False):
                            if words[i-counter].lower() in w2v_model:
                                for vector in w2v_model[words[i-counter].lower()]:
                                    previous_vector_feats.append(vector)
                                word_emb_found = True
                            else:
                                counter = counter + 1
                        if word_emb_found is False:
                            for vector in w2v_model['$start1']:
                                previous_vector_feats.append(vector)
                else:
                    for vector in w2v_model['$start1']:
                        previous_vector_feats.append(vector)

                        
                #check if second previous token is in model
                if (i-2) >= 0:
                    if words[i-2].lower() in w2v_model:
                        for vector in w2v_model[words[i-2].lower()]:
                            second_previous_vector_feats.append(vector)
                    else:
                        counter = 3 #i-2-1
                        word_emb_found = False
                        while (i-counter) >= 0 and (word_emb_found is False):
                            if words[i-counter].lower() in w2v_model:
                                for vector in w2v_model[words[i-counter].lower()]:
                                    second_previous_vector_feats.append(vector)
                                word_emb_found = True
                            else:
                                counter = counter + 1
                        if word_emb_found is False:
                            for vector in w2v_model['$start1']:
                                second_previous_vector_feats.append(vector)
                else:
                    for vector in w2v_model['$start2']:
                        second_previous_vector_feats.append(vector)
                        
                #check if next token is in model
                if (i+1) < len(words):
                    if words[i+1].lower() in w2v_model:
                        for vector in w2v_model[words[i+1].lower()]:
                            next_vector_feats.append(vector)
                    else:
                        counter = 2 #i+1+1
                        word_emb_found = False
                        while (i+counter) < len(words) and (word_emb_found is False):
                            if words[i+counter].lower() in w2v_model:
                                for vector in w2v_model[words[i+counter].lower()]:
                                    next_vector_feats.append(vector)
                                word_emb_found = True
                            else:
                                counter = counter + 1
                        if word_emb_found is False:
                            for vector in w2v_model['$end1']:
                                next_vector_feats.append(vector)
                else:
                    for vector in w2v_model['$end1']:
                        next_vector_feats.append(vector)

                #check if second next token is in model
                if (i+2) < len(words):
                    if words[i+2].lower() in w2v_model:
                        for vector in w2v_model[words[i+2].lower()]:
                            second_next_vector_feats.append(vector)
                    else:
                        counter = 3 #i+2+1
                        word_emb_found = False
                        while (i+counter) < len(words) and (word_emb_found is False):
                            if words[i+counter].lower() in w2v_model:
                                for vector in w2v_model[words[i+counter].lower()]:
                                    second_next_vector_feats.append(vector)
                                word_emb_found = True
                            else:
                                counter = counter + 1
                        if word_emb_found is False:
                            for vector in w2v_model['$end1']:
                                second_next_vector_feats.append(vector)
                else:
                    for vector in w2v_model['$end2']:
                        second_next_vector_feats.append(vector)


                #get the horizontal normalization of the word embeddings
                normalized_vector_feats = normalize_horizontal(vector_feats + previous_vector_feats + second_previous_vector_feats +
                                               next_vector_feats + second_next_vector_feats)

                #prefix 1,2,3 lexicon features
                for p1 in pre1_lexicon:
                    if p1 == w[0]:
                        pre1_feats.append(1)
                    else:
                        pre1_feats.append(0)

                for p2 in pre2_lexicon:
                    if len(w) > 1:
                        if p2 == w[0]+w[1]:
                            pre2_feats.append(1)
                        else:
                            pre2_feats.append(0)
                    else:
                        pre2_feats.append(0)

                for p3 in pre3_lexicon:
                    if len(w) > 2:
                        if p3 == w[0]+w[1]+w[2]:
                            pre3_feats.append(1)
                        else:
                            pre3_feats.append(0)
                    else:
                        pre3_feats.append(0)

                #suffix 1,2,3 lexicon features
                for s1 in suf1_lexicon:
                    if s1 == w[-1]:
                        suf1_feats.append(1)
                    else:
                        suf1_feats.append(0)

                for s2 in suf2_lexicon:
                    if len(w) > 1:
                        if s2 == w[-2]+w[-1]:
                            suf2_feats.append(1)
                        else:
                            suf2_feats.append(0)
                    else:
                        suf2_feats.append(0)

                for s3 in suf3_lexicon:
                    if len(w) > 2:
                        if s3 == w[-3]+w[-2]+w[-1]:
                            suf3_feats.append(1)
                        else:
                            suf3_feats.append(0)
                    else:
                        suf3_feats.append(0)

                #term lexicon features
                for t in term_lexicon:
                    if t == w.lower():
                        term_feats.append(1)
                    else:
                        term_feats.append(0)

                #morphological features
                if w[0].isupper(): #is first letter capital
                    morph_feats.append(1)
                else:
                    morph_feats.append(0)

                capitals = 0
                lowers = 0
                for letter in w:
                    if letter.isupper():
                        capitals = capitals + 1
                    if letter.islower():
                        lowers = lowers + 1

                if w[0].islower() and capitals > 0: #contains capitals, except 1st letter
                    morph_feats.append(1)
                else:
                    morph_feats.append(0)

                if capitals == len(w): #is all letters capitals
                    morph_feats.append(1)
                else:
                    morph_feats.append(0)

                if lowers == len(w): #is all letters lower
                    morph_feats.append(1)
                else:
                    morph_feats.append(0)

                if len(re.findall(r"\d", w)) == len(w): #is all letters digits
                    morph_feats.append(1)
                else:
                    morph_feats.append(0)

                if len(re.findall(r"[a-zA-Z]", w)) == len(w): #is all letters words
                    morph_feats.append(1)
                else:
                    morph_feats.append(0)

                if len(re.findall(r"[.]", w)) > 0: #is there a '.'
                    morph_feats.append(1)
                else:
                    morph_feats.append(0)

                if len(re.findall(r"[-]", w)) > 0: #is there a '-'
                    morph_feats.append(1)
                else:
                    morph_feats.append(0)

                if len(re.findall(r'''[][,;"'?():_`]''', w)) > 0: #is there a punctuation mark, except '.', '-'
                    morph_feats.append(1)
                else:
                    morph_feats.append(0)
                
                for p in pos_lexicon:
                    #check the POS tag of the current word
                    if tags_list[i] == p:
                        pos_feats.append(1)
                    else:
                        pos_feats.append(0)
                            
                    #check the POS tag of the previous word (if the index is IN list's bounds)
                    if (i-1) >= 0:
                        if tags_list[i-1] == p:
                            previous_pos_feats.append(1)
                        else:
                            previous_pos_feats.append(0)
                    else:
                        previous_pos_feats.append(0)
                            
                    #check the POS tag of the 2nd previous word (if the index is IN list's bounds)
                    if (i-2) >= 0:
                        if tags_list[i-2] == p:
                            second_previous_pos_feats.append(1)
                        else:
                            second_previous_pos_feats.append(0)
                    else:
                        second_previous_pos_feats.append(0)
                            
                    #check the POS tag of the next word (if the index is IN list's bounds)
                    if (i+1) < len(words):
                        if tags_list[i+1] == p:
                            next_pos_feats.append(1)
                        else:
                            next_pos_feats.append(0)
                    else:
                        next_pos_feats.append(0)
                            
                    #check the POS tag of the next word (if the index is IN list's bounds)
                    if (i+2) < len(words):
                        if tags_list[i+2] == p:
                            second_next_pos_feats.append(1)
                        else:
                            second_next_pos_feats.append(0)
                    else:
                        second_next_pos_feats.append(0)
                            
		#label the word, using IOB system,
                #B:start of aspect term, I:continue of aspect term, O: no aspect term
                term_found = False                
                for aspect_term in set(instance.get_aspect_terms()):
                    term_words = aspect_term.split()
                    for term_index, term in enumerate(term_words):
                        if (w.lower() == term) and (term_found is False):
                            if term_index == 0:
                                target_labels = [1] #1 is "B"
                                last_prediction = "1"
                                term_found = True                            
                            else:
                                if (last_prediction == "1") or (last_prediction == "2"):
                                    target_labels = [2] #2 is "I"
                                    last_prediction = "2"
                                    term_found = True                            
                                else:
                                    target_labels = [0]
                                    last_prediction = "0"

                if term_found is False:
                    target_labels = [0] #0 is "O"
                    last_prediction = "0"
            
                train_word_features = [pos_feats + previous_pos_feats + second_previous_pos_feats +
                                next_pos_feats + second_next_pos_feats + morph_feats + term_feats +
                                pre1_feats + pre2_feats + pre3_feats + suf1_feats + suf2_feats + suf3_feats +
                                vector_feats + previous_vector_feats + second_previous_vector_feats +
                                               next_vector_feats + second_next_vector_feats]
            if word_found is True:
                train_words.append(train_word_features)
                word_labels.append(target_labels)

        train_sentences_array = np.zeros((len(train_words), len(train_words[0][0])))
        index_i = 0
        for word in train_words:
            index_j = 0
            for features in word:
                for f in features:
                    train_sentences_array[index_i, index_j] = f
                    index_j = index_j + 1
            index_i = index_i + 1
        train_sentences.append(train_sentences_array)        

        sentence_labels_array = np.zeros((len(word_labels)))
        index_i = 0
        for label in word_labels:
            sentence_labels_array[index_i] = label[0]
            index_i = index_i + 1
        sentence_labels.append(sentence_labels_array.astype(np.int64))

    #the chain-crf needs a list (representing the sentences), that
    #contains a 2d-array(n_words, n_features), which in turn contains the
    #features extracted from each word. the sentence labels must be
    #an array of type int
    ssvm.fit(train_sentences, sentence_labels)

    print('Done!')
    print('Creating test feature vectors...')
    
    test_sentences = []
    for instance in testcorpus.corpus:
        words = nltk.word_tokenize(instance.text)
        
        tags = nltk.pos_tag(words)
        tags_list = [] #the pos list
        for _, t in tags:
            tags_list.append(t)

        test_words = []
        for i, w in enumerate(words):
            word_found = False
            if words[i] == w:
                word_found = True
                
                pos_feats = []
                previous_pos_feats = []
                second_previous_pos_feats = []
                next_pos_feats = []
                second_next_pos_feats = []
                morph_feats = []
                term_feats = []
                pre1_feats = []
                pre2_feats = []
                pre3_feats = []
                suf1_feats = []
                suf2_feats = []
                suf3_feats = []
                vector_feats = []
                previous_vector_feats = []
                second_previous_vector_feats = []
                next_vector_feats = []
                second_next_vector_feats = []
                window_vector_feats = []

                test_word_features = []

                #get the embedding vector of the target word and 2 next & previous ones
                #check if the current token is in model
                if words[i].lower() in w2v_model:
                    for vector in w2v_model[words[i].lower()]:
                        vector_feats.append(vector)
                else:
                    counter = 1
                    word_emb_found = False
                    while (i-counter) >= 0 and (word_emb_found is False):
                        if words[i-counter].lower() in w2v_model:
                            for vector in w2v_model[words[i-counter].lower()]:
                                vector_feats.append(vector)
                            word_emb_found = True
                        else:
                            counter = counter + 1
                    if word_emb_found is False:
                        for vector in w2v_model['$start1']:
                            vector_feats.append(vector)

                #check if previous token is in model
                if (i-1) >= 0:
                    if (words[i-1].lower() in w2v_model):
                        for vector in w2v_model[words[i-1].lower()]:
                            previous_vector_feats.append(vector)
                    else:
                        counter = 2 #i-1-1
                        word_emb_found = False
                        while (i-counter) >= 0 and (word_emb_found is False):
                            if words[i-counter].lower() in w2v_model:
                                for vector in w2v_model[words[i-counter].lower()]:
                                    previous_vector_feats.append(vector)
                                word_emb_found = True
                            else:
                                counter = counter + 1
                        if word_emb_found is False:
                            for vector in w2v_model['$start1']:
                                previous_vector_feats.append(vector)
                else:
                    for vector in w2v_model['$start1']:
                        previous_vector_feats.append(vector)

                        
                #check if second previous token is in model
                if (i-2) >= 0:
                    if words[i-2].lower() in w2v_model:
                        for vector in w2v_model[words[i-2].lower()]:
                            second_previous_vector_feats.append(vector)
                    else:
                        counter = 3 #i-2-1
                        word_emb_found = False
                        while (i-counter) >= 0 and (word_emb_found is False):
                            if words[i-counter].lower() in w2v_model:
                                for vector in w2v_model[words[i-counter].lower()]:
                                    second_previous_vector_feats.append(vector)
                                word_emb_found = True
                            else:
                                counter = counter + 1
                        if word_emb_found is False:
                            for vector in w2v_model['$start1']:
                                second_previous_vector_feats.append(vector)
                else:
                    for vector in w2v_model['$start2']:
                        second_previous_vector_feats.append(vector)
                        
                #check if next token is in model
                if (i+1) < len(words):
                    if words[i+1].lower() in w2v_model:
                        for vector in w2v_model[words[i+1].lower()]:
                            next_vector_feats.append(vector)
                    else:
                        counter = 2 #i+1+1
                        word_emb_found = False
                        while (i+counter) < len(words) and (word_emb_found is False):
                            if words[i+counter].lower() in w2v_model:
                                for vector in w2v_model[words[i+counter].lower()]:
                                    next_vector_feats.append(vector)
                                word_emb_found = True
                            else:
                                counter = counter + 1
                        if word_emb_found is False:
                            for vector in w2v_model['$end1']:
                                next_vector_feats.append(vector)
                else:
                    for vector in w2v_model['$end1']:
                        next_vector_feats.append(vector)

                #check if second next token is in model
                if (i+2) < len(words):
                    if words[i+2].lower() in w2v_model:
                        for vector in w2v_model[words[i+2].lower()]:
                            second_next_vector_feats.append(vector)
                    else:
                        counter = 3 #i+2+1
                        word_emb_found = False
                        while (i+counter) < len(words) and (word_emb_found is False):
                            if words[i+counter].lower() in w2v_model:
                                for vector in w2v_model[words[i+counter].lower()]:
                                    second_next_vector_feats.append(vector)
                                word_emb_found = True
                            else:
                                counter = counter + 1
                        if word_emb_found is False:
                            for vector in w2v_model['$end1']:
                                second_next_vector_feats.append(vector)
                else:
                    for vector in w2v_model['$end2']:
                        second_next_vector_feats.append(vector)

                #get the horizontal normalization of the word embeddings
                normalized_vector_feats = normalize_horizontal(vector_feats + previous_vector_feats + second_previous_vector_feats +
                                               next_vector_feats + second_next_vector_feats)

                #prefix 1,2,3 lexicon features
                for p1 in pre1_lexicon:
                    if p1 == w[0]:
                        pre1_feats.append(1)
                    else:
                        pre1_feats.append(0)

                for p2 in pre2_lexicon:
                    if len(w) > 1:
                        if p2 == w[0]+w[1]:
                            pre2_feats.append(1)
                        else:
                            pre2_feats.append(0)
                    else:
                        pre2_feats.append(0)

                for p3 in pre3_lexicon:
                    if len(w) > 2:
                        if p3 == w[0]+w[1]+w[2]:
                            pre3_feats.append(1)
                        else:
                            pre3_feats.append(0)
                    else:
                        pre3_feats.append(0)

                #suffix 1,2,3 lexicon features
                for s1 in suf1_lexicon:
                    if s1 == w[-1]:
                        suf1_feats.append(1)
                    else:
                        suf1_feats.append(0)

                for s2 in suf2_lexicon:
                    if len(w) > 1:
                        if s2 == w[-2]+w[-1]:
                            suf2_feats.append(1)
                        else:
                            suf2_feats.append(0)
                    else:
                        suf2_feats.append(0)

                for s3 in suf3_lexicon:
                    if len(w) > 2:
                        if s3 == w[-3]+w[-2]+w[-1]:
                            suf3_feats.append(1)
                        else:
                            suf3_feats.append(0)
                    else:
                        suf3_feats.append(0)

                #term lexicon features
                for t in term_lexicon:
                    if t == w.lower():
                        term_feats.append(1)
                    else:
                        term_feats.append(0)

                #morphological features
                if w[0].isupper(): #is first letter capital
                    morph_feats.append(1)
                else:
                    morph_feats.append(0)

                capitals = 0
                lowers = 0
                for letter in w:
                    if letter.isupper():
                        capitals = capitals + 1
                    if letter.islower():
                        lowers = lowers + 1

                if w[0].islower() and capitals > 0: #contains capitals, except 1st letter
                    morph_feats.append(1)
                else:
                    morph_feats.append(0)

                if capitals == len(w): #is all letters capitals
                    morph_feats.append(1)
                else:
                    morph_feats.append(0)

                if lowers == len(w): #is all letters lower
                    morph_feats.append(1)
                else:
                    morph_feats.append(0)

                if len(re.findall(r"\d", w)) == len(w): #is all letters digits
                    morph_feats.append(1)
                else:
                    morph_feats.append(0)

                if len(re.findall(r"[a-zA-Z]", w)) == len(w): #is all letters words
                    morph_feats.append(1)
                else:
                    morph_feats.append(0)

                if len(re.findall(r"[.]", w)) > 0: #is there a '.'
                    morph_feats.append(1)
                else:
                    morph_feats.append(0)

                if len(re.findall(r"[-]", w)) > 0: #is there a '-'
                    morph_feats.append(1)
                else:
                    morph_feats.append(0)

                if len(re.findall(r'''[][,;"'?():_`]''', w)) > 0: #is there a punctuation mark, except '.', '-'
                    morph_feats.append(1)
                else:
                    morph_feats.append(0)
                
                for p in pos_lexicon:
                    #check the POS tag of the current word
                    if tags_list[i] == p:
                        pos_feats.append(1)
                    else:
                        pos_feats.append(0)
                            
                    #check the POS tag of the previous word (if the index is IN list's bounds)
                    if (i-1) >= 0:
                        if tags_list[i-1] == p:
                            previous_pos_feats.append(1)
                        else:
                            previous_pos_feats.append(0)
                    else:
                        previous_pos_feats.append(0)
                            
                    #check the POS tag of the 2nd previous word (if the index is IN list's bounds)
                    if (i-2) >= 0:
                        if tags_list[i-2] == p:
                            second_previous_pos_feats.append(1)
                        else:
                            second_previous_pos_feats.append(0)
                    else:
                        second_previous_pos_feats.append(0)
                            
                    #check the POS tag of the next word (if the index is IN list's bounds)
                    if (i+1) < len(words):
                        if tags_list[i+1] == p:
                            next_pos_feats.append(1)
                        else:
                            next_pos_feats.append(0)
                    else:
                        next_pos_feats.append(0)
                            
                    #check the POS tag of the next word (if the index is IN list's bounds)
                    if (i+2) < len(words):
                        if tags_list[i+2] == p:
                            second_next_pos_feats.append(1)
                        else:
                            second_next_pos_feats.append(0)
                    else:
                        second_next_pos_feats.append(0)
            
                test_word_features = [pos_feats + previous_pos_feats + second_previous_pos_feats +
                                next_pos_feats + second_next_pos_feats + morph_feats + term_feats +
                                pre1_feats + pre2_feats + pre3_feats + suf1_feats + suf2_feats + suf3_feats +
                                vector_feats + previous_vector_feats + second_previous_vector_feats +
                                               next_vector_feats + second_next_vector_feats]
            if word_found is True:
                test_words.append(test_word_features)

        test_sentences_array = np.zeros((len(test_words), len(test_words[0][0])))
        index_i = 0
        for word in test_words:
            index_j = 0
            for features in word:
                for f in features:
                    test_sentences_array[index_i, index_j] = f
                    index_j = index_j + 1
            index_i = index_i + 1
        test_sentences.append(test_sentences_array)

    print('Done!')
    print('Predicting aspect terms...')

    predictions = ssvm.predict(test_sentences)
    #the predict function returns a list (symbolizing the sentences),
    #which contains a list that contains the predicted label for each word
    for sentence_index, sentence_predictions in enumerate(predictions):
            testcorpus.corpus[sentence_index].aspect_terms = []

            predicted_term = ""
            last_prediction = ""
            for word_index, word_prediction in enumerate(sentence_predictions):
                if word_prediction == 1:
                    if last_prediction == 1 or last_prediction == 2:
                        start, end = find_offsets(testcorpus.corpus[sentence_index].text.lower(), predicted_term)
                        testcorpus.corpus[sentence_index].add_aspect_term(term=predicted_term, offsets={'from': str(start), 'to': str(end)})
                        
                    c = find_term(testcorpus.corpus[sentence_index].text.lower(), word_index)
                    predicted_term = c
                    last_prediction = 1
                    
                elif word_prediction == 2:
                    if last_prediction == 1 or last_prediction == 2:
                        c = find_term(testcorpus.corpus[sentence_index].text.lower(), word_index)
                        if len(predicted_term) > 0:
                            predicted_term = predicted_term + " " + c
                        else:
                            predicted_term = c
                    last_prediction = 2

                elif word_prediction == 0:
                    if last_prediction == 1 or last_prediction == 2:
                        start, end = find_offsets(testcorpus.corpus[sentence_index].text.lower(), predicted_term)
                        testcorpus.corpus[sentence_index].add_aspect_term(term=predicted_term, offsets={'from': str(start), 'to': str(end)})
                    last_prediction = 0
                            
    print('Done!')
    return testcorpus.corpus


def find_term(sentence, word_index):
    '''Find the term of the sentence, gives it's index.'''
    words = nltk.word_tokenize(sentence)
    term = words[word_index]
    
    return term 

def find_offsets(sentence, word):
    '''Find the offsets of a word in a sentence.'''
    start = sentence.find(word)
    
    if start == -1:
        start = 0
        end = 0
    else:
        end = start + len(word)
        
    return start, end

    
class Evaluate():

    def __init__(self, correct, predicted):
        self.size = len(correct)
        self.correct = correct
        self.predicted = predicted

    # Aspect Extraction (no offsets considered)
    def aspect_extraction(self, b=1):
        common, relevant, retrieved = 0., 0., 0.
        for i in range(self.size):
            cor = []
            for a in set(self.correct[i].aspect_terms):
                if (a.term.lower()!="null") and (a.offsets not in cor):
                    cor.append(a.offsets)

            pre = []        
            for a in set(self.predicted[i].aspect_terms):
                if a.offsets not in pre:
                   pre.append(a.offsets)
                   
            common += len([a for a in pre if a in cor])
            retrieved += len(pre)
            relevant += len(cor)
        p = common / retrieved if retrieved > 0 else 0.
        r = common / relevant
        f1 = (1 + (b ** 2)) * p * r / ((p * b ** 2) + r) if p > 0 and r > 0 else 0.
        return p, r, f1, common, retrieved, relevant

def semEval_format(testWithReviews, testWithoutReviewsOTE):

    tree1 = ET.parse(testWithReviews)
    root1 = tree1.getroot()

    tree2 = ET.parse(testWithoutReviewsOTE)
    root2 = tree2.getroot()

    with open('ote_unc_rest_semeval.xml', 'w') as o:
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
                        for sentence2 in root2.findall('sentence'):
                            if sentence1.get('id') == sentence2.get('id'):
                                o.write('\t\t\t<sentence id="%s">\n' % (sentence2.get('id')))
                                o.write('\t\t\t\t<text>%s</text>\n' % (fix(sentence2.find('text').text)))
                                for opinions in sentence2.findall('Opinions'):
                                    if opinions.find('Opinion') != None:
                                        bug_flag = False
                                        for opinion in opinions.findall('Opinion'):
                                            if opinion.get('to') == "0":
                                                bug_flag = True #bug flag. if a term is given 0 in the offsets, we discard it. has to do with the ASCII format
                                        if bug_flag is False:
                                            o.write('\t\t\t\t<Opinions>\n')
                                            for opinion in opinions.findall('Opinion'):
                                                o.write('\t\t\t\t\t<Opinion target="%s" category="" from="%s" to="%s"/>\n' % (
                                                        fix(opinion.get('target')), opinion.get('from'), opinion.get('to')))
                                            o.write('\t\t\t\t</Opinions>\n')
                                o.write('\t\t\t</sentence>\n')
                o.write('\t\t</sentences>\n')
            o.write('\t</Review>\n')
        o.write('</Reviews>')

			
def main(argv=None):
    # Parse the input
    opts, args = getopt.getopt(argv, "hg:dt:k:", ["help", "grammar", "train=", "test="])
    trainfile, testfile = None, None
    use_msg = 'Use as:\n">>> python ote_unconstrained_restaurants.py --train train.xml --test test.xml"\n\nThis will parse a train set, examine whether is valid, perform target extraction on the test set provided, and write out a file with the predictions.'
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
        elements, aspects = validate(trainfile)
        print ('PASSED! This corpus has: %d sentences, %d aspect term occurrences, and %d distinct aspect terms.' % (
            len(elements), len(aspects), len(list(set(aspects)))))
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
             
    print ('Beginning the OTE task...')
    predicted = classify(traincorpus, unseen)
    corpus.write_out('%s--test.predicted-aspect.xml' % domain_name, predicted, short=False)
    print ('P = %f -- R = %f -- F1 = %f (#correct: %d, #retrieved: %d, #relevant: %d)' % Evaluate(seen.corpus, predicted).aspect_extraction())
        
    semEval_format(testfile, '%s--test.predicted-aspect.xml' % domain_name)

    os.remove(testfile_)
    os.remove(trainfile_)

if __name__ == "__main__": main(sys.argv[1:])
