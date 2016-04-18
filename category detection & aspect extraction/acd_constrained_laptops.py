#!/usr/bin/env python

'''
**Aspect Category Detection for the 5th task of SemEval 2016**
Constrained Submission for the Laptops domain

Run from the terminal:
>>> python acd_constrained_laptops.py --train train.xml --test test.xml
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
	'''Leave out the sentences with the attribute 'OutOfScope=="TRUE".'''
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

    #entity lexica
    laptop = []
    display = []
    cpu = []
    mb = []
    hd = []
    mem = []
    bat = []
    power = []
    keyb = []
    mouse = []
    fans = []
    opt = []
    ports = []
    graph = []
    mm = []
    hw = []
    os = []
    sw = []
    warranty = []
    ship = []
    supp = []
    comp = []

    #attribute lexica
    gen = []
    pr = []
    qual = []
    oper = []
    use = []
    des = []
    port = []
    conn = []
    misc = []

    f = open(lex_type+"_laptop_lexicon.txt", "r")
    for line in f:
        feats = line.split()
        if b is True:
            feats[0] = tuple(feats[0].split(','))
        laptop.append([feats[0], float(feats[1]), float(feats[2]), float(feats[3]), float(feats[4])])
    
    f = open(lex_type+"_display_lexicon.txt", "r")
    for line in f:
        feats = line.split()
        if b is True:
            feats[0] = tuple(feats[0].split(','))
        display.append([feats[0], float(feats[1]), float(feats[2]), float(feats[3]), float(feats[4])])

    f = open(lex_type+"_cpu_lexicon.txt", "r")
    for line in f:
        feats = line.split()
        if b is True:
            feats[0] = tuple(feats[0].split(','))
        cpu.append([feats[0], float(feats[1]), float(feats[2]), float(feats[3]), float(feats[4])])

    f = open(lex_type+"_mb_lexicon.txt", "r")
    for line in f:
        feats = line.split()
        if b is True:
            feats[0] = tuple(feats[0].split(','))
        mb.append([feats[0], float(feats[1]), float(feats[2]), float(feats[3]), float(feats[4])])

    f = open(lex_type+"_hd_lexicon.txt", "r")
    for line in f:
        feats = line.split()
        if b is True:
            feats[0] = tuple(feats[0].split(','))
        hd.append([feats[0], float(feats[1]), float(feats[2]), float(feats[3]), float(feats[4])])

    f = open(lex_type+"_mem_lexicon.txt", "r")
    for line in f:
        feats = line.split()
        if b is True:
            feats[0] = tuple(feats[0].split(','))
        mem.append([feats[0], float(feats[1]), float(feats[2]), float(feats[3]), float(feats[4])])

    f = open(lex_type+"_bat_lexicon.txt", "r")
    for line in f:
        feats = line.split()
        if b is True:
            feats[0] = tuple(feats[0].split(','))
        bat.append([feats[0], float(feats[1]), float(feats[2]), float(feats[3]), float(feats[4])])

    f = open(lex_type+"_power_lexicon.txt", "r")
    for line in f:
        feats = line.split()
        if b is True:
            feats[0] = tuple(feats[0].split(','))
        power.append([feats[0], float(feats[1]), float(feats[2]), float(feats[3]), float(feats[4])])

    f = open(lex_type+"_keyb_lexicon.txt", "r")
    for line in f:
        feats = line.split()
        if b is True:
            feats[0] = tuple(feats[0].split(','))
        keyb.append([feats[0], float(feats[1]), float(feats[2]), float(feats[3]), float(feats[4])])

    f = open(lex_type+"_mouse_lexicon.txt", "r")
    for line in f:
        feats = line.split()
        if b is True:
            feats[0] = tuple(feats[0].split(','))
        mouse.append([feats[0], float(feats[1]), float(feats[2]), float(feats[3]), float(feats[4])])

    f = open(lex_type+"_fans_lexicon.txt", "r")
    for line in f:
        feats = line.split()
        if b is True:
            feats[0] = tuple(feats[0].split(','))
        fans.append([feats[0], float(feats[1]), float(feats[2]), float(feats[3]), float(feats[4])])

    f = open(lex_type+"_opt_lexicon.txt", "r")
    for line in f:
        feats = line.split()
        if b is True:
            feats[0] = tuple(feats[0].split(','))
        opt.append([feats[0], float(feats[1]), float(feats[2]), float(feats[3]), float(feats[4])])

    f = open(lex_type+"_ports_lexicon.txt", "r")
    for line in f:
        feats = line.split()
        if b is True:
            feats[0] = tuple(feats[0].split(','))
        ports.append([feats[0], float(feats[1]), float(feats[2]), float(feats[3]), float(feats[4])])

    f = open(lex_type+"_graphs_lexicon.txt", "r")
    for line in f:
        feats = line.split()
        if b is True:
            feats[0] = tuple(feats[0].split(','))
        graph.append([feats[0], float(feats[1]), float(feats[2]), float(feats[3]), float(feats[4])])

    f = open(lex_type+"_mm_lexicon.txt", "r")
    for line in f:
        feats = line.split()
        if b is True:
            feats[0] = tuple(feats[0].split(','))
        mm.append([feats[0], float(feats[1]), float(feats[2]), float(feats[3]), float(feats[4])])

    f = open(lex_type+"_hw_lexicon.txt", "r")
    for line in f:
        feats = line.split()
        if b is True:
            feats[0] = tuple(feats[0].split(','))
        hw.append([feats[0], float(feats[1]), float(feats[2]), float(feats[3]), float(feats[4])])

    f = open(lex_type+"_os_lexicon.txt", "r")
    for line in f:
        feats = line.split()
        if b is True:
            feats[0] = tuple(feats[0].split(','))
        os.append([feats[0], float(feats[1]), float(feats[2]), float(feats[3]), float(feats[4])])

    f = open(lex_type+"_sw_lexicon.txt", "r")
    for line in f:
        feats = line.split()
        if b is True:
            feats[0] = tuple(feats[0].split(','))
        sw.append([feats[0], float(feats[1]), float(feats[2]), float(feats[3]), float(feats[4])])

    f = open(lex_type+"_warranty_lexicon.txt", "r")
    for line in f:
        feats = line.split()
        if b is True:
            feats[0] = tuple(feats[0].split(','))
        warranty.append([feats[0], float(feats[1]), float(feats[2]), float(feats[3]), float(feats[4])])

    f = open(lex_type+"_ship_lexicon.txt", "r")
    for line in f:
        feats = line.split()
        if b is True:
            feats[0] = tuple(feats[0].split(','))
        ship.append([feats[0], float(feats[1]), float(feats[2]), float(feats[3]), float(feats[4])])

    f = open(lex_type+"_supp_lexicon.txt", "r")
    for line in f:
        feats = line.split()
        if b is True:
            feats[0] = tuple(feats[0].split(','))
        supp.append([feats[0], float(feats[1]), float(feats[2]), float(feats[3]), float(feats[4])])

    f = open(lex_type+"_comp_lexicon.txt", "r")
    for line in f:
        feats = line.split()
        if b is True:
            feats[0] = tuple(feats[0].split(','))
        comp.append([feats[0], float(feats[1]), float(feats[2]), float(feats[3]), float(feats[4])])

    f = open(lex_type+"_gen_lexicon.txt", "r")
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

    f = open(lex_type+"_qual_lexicon.txt", "r")
    for line in f:
        feats = line.split()
        if b is True:
            feats[0] = tuple(feats[0].split(','))
        qual.append([feats[0], float(feats[1]), float(feats[2]), float(feats[3]), float(feats[4])])

    f = open(lex_type+"_oper_lexicon.txt", "r")
    for line in f:
        feats = line.split()
        if b is True:
            feats[0] = tuple(feats[0].split(','))
        oper.append([feats[0], float(feats[1]), float(feats[2]), float(feats[3]), float(feats[4])])

    f = open(lex_type+"_use_lexicon.txt", "r")
    for line in f:
        feats = line.split()
        if b is True:
            feats[0] = tuple(feats[0].split(','))
        use.append([feats[0], float(feats[1]), float(feats[2]), float(feats[3]), float(feats[4])])

    f = open(lex_type+"_des_lexicon.txt", "r")
    for line in f:
        feats = line.split()
        if b is True:
            feats[0] = tuple(feats[0].split(','))
        des.append([feats[0], float(feats[1]), float(feats[2]), float(feats[3]), float(feats[4])])

    f = open(lex_type+"_port_lexicon.txt", "r")
    for line in f:
        feats = line.split()
        if b is True:
            feats[0] = tuple(feats[0].split(','))
        port.append([feats[0], float(feats[1]), float(feats[2]), float(feats[3]), float(feats[4])])

    f = open(lex_type+"_conn_lexicon.txt", "r")
    for line in f:
       feats = line.split()
       if b is True:
           feats[0] = tuple(feats[0].split(','))
       conn.append([feats[0], float(feats[1]), float(feats[2]), float(feats[3]), float(feats[4])])

    f = open(lex_type+"_misc_lexicon.txt", "r")
    for line in f:
        feats = line.split()
        if b is True:
            feats[0] = tuple(feats[0].split(','))
        misc.append([feats[0], float(feats[1]), float(feats[2]), float(feats[3]), float(feats[4])])
    
    f.close()
    return [laptop, display, cpu, mb, hd, mem, bat, power, keyb, mouse, fans, opt, ports, graph, mm, hw, os, sw, warranty, ship, supp, comp, gen, pr, qual, oper, use, des, port, conn, misc]


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
    '''An instance is a sentence, modeled out of XML. It contains the text, and any aspect categories.'''

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

    def split(self, threshold=1, shuffle=False):
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
    laptop_clf1 = SVC(kernel='rbf', C=5.656854249492381, gamma=0.02209708691207961, probability=True)
    display_clf1 = SVC(kernel='rbf', C=2.378414230005442, gamma=0.02209708691207961, probability=True)
    cpu_clf1 = SVC(kernel='sigmoid', C=64, gamma=0.04419417382415922, probability=True)
    mb_clf1 = SVC(kernel='linear', C=0.5946035575013605, probability=True)
    hd_clf1 = SVC(kernel='rbf', C=64.0, gamma=0.0078125, probability=True)
    memory_clf1 = SVC(kernel='sigmoid', C=53.81737057623773, gamma=0.03125, probability=True)
    battery_clf1 = SVC(kernel='sigmoid', C=22.627416997969522, gamma=0.04419417382415922, probability=True)
    power_clf1 = SVC(kernel='sigmoid', C=107.63474115247546, gamma=0.018581361171917516, probability=True)
    keyboard_clf1 = SVC(kernel='poly', C=0.0011613350732448448, gamma=1.189207115002721, probability=True)
    mouse_clf1 = SVC(kernel='rbf', C=5.656854249492381, gamma=0.0078125, probability=True)
    fans_clf1 = SVC(kernel='sigmoid', C=13.454342644059432, gamma=0.03125, probability=True)
    opt_drives_clf1 = SVC(kernel='sigmoid', C=1024, gamma=0.052556025953357156, probability=True)
    ports_clf1 = SVC(kernel='linear', C=1.189207115002721, probability=True)
    graphics_clf1 = SVC(kernel='sigmoid', C=128.0, gamma=0.03125, probability=True)
    mm_devs_clf1 = SVC(kernel='sigmoid', C=304.4370214406966, gamma=0.02209708691207961, probability=True)
    hardw_clf1 = SVC(kernel='sigmoid', C=32, gamma=0.03716272234383503, probability=True)
    os_clf1 = SVC(kernel='poly', C=0.000244140625, gamma=4.0, probability=True)
    softw_clf1 = SVC(kernel='sigmoid', C=6.727171322029716, gamma=0.03716272234383503, probability=True)
    warranty_clf1 = SVC(kernel='sigmoid', C=107.63474115247546, gamma=0.07432544468767006, probability=True)
    shipping_clf1 = SVC(kernel='linear', C=1.0, probability=True)
    support_clf1 = SVC(kernel='rbf', C=38.05462768008707, gamma=0.011048543456039806, probability=True)
    company_clf1 = SVC(kernel='rbf', C=26.908685288118864, gamma=0.02209708691207961, probability=True)
    general_clf1 = SVC(kernel='rbf', C=3.363585661014858, gamma=0.0625, probability=True)
    price_clf1 = SVC(kernel='sigmoid', C=6.727171322029716, gamma=0.03716272234383503, probability=True)
    quality_clf1 = SVC(kernel='sigmoid', C=4.0, gamma=0.026278012976678578, probability=True)
    op_perf_clf1 = SVC(kernel='rbf', C=4.0, gamma=0.125, probability=True)
    usability_clf1 = SVC(kernel='rbf', C=13.454342644059432, gamma=0.018581361171917516, probability=True)
    des_feats_clf1 = SVC(kernel='rbf', C=2, gamma=0.10511205190671431, probability=True)
    portability_clf1 = SVC(kernel='rbf', C=2.378414230005442, gamma=0.10511205190671431, probability=True)
    connectivity_clf1 = SVC(kernel='sigmoid', C=22.627416997969522, gamma=0.052556025953357156, probability=True)
    misc_clf1 = SVC(kernel='rbf', C=1.4142135623730951, gamma=0.10511205190671431, probability=True)

    stemmer = PorterStemmer()
    
    unigrams_lexica = load_lexicon("lexica/laptops/unigrams", False)
    bigrams_lexica = load_lexicon("lexica/laptops/bigram", True)
    bipos_lexica = load_lexicon("lexica/laptops/bipos", True)
    stemmed_unigrams_lexica = load_lexicon("lexica/laptops/stemmed_unigrams", False)
    stemmed_bigrams_lexica = load_lexicon("lexica/laptops/stemmed_bigrams", True)
    
    train_sentences1 = [] #the sentences to be used for feature extraction

    #entity labels
    laptop_labels = []
    display_labels = []
    cpu_labels = []
    mb_labels = []
    hd_labels = []
    memory_labels = []
    battery_labels = []
    power_labels = []
    keyboard_labels = []
    mouse_labels = []
    fans_labels = []
    opt_drives_labels = []
    ports_labels = []
    graphics_labels = []
    mm_devs_labels = []
    hardw_labels = []
    os_labels = []
    softw_labels = []
    warranty_labels = []
    shipping_labels = []
    support_labels = []
    company_labels = []

    #attribute labels
    general_labels = []
    price_labels = []
    quality_labels = []
    op_perf_labels = []
    usability_labels = []
    des_feats_labels = []
    portability_labels = []
    connectivity_labels = []
    misc_labels = []

    #number of categories
    cats = []

    print('Creating train feature vectors...')

    #extracting words from sentences and appending them labels
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
        #categories like laptop#quality and laptop#price assigned to a sentence
        ent_set = set()
        attr_set = set()
        for c in instance.get_aspect_categories():
            ent_attr = c.split('#')
            ent_set.add(ent_attr[0]) #store the entity
            attr_set.add(ent_attr[1]) #store the attribute
            cats.append(c)
            
        #check entity category
        if "laptop" in ent_set:
            laptop_labels.append(1)
        else:
            laptop_labels.append(0)

        if "display" in ent_set:
            display_labels.append(1)
        else:
            display_labels.append(0)
            
        if "cpu" in ent_set:
            cpu_labels.append(1)
        else:
            cpu_labels.append(0)
            
        if "motherboard" in ent_set:
            mb_labels.append(1)
        else:
            mb_labels.append(0)
            
        if "hard_disc" in ent_set:
            hd_labels.append(1)
        else:
            hd_labels.append(0)
            
        if "memory" in ent_set:
            memory_labels.append(1)
        else:
            memory_labels.append(0)
            
        if "battery" in ent_set:
            battery_labels.append(1)
        else:
            battery_labels.append(0)
            
        if "power_supply" in ent_set:
            power_labels.append(1)
        else:
            power_labels.append(0)
            
        if "keyboard" in ent_set:
            keyboard_labels.append(1)
        else:
            keyboard_labels.append(0)
            
        if "mouse" in ent_set:
            mouse_labels.append(1)
        else:
            mouse_labels.append(0)
            
        if "fans_cooling" in ent_set:
            fans_labels.append(1)
        else:
            fans_labels.append(0)
            
        if "optical_drives" in ent_set:
            opt_drives_labels.append(1)
        else:
            opt_drives_labels.append(0)
            
        if "ports" in ent_set:
            ports_labels.append(1)
        else:
            ports_labels.append(0)
            
        if "graphics" in ent_set:
            graphics_labels.append(1)
        else:
            graphics_labels.append(0)
            
        if "multimedia_devices" in ent_set:
            mm_devs_labels.append(1)
        else:
            mm_devs_labels.append(0)
            
        if "hardware" in ent_set:
            hardw_labels.append(1)
        else:
            hardw_labels.append(0)
            
        if "os" in ent_set:
            os_labels.append(1)
        else:
            os_labels.append(0)
            
        if "software" in ent_set:
            softw_labels.append(1)
        else:
            softw_labels.append(0)
            
        if "warranty" in ent_set:
            warranty_labels.append(1)
        else:
            warranty_labels.append(0)
            
        if "shipping" in ent_set:
            shipping_labels.append(1)
        else:
            shipping_labels.append(0)
            
        if "support" in ent_set:
            support_labels.append(1)
        else:
            support_labels.append(0)
            
        if "company" in ent_set:
            company_labels.append(1)
        else:
            company_labels.append(0)

        #check attribute category
        if "general" in attr_set:
            general_labels.append(1)
        else:
            general_labels.append(0)
            
        if "price" in attr_set: 
            price_labels.append(1)
        else:
            price_labels.append(0)
            
        if "quality" in attr_set: 
            quality_labels.append(1)
        else:
            quality_labels.append(0)
            
        if "operation_performance" in attr_set: 
            op_perf_labels.append(1)
        else:
            op_perf_labels.append(0)
            
        if "usability" in attr_set: 
            usability_labels.append(1)
        else:
            usability_labels.append(0)
            
        if "design_features" in attr_set: 
            des_feats_labels.append(1)
        else:
            des_feats_labels.append(0)
            
        if "portability" in attr_set: 
            portability_labels.append(1)
        else:
            portability_labels.append(0)
            
        if "connectivity" in attr_set: 
            connectivity_labels.append(1)
        else:
            connectivity_labels.append(0)
            
        if "miscellaneous" in attr_set: 
            misc_labels.append(1)
        else:
            misc_labels.append(0)

    cat_dict = Counter(cats)

    train_features1 = np.asarray(train_sentences1)
    
    laptop_clf1.fit(train_features1, laptop_labels)
    display_clf1.fit(train_features1, display_labels)
    cpu_clf1.fit(train_features1, cpu_labels)
    mb_clf1.fit(train_features1, mb_labels)
    hd_clf1.fit(train_features1, hd_labels)
    memory_clf1.fit(train_features1, memory_labels)
    battery_clf1.fit(train_features1, battery_labels)
    power_clf1.fit(train_features1, power_labels)
    keyboard_clf1.fit(train_features1, keyboard_labels)
    mouse_clf1.fit(train_features1, mouse_labels)
    fans_clf1.fit(train_features1, fans_labels)
    opt_drives_clf1.fit(train_features1, opt_drives_labels)
    ports_clf1.fit(train_features1, ports_labels)
    graphics_clf1.fit(train_features1, graphics_labels)
    mm_devs_clf1.fit(train_features1, mm_devs_labels)
    hardw_clf1.fit(train_features1, hardw_labels)
    os_clf1.fit(train_features1, os_labels)
    softw_clf1.fit(train_features1, softw_labels)
    warranty_clf1.fit(train_features1, warranty_labels)
    shipping_clf1.fit(train_features1, shipping_labels)
    support_clf1.fit(train_features1, support_labels)
    company_clf1.fit(train_features1, company_labels)    
    general_clf1.fit(train_features1, general_labels)
    price_clf1.fit(train_features1, price_labels)
    quality_clf1.fit(train_features1, quality_labels)
    op_perf_clf1.fit(train_features1, op_perf_labels)
    usability_clf1.fit(train_features1, usability_labels)
    des_feats_clf1.fit(train_features1, des_feats_labels)
    portability_clf1.fit(train_features1, portability_labels)
    connectivity_clf1.fit(train_features1, connectivity_labels)
    misc_clf1.fit(train_features1, misc_labels)

    print('Done!')
    print('Creating test feature vectors...')
    
    test_sentences1 = []
    test_sentences2 = []
    for instance in testcorpus.corpus:
        words = (re.findall(r"[\w']+", instance.text.lower())) #the unigrams list

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
    aspect_categories = []

    print('Done!')
    print('Predicting categories...')

    for i, test_fvector in enumerate(test_features1):     
            laptop_pred1 = laptop_clf1.predict_proba(test_fvector.reshape(1,-1))[0,1]
            display_pred1 = display_clf1.predict_proba(test_fvector.reshape(1,-1))[0,1]
            cpu_pred1 = cpu_clf1.predict_proba(test_fvector.reshape(1,-1))[0,1]
            mb_pred1 = mb_clf1.predict_proba(test_fvector.reshape(1,-1))[0,1]
            hd_pred1 = hd_clf1.predict_proba(test_fvector.reshape(1,-1))[0,1]
            memory_pred1 = memory_clf1.predict_proba(test_fvector.reshape(1,-1))[0,1]
            battery_pred1 = battery_clf1.predict_proba(test_fvector.reshape(1,-1))[0,1]
            power_pred1 = power_clf1.predict_proba(test_fvector.reshape(1,-1))[0,1]
            keyboard_pred1 = keyboard_clf1.predict_proba(test_fvector.reshape(1,-1))[0,1]
            mouse_pred1 = mouse_clf1.predict_proba(test_fvector.reshape(1,-1))[0,1]
            fans_pred1 = fans_clf1.predict_proba(test_fvector.reshape(1,-1))[0,1]
            opt_drives_pred1 = opt_drives_clf1.predict_proba(test_fvector.reshape(1,-1))[0,1]
            ports_pred1 = ports_clf1.predict_proba(test_fvector.reshape(1,-1))[0,1]
            graphics_pred1 = graphics_clf1.predict_proba(test_fvector.reshape(1,-1))[0,1]
            mm_devs_pred1 = mm_devs_clf1.predict_proba(test_fvector.reshape(1,-1))[0,1]
            hardw_pred1 = hardw_clf1.predict_proba(test_fvector.reshape(1,-1))[0,1]
            os_pred1 = os_clf1.predict_proba(test_fvector.reshape(1,-1))[0,1]
            softw_pred1 = softw_clf1.predict_proba(test_fvector.reshape(1,-1))[0,1]
            warranty_pred1 = warranty_clf1.predict_proba(test_fvector.reshape(1,-1))[0,1]
            shipping_pred1 = shipping_clf1.predict_proba(test_fvector.reshape(1,-1))[0,1]
            support_pred1 = support_clf1.predict_proba(test_fvector.reshape(1,-1))[0,1]
            company_pred1 = company_clf1.predict_proba(test_fvector.reshape(1,-1))[0,1]
            general_pred1 = general_clf1.predict_proba(test_fvector.reshape(1,-1))[0,1]
            price_pred1 = price_clf1.predict_proba(test_fvector.reshape(1,-1))[0,1]
            quality_pred1 = quality_clf1.predict_proba(test_fvector.reshape(1,-1))[0,1]
            op_perf_pred1 = op_perf_clf1.predict_proba(test_fvector.reshape(1,-1))[0,1]
            usability_pred1 = usability_clf1.predict_proba(test_fvector.reshape(1,-1))[0,1]
            des_feats_pred1 = des_feats_clf1.predict_proba(test_fvector.reshape(1,-1))[0,1]
            portability_pred1 = portability_clf1.predict_proba(test_fvector.reshape(1,-1))[0,1]
            connectivity_pred1 = connectivity_clf1.predict_proba(test_fvector.reshape(1,-1))[0,1]
            misc_pred1 = misc_clf1.predict_proba(test_fvector.reshape(1,-1))[0,1]

            entity_prob = {"laptop": laptop_pred1, "display": display_pred1, "cpu": cpu_pred1,
                                "motherboard": mb_pred1, "hard_disc": hd_pred1,
                               "memory": memory_pred1, "battery": battery_pred1,
                               "power_supply": power_pred1, "keyboard": keyboard_pred1,
                               "mouse": mouse_pred1,
                               "fans_cooling": fans_pred1, "optical_drives": opt_drives_pred1,
                                "ports": ports_pred1, "graphics": graphics_pred1,
                               "multimedia_devices": mm_devs_pred1, "hardware": hardw_pred1,
                               "os": os_pred1, "software": softw_pred1, "warranty": warranty_pred1,
                               "shipping": shipping_pred1, "support": support_pred1,
                               "company": company_pred1}

            attr_prob = {"general": general_pred1, "price": price_pred1, "quality": quality_pred1,
                             "operation_performance": op_perf_pred1, "usability": usability_pred1,
                             "design_features": des_feats_pred1,
                             "portability": portability_pred1, "connectivity": connectivity_pred1,
                             "miscellaneous": misc_pred1}

            sorted_entity_prob = sorted(entity_prob.items(), key=operator.itemgetter(1), reverse=True)
            sorted_attr_prob = sorted(attr_prob.items(), key=operator.itemgetter(1), reverse=True)
            categories = []
            
            for entity in sorted_entity_prob:
                for attr in sorted_attr_prob:
                    if entity[1] > 0.4 and attr[1] > 0.4:
                        category = entity[0]+'#'+attr[0]
                        for c in cat_dict:
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

    with open('acd_const_lapt_semeval.xml', 'w') as o:
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
    use_msg = 'Use as:\n">>> python acd_constrained_laptops.py --train train.xml --test test.xml"\n\nThis will parse a train set, examine whether is valid, perform category detection on the test set provided, and write out a file with the predictions.'
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
    domain_name = 'laptops'

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
