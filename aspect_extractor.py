import xml.etree.ElementTree as ET
import copy
from xml.sax.saxutils import escape
import re
import json


class Aspect:
    '''Aspect objects contain the term (e.g., battery life) and polarity (i.e., pos, neg, neu, conflict) of an aspect.'''

    def __init__(self, term, polarity, offsets):
        self.term = term
        self.polarity = polarity
        self.offsets = offsets

    def create(self, element):
        self.term = element.attrib['term']
        self.polarity = element.attrib['polarity']
        self.offsets = {'from': str(element.attrib['from']), 'to': str(element.attrib['to'])}
        return self

    def update(self, term='', polarity=''):
        self.term = term
        self.polarity = polarity


class Category:
    '''Category objects contain the term and polarity (i.e., pos, neg, neu, conflict) of the category (e.g., food, price, etc.) of a sentence.'''

    def __init__(self, term='', polarity=''):
        self.term = term
        self.polarity = polarity

    def create(self, element):
        self.term = element.attrib['category']
        self.polarity = element.attrib['polarity']
        return self

    def update(self, term='', polarity=''):
        self.term = term
        self.polarity = polarity


class Instance:
    '''An instance is a sentence, modeled out of XML (pre-specified format, based on the 4th task of SemEval 2014).
    It contains the text, the aspect terms, and any aspect categories.'''

    def __init__(self, element):
        self.text = element.find('text').text
        self.id = element.get('id')
        self.aspect_terms = [Aspect('', '', offsets={'from': '', 'to': ''}).create(e) for es in
                             element.findall('aspectTerms') for e in es if
                             es is not None]
        self.aspect_categories = [Category(term='', polarity='').create(e) for es in element.findall('aspectCategories')
                                  for e in es if
                                  es is not None]

    def get_aspect_terms(self):
        return [a.term.lower() for a in self.aspect_terms]

    def get_aspect_categories(self):
        return [c.term.lower() for c in self.aspect_categories]

    def add_aspect_term(self, term, polarity='', offsets={'from': '', 'to': ''}):
        a = Aspect(term, polarity, offsets)
        self.aspect_terms.append(a)

    def add_aspect_category(self, term, polarity=''):
        c = Category(term, polarity)
        self.aspect_categories.append(c)


def fd(counts):
    '''Given a list of occurrences (e.g., [1,1,1,2]), return a dictionary of frequencies (e.g., {1:3, 2:1}.)'''
    d = {}
    for i in counts: d[i] = d[i] + 1 if i in d else 1
    return d


freq_rank = lambda d: sorted(d, key=d.get, reverse=True)
fix = lambda text: escape(text.encode('utf8')).replace('\"','&quot;')


class Corpus:
    '''A corpus contains instances, and is useful for training algorithms or splitting to train/test files.'''

    def __init__(self, elements):
        self.corpus = [Instance(e) for e in elements]
        self.size = len(self.corpus)
        self.aspect_terms_fd = fd([a for i in self.corpus for a in i.get_aspect_terms()])
        self.top_aspect_terms = freq_rank(self.aspect_terms_fd)
        self.texts = [t.text for t in self.corpus]

#     def echo(self):
#         print '%d instances\n%d distinct aspect terms' % (len(self.corpus), len(self.top_aspect_terms))
#         print 'Top 10 aspect terms: %s' % (', '.join(self.top_aspect_terms[:10]))

    def clean_tags(self):
        for i in range(len(self.corpus)):
            self.corpus[i].aspect_terms = []

    def split(self, threshold=0.8, shuffle=False):
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
                # o.write('\t\t<text>%s</text>\n' % fix(i.text))
                o.write('\t\t<text>%s</text>\n' % i.text)
                o.write('\t\t<aspectTerms>\n')
                if not short:
                    for a in i.aspect_terms:
#                         o.write('\t\t\t<aspectTerm term="%s" polarity="%s" from="%s" to="%s"/>\n' % (
#                             fix(a.term), a.polarity, a.offsets['from'], a.offsets['to']))
                        o.write('\t\t\t<aspectTerm term="%s" polarity="%s" from="%s" to="%s"/>\n' % (
                            a.term, a.polarity, a.offsets['from'], a.offsets['to']))
                o.write('\t\t</aspectTerms>\n')
                o.write('\t\t<aspectCategories>\n')
                if not short:
                    for c in i.aspect_categories:
                        o.write('\t\t\t<aspectCategory category="%s" polarity="%s"/>\n' % (c.term, c.polarity))
                o.write('\t\t</aspectCategories>\n')
                o.write('\t</sentence>\n')
            o.write('</sentences>')


class BaselineAspectExtractor():
    '''Extract the aspects from a text.
    Use the aspect terms from the train data, to tag any new (i.e., unseen) instances.'''

    def __init__(self, corpus):
        self.candidates = [a.lower() for a in corpus.top_aspect_terms]

    def find_offsets_quickly(self, term, text):
        start = 0
        while True:
            start = text.find(term, start)
            if start == -1: return
            yield start
            start += len(term)

    def find_offsets(self, term, text):
        offsets = [(i, i + len(term)) for i in list(self.find_offsets_quickly(term, text))]
        return offsets

    def tag(self, test_instances):
        clones = []
        for i in test_instances:
            i_ = copy.deepcopy(i)
            i_.aspect_terms = []
            for c in set(self.candidates):
                if c in i_.text:
                    offsets = self.find_offsets(' ' + c + ' ', i.text)
                    for start, end in offsets: i_.add_aspect_term(term=c,
                                                                  offsets={'from': str(start + 1), 'to': str(end - 1)})
            clones.append(i_)
        return clones


def dice(t1, t2, stopwords=[]):
    tokenize = lambda t: set([w for w in t.split() if (w not in stopwords)])
    t1, t2 = tokenize(t1), tokenize(t2)
    return 2. * len(t1.intersection(t2)) / (len(t1) + len(t2))


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


class BaselineCategoryDetector():
    '''Detect the category (or categories) of an instance.
    For any new (i.e., unseen) instance, fetch the k-closest instances from the train data, and vote for the number of categories and the categories themselves.'''

    def __init__(self, corpus):
        self.corpus = corpus

    # Fetch k-neighbors (i.e., similar texts), using the Dice coefficient, and vote for #categories and category values
    def fetch_k_nn(self, text, k=5, multi=False):
        neighbors = dict([(i, dice(text, n, stopwords)) for i, n in enumerate(self.corpus.texts)])
        ranked = freq_rank(neighbors)
        topk = [self.corpus.corpus[i] for i in ranked[:k]]
        num_of_cats = 1 if not multi else int(sum([len(i.aspect_categories) for i in topk]) / float(k))
        cats = freq_rank(fd([c for i in topk for c in i.get_aspect_categories()]))
        categories = [cats[i] for i in range(num_of_cats)]
        return categories

    def tag(self, test_instances):
        clones = []
        for i in test_instances:
            i_ = copy.deepcopy(i)
            i_.aspect_categories = [Category(term=c) for c in self.fetch_k_nn(i.text)]
            clones.append(i_)
        return clones


class AspectExtractor():
    '''Stage I: Aspect Term Extraction and Aspect Category Detection.'''

    def __init__(self, b1, b2):
        self.b1 = b1
        self.b2 = b2

    def tag(self, test_instances):
        clones = []
        for i in test_instances:
            i_ = copy.deepcopy(i)
            i_.aspect_categories, i_.aspect_terms = [], []
            for a in set(self.b1.candidates):
                offsets = self.b1.find_offsets(' ' + a + ' ', i_.text)
                for start, end in offsets:
                    i_.add_aspect_term(term=a, offsets={'from': str(start + 1), 'to': str(end - 1)})
            for c in self.b2.fetch_k_nn(i_.text):
                i_.aspect_categories.append(Category(term=c))
            clones.append(i_)
        return clones

# prepocess data
json_file = 'rrdata.json'
Handle = open(json_file,'r')
Buff = Handle.readlines()

data_dict = []
for line in Buff:
    data_dict.append(json.loads(line))
rr_data = data_dict[0]

review_list = []
for i in range(len(rr_data)):
    review_list += rr_data[i]['review']

partOfReview = review_list[:1000]

from nltk.tokenize import sent_tokenize
import string
import re

# sentence segmentaion
reviewSentence = []
for review in partOfReview:
    reviewSentence += sent_tokenize(review)

regex = re.compile('[%s]' % re.escape(string.punctuation))
finalSentence = []

for sentence in reviewSentence:
    finalSentence.append(regex.sub(u'', sentence))

while [] in finalSentence:
    finalSentence.remove([])
    
while "" in finalSentence:
    finalSentence.remove("")

# transform format
with open("testReview.xml", 'w') as o:
    o.write('<sentences>\n')
    for index, sentence in enumerate(finalSentence):
        o.write('\t<sentence id="%s">\n' % index)
        o.write('\t\t<text>%s</text>\n' % sentence)
        o.write('\t\t<aspectTerms>\n')
        o.write('\t\t</aspectTerms>\n')
        o.write('\t\t<aspectCategories>\n')
        o.write('\t\t</aspectCategories>\n')
        o.write('\t</sentence>\n')
    o.write('</sentences>')

# specify train file and test file
trainfile = "/Users/cee/Downloads/AspectBasedSentimentAnalysis/datasets/Restaurants_Train_v2.xml"
testfile = "testReview.xml"

corpus = Corpus(ET.parse(trainfile).getroot().findall('sentence'))
domain_name = 'AspectBasedSA'

traincorpus = corpus
seen = Corpus(ET.parse(testfile).getroot().findall('sentence'))

corpus.write_out('%s--test.xml' % domain_name, seen.corpus)
unseen = Corpus(ET.parse('%s--test.xml' % domain_name).getroot().findall('sentence'))

# initialize aspect extractor and category detector
baselineAspectExtractor = BaselineAspectExtractor(traincorpus)
baselineCategoryDetector = BaselineCategoryDetector(traincorpus)

# predict category of test file
aspectExtractor = AspectExtractor(baselineAspectExtractor, baselineCategoryDetector)
predicted = aspectExtractor.tag(unseen.corpus)