# -*- coding: utf-8 -*-


import re
import nltk
import string

def cleanhtmlfun(raw):
  cleanit = re.compile('<.*?>')
  return re.sub(cleanit, '', raw)



f=open('wiki_00', "r",encoding="utf8")
raw=f.read()
raw=cleanhtmlfun(raw)


from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize, pos_tag
from collections import defaultdict

tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV


tokens = word_tokenize(raw)
tokens = [''.join(c for c in s if c not in string.punctuation) for s in tokens]
tokens = [s for s in tokens if s]
lm = WordNetLemmatizer()

tokens2=[]
for token, tag in pos_tag(tokens):
    lemma = lm.lemmatize(token, tag_map[tag[0]])
    tokens2.append(lemma)

unigrams=nltk.ngrams(tokens2,1)
fdist_unigrams=nltk.FreqDist(unigrams)
unique_unigrams=fdist_unigrams.B() #This gives the total number of unique unigrams


import matplotlib.pyplot as plt

Y = fdist_unigrams.values()
Y = sorted(Y, reverse=True)
X = range(len(Y))
plt.figure()
plt.loglog(X, Y)
plt.xlabel('Unigram')
plt.ylabel('Frequency')
plt.title('Unigram Frequencies')
plt.grid()
plt.show()

size = sum(Y)
count=0
most_freq_unigrams=0
corp_req=0.90*size
for freq in Y:
    count+=freq
    most_freq_unigrams+=1 #This gives the number of unique unigrams required to cover 90% of the corpus
    if count>=corp_req:
        break