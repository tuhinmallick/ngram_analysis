# -*- coding: utf-8 -*-

import re
import nltk
import string

def cleanhtmlfun(raw):
  cleanit = re.compile('<.*?>')
  text = re.sub(cleanit, '', raw)
  return text



f=open('wiki_00', "r",encoding="utf8")
raw=f.read()
raw=cleanhtmlfun(raw)

from nltk.tokenize import TreebankWordTokenizer
tbw=TreebankWordTokenizer()
tokens=tbw.tokenize(raw)
tokens = [''.join(c for c in s if c not in string.punctuation) for s in tokens]
tokens = [s for s in tokens if s]


from nltk.stem.snowball import SnowballStemmer
stemmer=SnowballStemmer("english")
tokens=[stemmer.stem(word) for word in tokens]

unigrams=nltk.ngrams(tokens,1)
fdist_unigrams=nltk.FreqDist(unigrams) 
unique_unigrams=fdist_unigrams.B()  #This gives the total number of unique unigrams



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

size=0
for freq in Y:
    size+=freq

count=0
most_freq_unigrams=0
corp_req=0.90*size
for freq in Y:
    count+=freq
    most_freq_unigrams+=1   #This gives the number of unique unigrams required to cover 90% of the corpus
    if count>=corp_req:
        break