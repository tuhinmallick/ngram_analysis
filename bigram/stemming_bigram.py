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

from nltk.tokenize import TreebankWordTokenizer
tbw=TreebankWordTokenizer()
tokens=tbw.tokenize(raw)
tokens = [''.join(c for c in s if c not in string.punctuation) for s in tokens]
tokens = [s for s in tokens if s]


from nltk.stem.snowball import SnowballStemmer
stemmer=SnowballStemmer("english")
tokens=[stemmer.stem(word) for word in tokens]


bigrams=nltk.ngrams(tokens,2)
fdist_bigrams=nltk.FreqDist(bigrams)
unique_bigrams=fdist_bigrams.B()    #This gives the total number of unique bigrams



import matplotlib.pyplot as plt

Y = fdist_bigrams.values()
Y = sorted(Y, reverse=True)
X = range(len(Y))
plt.figure()
plt.loglog(X, Y)
plt.xlabel('Bigram')
plt.ylabel('Frequency')
plt.title('Bigram Frequencies')
plt.grid()
plt.show()

size = sum(Y)
count=0
most_freq_bigrams=0
corp_req=0.80*size
for freq in Y:
    count+=freq
    most_freq_bigrams+=1    #This gives the number of unique bigrams required to cover 80% of the corpus
    if count>=corp_req:
        break