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

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

tokens = [w for w in tokens if not w in stop_words]

bigrams = [(tokens[i-1], tokens[i]) for i in range(1, len(tokens))]

words=[]
for i in range(1, len(bigrams)):
        
        
            x11=0
            x21=0
            x12=0
            x22=0
            for j in range(1, len(bigrams)):
                if bigrams[j][0]==bigrams[i][0] :
                    if bigrams[j][1]==bigrams[i][1]:
                        x11+=1
                    else:
                        x21+=1
                else:
                    if bigrams[j][1]==bigrams[i][1]:
                        x12+=1
                    else:
                        x22+=1
            chi=len(bigrams)*((x11*x22)-(x12*x21))**2/((x11+x12)*(x11+x21)*(x12+x22)*(x21+x22))
            words.append((bigrams[i][0],bigrams[i][1],chi))
            
fdist_words=nltk.FreqDist(words)
print(fdist_words.most_common(20))
    
    
    
    
    
    