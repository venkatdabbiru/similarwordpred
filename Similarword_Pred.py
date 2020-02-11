# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 17:06:50 2020

@author: 703172796
"""

import gensim
import os
import re
import nltk
nltk.download('gutenberg')
nltk.download('punkt')
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import sent_tokenize 
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from gensim.models import Word2Vec, KeyedVectors
import warnings
warnings.filterwarnings("ignore")
#from sklearn.decomposition import PCA
from nltk.corpus import gutenberg
import pickle


stemmer = SnowballStemmer("english")
stopWords = pd.read_csv('stopwords.txt').values


#fileslist = gutenberg.fileids()
#print(fileslist)
#print(textsample)
#vocabulary = []
#textsample = gutenberg.raw(fileids='bible-kjv.txt')
#print(textsample)
#for line in textsample:
#  words = re.findall(r'(\b[a-z][a-z]*\b)', line.lower())
#  words = [word for word in words if not word in stopwords.words('english')]
#  for word in words:
#    vocabulary.append(words)
#len(vocabulary)

class Load_Data(object):
    def __init__(self, fnamelist):
        self.fnamelist = fnamelist
        self.vocabulary = []
        
    
    def __iter__(self):
        for fname in self.fnamelist:
            for line in open(fname, encoding='latin1'):
                words = re.findall(r'(\b[a-z][a-z]*\b)', line.lower())
                words = [stemmer.stem(word) for word in words if not word in stopWords]
                for word in words:
                    self.vocabulary.append(word)
                yield words


vocabulary = []           
textsample = gutenberg.raw(fileids='bible-kjv.txt')
print(textsample)
for line in sent_tokenize(textsample):
    #print(line)
    words = re.findall(r'(\b[a-z][a-z]*\b)', line.lower())
    print(word)
    vocabulary.append(words)

print(vocabulary[:5])

#gutenberg.raw (fileids='austen-emma.txt')

#tok = sent_tokenize(textsample)
#print(tok[5:20])
                
MB_txt = Load_Data(textsample)
#print(MB_txt)
model = Word2Vec(vocabulary, min_count=30)
#min_count (int, optional) – Ignores all words with total frequency lower than this.



model.save("MB2Vec_Without_stemmer.bin")

model.most_similar('alice')

mostsimilar = model.most_similar('the')[:5]

for name, similarity in mostsimilar:
    print("Name: {} similarity: {}".format(name, round(similarity,2)))

print("\n\nChecking if similarity is symmetric")
word_without_stemmer = model.most_similar('doubt')[:5]

for name, similarity in word_without_stemmer:
    print("Name: {} similarity: {}".format(name, round(similarity,2)))
    
    
    
    
class Load_Data_stemmed(object):
    def __init__(self, fnamelist):
        self.fnamelist = fnamelist
        # Creating a set of vocabulary
        self.vocabulary = []

    def __iter__(self):
        for fname in self.fnamelist:
            for line in open(fname, encoding='latin1'):
                words = re.findall(r'(\b[a-z][a-z]*\b)', line.lower())
                # Stemming a word.
                words = [ stemmer.stem(word) for word in words if not word in stopWords]
                for word in words:
                    self.vocabulary.append(word)
                yield words
                
                
MB_txt_stemmed = Load_Data_stemmed(['MB.txt'])
model = gensim.models.Word2Vec(MB_txt_stemmed, min_count=100)

model.save("MB2Vec_With_stemmer.bin")

krishna5_with_stemmer =  model.wv.most_similar('krishna')[:5]
for name, similarity in krishna5_with_stemmer:
  print("Name: {} similarity: {}".format(name, round(similarity,2)))
  
  
###############################################################

#MB_txt = Load_Data(['MB.txt'])
#model = gensim.models.Word2Vec(MB_txt, min_count=100)
model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True, limit=500000)

#print(vocabulary[5:20])
#tok = sent_tokenize(textsample)
#print(tok[5:20])
#print(vocabulary)
model.save("Myword_similarity_model.bin")
pickle.dump(model, open('model.pkl', 'wb'))

word1_ = model.most_similar('repair')[:10]

for name, similarity in word1_:
    print("Name: {} similarity: {}".format(name, round(similarity,2)))

print('-------------------------------------')

word2_ = model.most_similar('lease')[:10]

for name, similarity in word2_:
    print("Name: {} similarity: {}".format(name, round(similarity,2)))

print('-------------------------------------')
