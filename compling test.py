#!/usr/bin/env python
# coding: utf-8

# In[6]:


from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import pymorphy2
import operator


# Задание 1.

# In[8]:


text = "Зонды будут запущены в космос в 2029 году и отправятся ко второй точке Лагранжа в системе Солнце"


# In[9]:


morph = pymorphy2.MorphAnalyzer()

tokens = text.split(' ')
text_parts = [] 
verbs = []

for i in tokens:
    pos = morph.parse(i.lower())[0].tag.POS
    text_parts.append([i, pos])
    if pos == "VERB":
        verbs.append([i, pos])


# In[10]:


print(verbs)


# Задание 2.

# In[11]:


import requests
r = requests.get(
    'https://raw.githubusercontent.com/mannefedov/compling_nlp_hse_course/master/data/anna_karenina.txt'
)

# работайте с этими предложениями
sentences = r.text.split('\n')
sentences = [sent for sent in sentences if len(sent) > 10]


# In[12]:


vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(sentences)


# In[13]:


feature_index = X[0,:].nonzero()[1]
tfidf_scores = zip(feature_index, [X[0, i] for i in feature_index])
rate = []
for w, s in [(vectorizer.get_feature_names()[i], s) for (i, s) in tfidf_scores]:
    rate.append([w, s])
sortRate  = sorted(rate, key = operator.itemgetter(1,0))
print(sortRate[-1])


# Задание 3.

# In[18]:


from razdel import tokenize
from collections import Counter


# In[19]:


a = []
for sentence in sentences:
    tokens = list(tokenize(sentence))
    a.append([_.text.lower() for _ in tokens])

from nltk import word_tokenize 
from nltk.util import ngrams

bigram_list = []

for i in a:
    bigram = list(ngrams(i, 2))
    bigram_list.append(bigram)
    
bigram_list

bigram_list_flat = []

for i in bigram_list:
    for j in i:
        if j[0] == 'красный':
            bigram_list_flat.append(j)

Counter(bigram_list_flat)


# Задание 4.

# In[90]:


words = ["решение","ршеение","ренешик","рещиние","ришение"]


# In[88]:


get_ipython().system('pip install textdistance')
import textdistance


# In[91]:


dl_dict1 = {}
dl_dict2 = {}

for el in words:
  el_dl_dict1 = {}
  el_dl_dict2 = {}
  for els in words:
    el_dl_dict1[els] = textdistance.damerau_levenshtein.distance(el, els)
    el_dl_dict2[els] = textdistance.levenshtein.distance(el, els)
  dl_dict1[el] = el_dl_dict1
  dl_dict2[el] = el_dl_dict2


# In[92]:


dl_dict1


# Задание 5.

# In[77]:


get_ipython().system('pip install fasttext')


# In[84]:


from transformers import DistilBertTokenizer, DistilBertModel
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained("distilbert-base-uncased")
import requests
import numpy as np
import operator
import fasttext

for i in sentences:
    ids = tokenizer.encode(i)
    [tokens.append(i) for i in ids]
print(tokens)

model = fasttext.train_unsupervised(tokens, model='cbow', minn = 2, maxn = 7, dim=300, ws=5, )

