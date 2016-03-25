import gensim
import json
import re
from nltk.corpus import stopwords

with open('kw.json') as data_file:
    data = json.load(data_file)

corpus = [' '.join(x['keywords']) for x in data]


for i,row in enumerate(corpus):
    corpus[i]=re.sub("[^a-zA-Z1-9]"," ",row.replace(row,row.lower())).split()

stops = set(stopwords.words("english"))
for row,col in enumerate(corpus):
    for i in col:
        if i in stops:
            col.remove(i)

model = gensim.models.Word2Vec(corpus,size=200, window=5, workers=4)
model.save_word2vec_format('w2v_model.txt')
model.save_word2vec_format('w2v_model.bin', binary=True)