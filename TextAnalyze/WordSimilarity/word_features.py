import formatter
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF


with open('kw.json') as data_file:
    data = json.load(data_file)

corpus = [' '.join(x['keywords']) for x in data]
tfidf_vectorizer = TfidfVectorizer(sublinear_tf=True, use_idf=True, stop_words='english')

X = tfidf_vectorizer.fit_transform(corpus)
y = np.array([x['downloads_count'] for x in data])


nmf = NMF(n_components=50, random_state=1, alpha=.1, l1_ratio=.5).fit(X)
nmf.transform(X)

features = tfidf_vectorizer.get_feature_names()

np.set_printoptions(linewidth=500)
feature_file = open('features','w')
for i in range(nmf.components_.shape[1]):
    feature_file.write(features[i]+" "+' '.join(map(str,nmf.components_[:,i]))+"\n")

feature_file.close()
