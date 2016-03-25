import json
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD, NMF, PCA
from sklearn.preprocessing import normalize
from bokeh.plotting import figure,output_file,show


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()


def plot_words(matrix, feature_names, filename,title):
    output_file(filename+".html")
    p=figure(plot_width=1024,plot_height=768, title=title)
    p.text(matrix[:,0],matrix[:,1], features_names)
    show(p)

with open('istock.json') as data_file:
    data = json.load(data_file)

corpus = [' '.join(x['keywords']) for x in data]
tfidf_vectorizer = TfidfVectorizer(sublinear_tf=True, use_idf=True, stop_words='english')

X = tfidf_vectorizer.fit_transform(corpus)
y = np.array([x['downloads_count'] for x in data])


nmf = NMF(n_components=10, random_state=1, alpha=.1, l1_ratio=.5).fit(X)
X_nmf_red = nmf.transform(X)

print_top_words(nmf,tfidf_vectorizer.get_feature_names(),10)

features_names = tfidf_vectorizer.get_feature_names()

tsne = TSNE(n_components=2, random_state=0)
np.set_printoptions(suppress=True)
svd = TruncatedSVD(n_components=2)
pca = PCA(n_components=2)
wpca = PCA(n_components=2, whiten=True)

#use A=X.T.dot(X) matrix

A=X.T.dot(X)

A_svd_reduced = svd.fit_transform(A)
plot_words(A_svd_reduced,features_names,"svd_a","SVD A=X.T.dot(X)")

svd.set_params(n_components = 50)
A_svd_reduced = svd.fit_transform(A)

A_pca = pca.fit_transform(A_svd_reduced)
plot_words(A_pca,features_names,"pca_a", "PCA A reduced matrix")

A_wpca  = wpca.fit_transform(A_svd_reduced)
plot_words(A_wpca,features_names,"wpca_a", "PCA A reduced matrix, Whiten=True")

A_tsne = tsne.fit_transform(A_svd_reduced)
plot_words(A_tsne,features_names,"tsne_a", "t-SNE A reduced matrix")

#use nmf.components_ matrix

t_nmf = nmf.components_.T
tsne_nmf = tsne.fit_transform(t_nmf)
plot_words(tsne_nmf,features_names,"tsne_nmf", "t-SNE nmf W.T matrix")

svd.set_params(n_components = 2)
svd_nmf = svd.fit_transform(t_nmf)
plot_words(svd_nmf,features_names,"svd_nmf", "SVD nmf W.T matrix")

pca_nmf = pca.fit_transform(t_nmf)
plot_words(pca_nmf,features_names,"pca_nmf", "PCA nmf W.T  matrix")

wpca_nmf  = wpca.fit_transform(t_nmf)
plot_words(wpca_nmf,features_names,"wpca_nmf", "PCA nmf W.T  matrix, Whiten=True")

#use normalized nmf.components_ matrix
t_nmf_norm = normalize(t_nmf, norm='l2')
tsne_nmf_norm = tsne.fit_transform(t_nmf_norm)
plot_words(tsne_nmf_norm,features_names,"tsne_nmf_norm", "t-SNE nmf W.T normalized  matrix")

svd_nmf_norm = svd.fit_transform(t_nmf_norm)
plot_words(svd_nmf_norm,features_names,"svd_nmf_norm", "SVD nmf W.T normalized matrix")

pca_nmf_norm = pca.fit_transform(t_nmf_norm)
plot_words(pca_nmf_norm,features_names,"pca_nmf_norm", "PCA nmf W.T normalized matrix")

wpca_nmf_norm  = wpca.fit_transform(t_nmf_norm)
plot_words(pca_nmf_norm,features_names,"wpca_nmf_norm", "PCA nmf W.T normalized matrix, Whiten=True")