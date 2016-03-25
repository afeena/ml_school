import json
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, Lasso


def compute_grad(X, y, w, lambda_reg):
    grad = -2 * X.T.dot((y - X.dot(w))) + 2 * lambda_reg * w
    return grad


with open('istock.json') as data_file:
    data = json.load(data_file)

corpus = [' '.join(x['keywords']) for x in data]
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
y = np.array([x['downloads_count'] for x in data])
keyw = list(vectorizer.get_feature_names())

lambdas = [0.0001,0.001, 0.01, 0.1, 0.5, 1, 10, 100]
topwords = []
w = np.zeros(X.shape[1])
alpha = 0.1
for lambda_reg in lambdas:
    print("lambda", lambda_reg)
    for i in range(1000):
        w = w - alpha * (compute_grad(X, y, w, lambda_reg) / X.shape[0])

    w_abs = np.abs(w)
    sorted_weights_ind = np.argsort(w_abs)[::-1]
    print("GD done", mean_squared_error(y, X.dot(w)))

    ridge_clf = Ridge(alpha=lambda_reg, max_iter=1000)
    ridge_clf.fit(X, y)
    ridge_weights = ridge_clf.coef_
    ridge_abs_weights = np.abs(ridge_weights)
    sorted_rw_ind = np.argsort(ridge_abs_weights)[::-1]
    print("sklearn r done", mean_squared_error(y, X.dot(ridge_weights)))

    #lasso_clf = Lasso(alpha=lambda_reg, max_iter=1000)
    #lasso_clf.fit(X, y)
    #lasso_weights = lasso_clf.coef_
    #lasso_abs_weights = np.abs(lasso_weights)
    #sorted_lw_ind = np.argsort(lasso_abs_weights)[::-1]
    #print("sklearn l done", mean_squared_error(y, X.dot(lasso_weights)))

    topfive = []
    for i in range(5):
        list = {"lambda": lambda_reg, "custom_regr": keyw[sorted_weights_ind[i]],
                "ridge regression": keyw[sorted_rw_ind[i]],"place": i} #,"lasso regression": keyw[sorted_lw_ind[i]]}
        topfive.append(list)

    topwords.append(topfive)

for i in range(len(topwords)):
    print("____________________")
    for j in range(len(topwords[0])):
        print("lambda:", topwords[i][j]["lambda"],
              "place:", topwords[i][j]["place"],
              "custom:", topwords[i][j]["custom_regr"],
              "ridge sklearn:", topwords[i][j]["ridge regression"])
              #,"lasso sklearn:", topwords[i][j]["lasso regression"])
