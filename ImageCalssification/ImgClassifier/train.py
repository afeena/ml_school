# -*- coding: utf-8 -*-
import cv2
import numpy as np
import pandas as pd
import numpy as np
from sklearn import metrics, preprocessing, ensemble
import xgboost as xgb

root_path = './'
train_data = pd.read_csv(root_path + 'final_train.csv')
test_data = pd.read_csv(root_path + '/final_test.csv')


def extract_features(data, limit=np.inf):
    X = np.zeros((min(limit, data.shape[0]), 9100))
    for index, row in data.iterrows():
        if index == limit:
            break
        img = cv2.imread(root_path + '/images/' + str(row['image_id']) + '.jpg')
        img1 = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

        histogram = cv2.calcHist([img1], [0, 1, 2], None, [10, 10, 10], [0, 256, 0, 256, 0, 256]).ravel()

        img2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img2 = cv2.resize(img2, (128, 128))
        hog = cv2.HOGDescriptor("hog.xml")
        h = hog.compute(img2).ravel()
        X[index] = np.hstack([histogram,h]).ravel() 

    return X


def fscore(predicted, raw):
    y = raw.get_label()
    y_pred = np.copy(predicted)
    for val in np.nditer(y_pred, op_flags=['readwrite']):
        val[...] = 1 if val[...] > 0.5 else 0
    fscr = metrics.f1_score(y, y_pred)
    return ("fscore", fscr)


X = extract_features(train_data)
y = train_data['image_label'].values[:X.shape[0]].ravel()


model_tmp = ensemble.RandomForestClassifier(n_estimators=2000, n_jobs=-1).fit(X, y)
indices = np.argsort(model_tmp.feature_importances_)[::-1]
indices = indices[:1500]
X = X[:, indices]

DX = xgb.DMatrix(X, label=y)
params = {'booster': 'gbtree',
          'max_depth': 10,
          'eta': 0.03,
          'silent': 1,
          'subsample': 0.9,
          'colsample_bytree': 0.9,
          'objective': 'binary:logistic',
          'nthread': 4,
        }

xgb.cv(params=params, dtrain=DX, nfold=5, show_progress=True, num_boost_round=2000, feval=fscore)

bst = xgb.Booster(params, [DX])
for i in range(2000):
    bst.update(DX, i)
bst.save_model("booster")
np.savetxt('indices',indices,fmt='%i',delimiter=' ')
