# -*- coding: utf-8 -*-
import cv2
import numpy as np
import pandas as pd
from sklearn import metrics, cross_validation, neighbors, pipeline, preprocessing, svm, ensemble, linear_model
import xgboost as xgb
root_path='./'
train_data=pd.read_csv(root_path + 'final_train.csv')
test_data=pd.read_csv(root_path + '/final_test.csv')


def extract_features(data, limit=np.inf):
    X = np.zeros((min(limit, data.shape[0]), 1000))#alloc space for the features
    for index, row in data.iterrows():#iterate over csv file
        if index==limit:
            break
        img=cv2.imread(root_path + '/images/' + str(row['image_id']) + '.jpg')
        img1=cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        histogram = cv2.calcHist([img1],[0,1,2],None,[10,10,10],[0,256,0,256,0,256]).ravel()
        X[index]=histogram.ravel()#to 1d array
    return X
    
X = extract_features(train_data, 12000)
y = train_data['image_label'].values[:X.shape[0]].ravel()

model = xgb.sklearn.XGBClassifier(max_depth=15, learning_rate=0.01, subsample=0.9, colsample_bytree=0.9, n_estimators=3000)

kf=cross_validation.StratifiedKFold(y, n_folds=5)
scores = np.array([])
for train_index, val_index in kf:
    model.fit(X[train_index], y[train_index])
    preds=model.predict_proba(X[val_index])[:, 1]
    print ('AUC ROC: ', metrics.roc_auc_score(y[val_index], preds))
    scores = np.hstack([scores, metrics.roc_auc_score(y[val_index], preds)])

print ('mean: ', scores.mean(), 'std: ', scores.std())

# submission generating

preds = model.fit(X, y).predict_proba(extract_features(test_data))[:, 1]
test_data = test_data.drop('image_url', 1)
test_data['image_label'] = preds
test_data.to_csv(root_path + '/res.csv', index=False)
