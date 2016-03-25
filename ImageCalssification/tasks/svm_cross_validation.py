# -*- coding: utf-8 -*-
import cv2
import numpy as np
import pandas as pd
from sklearn import metrics, svm, cross_validation

root_path='./'
train_data=pd.read_csv(root_path + 'final_train.csv')
test_data=pd.read_csv(root_path + '/final_test.csv')

def extract_features(data, limit=np.inf):
    X = np.zeros((min(limit, data.shape[0]), 768))#alloc space for the features
    for index, row in data.iterrows():#iterate over csv file
        if index==limit:
            break
        img=cv2.imread(root_path + '/images/' + str(row['image_id']) + '.jpg')
        histogram=np.zeros((3, 256))
        for i in range(3):#calc hist for each channel
            histogram[i] = cv2.calcHist([img],[i],None,[256],[0,256]).ravel()
            cv2.normalize(histogram[i], histogram[i], 0, 1, cv2.NORM_MINMAX)
        X[index]=histogram.ravel()#to 1d array
    return X

X = extract_features(train_data, 1000)
y = train_data['image_label'].values[:X.shape[0]].ravel()

model = svm.SVC(C=1, kernel='linear', probability=True)
kf=cross_validation.KFold(X.shape[0], n_folds=5)
for train_index, val_index in kf:
    model.fit(X[train_index], y[train_index])
    preds=model.predict_proba(X[val_index])[:, 1]
    print ('AUC ROC: ', metrics.roc_auc_score(y[val_index], preds))

# submission generating
preds = model.fit(X, y).predict_proba(extract_features(test_data))[:, 1]
test_data = test_data.drop('image_url', 1)
test_data['image_label'] = preds
test_data.to_csv(root_path + '/res.csv', index=False)
