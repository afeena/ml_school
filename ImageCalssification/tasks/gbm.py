# -*- coding: utf-8 -*-
import cv2
import numpy as np
import pandas as pd
import numpy as np
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

X = extract_features(train_data)
y = train_data['image_label'].values[:X.shape[0]].ravel()

DX = xgb.DMatrix(X, label=y)
params = {'booster':'gbtree',
     'max_depth':20,
     'eta':0.03,
     'silent':1,
     'objective':'binary:logistic',
     'nthread':4,
     'eval_metric':'auc'
     }

xgb.cv(params=params, dtrain=DX, nfold=5, num_boost_round=2000)

bst = xgb.Booster(params, [DX])
for i in range(100):
    bst.update(DX, i)
    print("iteration: ", i)
preds = bst.predict(xgb.DMatrix(extract_features(test_data)))

# submission generating
test_data = test_data.drop('image_url', 1)
test_data['image_label'] = preds
test_data.to_csv(root_path + '/res.csv', index=False)# -*- coding: utf-8 -*-

