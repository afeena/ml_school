# -*- coding: utf-8 -*-
import cv2
import numpy as np
import pandas as pd
import numpy as np
import xgboost as xgb
import pymysql

root_path='./'
train_data=pd.read_csv(root_path + 'final_train.csv')
test_data=pd.read_csv(root_path + 'final_test.csv')

def extract_features(data, limit=np.inf):
    X = np.zeros((min(limit, data.shape[0]), 9100))#alloc space for the features
    for index, row in data.iterrows():#iterate over csv file
        if index==limit:
            break
        img=cv2.imread('/data/images/' + str(row['image_id']) + '.jpg')
        img1=cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

        histogram = cv2.calcHist([img1],[0,1,2],None,[10,10,10],[0,256,0,256,0,256]).ravel()

        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img2 = cv2.resize(img2,(128,128))
        hog = cv2.HOGDescriptor((128,128),(16,16),(8,8),(8,8),9,1,4,0,2.00000000001e-01,0,64)
        h = hog.compute(img2).ravel()
        X[index]=np.hstack([histogram.ravel(),h]).ravel()#to 1d array

    return X

# X = extract_features(train_data)
# y = train_data['image_label'].values[:X.shape[0]].ravel()
#
# DX = xgb.DMatrix(X, label=y)
# params = {'booster':'gbtree',
#      'max_depth':10,
#      'eta':0.03,
#      'silent':1,
#      'objective':'binary:logistic',
#      'nthread':4,
#      'eval_metric':'auc'
#      }
#
# xgb.cv(params=params, dtrain=DX, nfold=5, show_progress = True, num_boost_round=2000)
indices = np.loadtxt("indices_old",dtype='int',delimiter=' ');
bst = xgb.Booster(params=None)
bst.load_model("./booster_old")
#for i in range(2000):
#    bst.update(DX, i)
preds = bst.predict(xgb.DMatrix(extract_features(test_data)[:,indices]))
#test_data = test_data.drop('image_url', 1)
test_data['image_label'] = preds
test_data.to_csv(root_path + '/res.csv', index=False)

server = "localhost"
user = "mlstudent1"
password = "b3e462d2b0"

conn = pymysql.connect(server, user, password, "mlschool")
cursor = conn.cursor()
for index, row in test_data.iterrows():
    label = float(row['image_label']);   
    label = 1 if label>0.18 else 0   
    imid = int(row['image_id'])   
    cursor.execute(
    """REPLACE INTO mlstudent1 VALUES (%s, %s)""",
    (int(imid),int(label)))
# you must call commit() to persist your data if you don't set autocommit to True
    conn.commit()
conn.close
#bst.save_model("./boost")



