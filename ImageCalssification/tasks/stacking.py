import cv2
import numpy as np
import pandas as pd
from sklearn import linear_model, metrics, cross_validation, neighbors, pipeline, preprocessing, svm, ensemble, \
    linear_model
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
import xgboost as xgb
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
import keras
import math
from skimage.feature import hog
from skimage import data, color, exposure

root_path = './'
train_data = pd.read_csv(root_path + 'final_train.csv')
test_data = pd.read_csv(root_path + '/final_test.csv')
models_preds = np.array([])


def extract_features(data, limit=np.inf):
    X = np.zeros((min(limit, data.shape[0]), 1800))  # alloc space for the features
    for index, row in data.iterrows():  # iterate over csv file
        if index == limit:
            break
        img = cv2.imread(root_path + '/images/' + str(row['image_id']) + '.jpg')
        img1 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        histogram = cv2.calcHist([img1], [0, 1, 2], None, [10, 10, 10], [0, 256, 0, 256, 0, 256]).ravel()
        img1 = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img1 = cv2.resize(img1,(100,100))
        fd, hog_image = hog(img1, orientations=8, pixels_per_cell=(10, 10),
                    cells_per_block=(1, 1), visualise=True)
        # histogram = cv2.calcHist([img1], [0, 1, 2], None, [10, 10, 10], [0, 256, 0, 256, 0, 256]).ravel()
        X[index] = np.hstack([fd,histogram]).ravel()  # to 1d array
    return X


def logit(value):
    value = 1 / (1 + math.exp(-value))
    return value


X = extract_features(train_data)
y = train_data['image_label'].values[:X.shape[0]].ravel()

model_tmp = ensemble.RandomForestClassifier(n_estimators=1000, n_jobs=-1).fit(X, y)
indices = np.argsort(model_tmp.feature_importances_)[::-1]
indices = indices[:1400]
X = X[:, indices]



model = xgb.sklearn.XGBClassifier(max_depth=10, learning_rate=0.03, subsample=0.9, colsample_bytree=0.9,
                                  n_estimators=2000)

kf = cross_validation.StratifiedKFold(y, n_folds=5)
scores = np.array([])
for train_index, val_index in kf:
    model.fit(X[train_index], y[train_index])
    preds = model.predict_proba(X[val_index])[:, 1]
    print('AUC ROC: ', metrics.roc_auc_score(y[val_index], preds))
    scores = np.hstack([scores, metrics.roc_auc_score(y[val_index], preds)])

print('mean: ', scores.mean(), 'std: ', scores.std())
preds = model.fit(X, y).predict_proba(extract_features(test_data)[:, indices])[:, 1]
models_preds = np.vstack([models_preds, preds]) if models_preds.size else preds

X = np.log2(1 + X)
pre = preprocessing.StandardScaler()
X = pre.fit_transform(X)

kf = cross_validation.StratifiedKFold(y, n_folds=10)
scores = np.array([])
batch_size = 32
nb_epoch = 30

for train_index, val_index in kf:
    model = Sequential()
    model.add(Dense(128, input_shape=(X.shape[1],), init='uniform'))
    model.add(Activation('relu'))
    model.add(Dense(256, init='uniform'))
    model.add(Activation('relu'))
    model.add(Dense(128, init='uniform'))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    opt = keras.optimizers.SGD(lr=0.001, momentum=0.9, decay=1e-6)
    model.compile(loss='binary_crossentropy', optimizer=opt, class_mode='binary')
    model.fit(X[train_index], y[train_index], batch_size=batch_size,
              nb_epoch=50, show_accuracy=False,
              verbose=0, class_weight={0: 9, 1: 1})
    preds = model.predict_proba(X[val_index])

    print('AUC ROC: ', metrics.roc_auc_score(y[val_index], preds))

X_test = pre.transform(np.log2(1 + extract_features(test_data)[:, indices]))
model = Sequential()
model.add(Dense(128, input_shape=(X.shape[1],), init='uniform'))
model.add(Activation('relu'))
model.add(Dense(256, init='uniform'))
model.add(Activation('relu'))
model.add(Dense(128, init='uniform'))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('sigmoid'))

opt = keras.optimizers.SGD(lr=0.001, momentum=0.9, decay=1e-6)
model.compile(loss='binary_crossentropy', optimizer=opt, class_mode='binary')

model.fit(X, y, batch_size=batch_size,
          nb_epoch=50, show_accuracy=False,
          verbose=0, class_weight={0: 9, 1: 1})
preds = model.predict_proba(X_test).ravel()

models_preds = np.vstack([models_preds, preds]) if models_preds.size else preds

result_preds = np.array([])
result_preds = np.sum(models_preds, axis=0)

for val in result_preds:
    val = logit(val)

print("SUM DONE")

# submission generating
test_data = test_data.drop('image_url', 1)
test_data['image_label'] = result_preds
test_data.to_csv(root_path + '/res.csv', index=False)  # -*- coding: utf-8 -*-
