import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import ensemble
from sklearn import preprocessing

train_data = pd.read_csv('train.csv')
labels = train_data['target']

test_data = pd.read_csv('test.csv')
id = test_data['ID']

train_features = train_data.drop(['ID', 'target'], axis=1)
test_features = test_data.drop(['ID'], axis=1)
for (train_name, train_ser), (test_name, test_ser) in zip(train_features.iteritems(), test_features.iteritems()):
    if train_ser.dtype == 'O':
        train_features[train_name], tmp_indexer = pd.factorize(train_features[train_name])
        test_features[test_name] = tmp_indexer.get_indexer(test_features[test_name])

imp = preprocessing.Imputer(missing_values='NaN', strategy='mean')
train_features = imp.fit_transform(train_features)
clf = ensemble.RandomForestClassifier(n_estimators=1500, n_jobs=1).fit(train_features, labels)
indices = np.argsort(clf.feature_importances_)[::-1]
indices = indices[:50]
DX = xgb.DMatrix(train_features[:, indices], label=labels, missing=float('NaN'))
params = {'booster': 'gbtree',
          'max_depth': 11,
          'eta': 0.01,
          'silent': 1,
          'objective': 'binary:logistic',
          'nthread': 1,
          'subsample': 0.96,
          'colsample_bytree': 0.45,
          'min_child_weight': 1,
          'eval_metric': 'logloss'
          }

xgb.cv(params=params, dtrain=DX, show_progress=True, nfold=5, num_boost_round=1500)

bst = xgb.Booster(params, [DX])
for i in range(1500):
    bst.update(DX, i)
    print("iteration: ", i)

bst.save_model('boost1')

test_features = imp.fit_transform(test_features)[:, indices]
preds = bst.predict(xgb.DMatrix(test_features, missing=float('NaN')))

d = {'ID': id, 'PredictedProb': preds}

df = pd.DataFrame(d)
df.to_csv('bnp_submission', index=False)
