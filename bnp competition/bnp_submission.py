import  pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import Imputer
from sklearn import ensemble

def extract_features(data):
    for(feature_name, feature_ser) in data.iteritems():
        if feature_ser.dtype == 'O':
            data[feature_name], tmp_index = pd.factorize(data[feature_name])


    return data.values


train_data = pd.read_csv('train.csv')
labels = train_data['target'].values
features = extract_features(train_data.drop(['ID','target'], axis=1))

test_data = pd.read_csv('test.csv')
id = test_data['ID']



imp = Imputer(missing_values='NaN', strategy='median')
imp_fetures = imp.fit_transform(features)

model_tmp = ensemble.RandomForestClassifier(n_estimators=2000, n_jobs=-1).fit(imp_fetures, labels)
indices = np.argsort(model_tmp.feature_importances_)[::-1]
indices = indices[:70]
X = imp_fetures[:, indices]

np.savetxt('features',X);
DX = xgb.DMatrix(X, label=labels)
params = {'booster':'gbtree',
     'max_depth':10,
     'eta':0.03,
     'silent':1,
     'objective':'binary:logistic',
     'nthread':4,
      'subsample': 0.7,
      'colsample_bytree': 0.9,
     'eval_metric':'logloss'
     }

xgb.cv(params=params, dtrain=DX, show_progress=True, nfold=5, num_boost_round=700)

bst = xgb.Booster(params, [DX])
for i in range(700):
    bst.update(DX, i)
    print("iteration: ", i)

bst.save_model('boost')


preds = bst.predict(xgb.DMatrix(imp.fit_transform(extract_features(test_data.drop(['ID'], axis=1)))[:,indices]))

d = {'ID': id, 'PredictedProb':preds}

df = pd.DataFrame(d)
df.to_csv('bnp_submission', index=False)
