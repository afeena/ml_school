import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import ensemble
from sklearn import preprocessing, feature_extraction, decomposition
from sklearn.base import TransformerMixin

train_data = pd.read_csv('train.csv', header=0)
labels = train_data['target']

test_data = pd.read_csv('test.csv', header=0)
id = test_data['ID']

train_features = train_data.drop(['ID', 'target', 'v8', 'v23', 'v25', 'v36', 'v37', 'v46',
                                  'v51', 'v53', 'v54', 'v63', 'v73', 'v81',
                                  'v82', 'v89', 'v92', 'v95', 'v105', 'v107',
                                  'v108', 'v109', 'v116', 'v117', 'v118',
                                  'v119', 'v123', 'v124', 'v128'], axis=1)

test_features = test_data.drop(['ID', 'v8', 'v23', 'v25', 'v36', 'v37', 'v46',
                                'v51', 'v53', 'v54', 'v63', 'v73', 'v81',
                                'v82', 'v89', 'v92', 'v95', 'v105', 'v107',
                                'v108', 'v109', 'v116', 'v117', 'v118',
                                'v119', 'v123', 'v124', 'v128'], axis=1)


class DataFrameImputer(TransformerMixin):
    def fit(self, X, y=None):
        self.fill = pd.Series([X[c].value_counts().index[0]
                               if X[c].dtype == np.dtype('O') else X[c].median() for c in X],
                              index=X.columns)
        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)


nonnumeric_columns = ['v3', 'v24', 'v30', 'v31', 'v47', 'v52', 'v56', 'v66', 'v71', 'v74', 'v75', 'v79', 'v91',
                      'v110', 'v112', 'v113', 'v125']

big_X = train_features.append(test_features)
# big_X_imputed = DataFrameImputer().fit_transform(big_X)


le = preprocessing.LabelEncoder()
for feature in nonnumeric_columns:
    big_X[feature].fillna('', inplace=True)
    big_X[feature] = le.fit_transform(big_X[feature])
    bool_vec = np.copy(big_X[feature].values)
    bool_vec[bool_vec > 0] = 1
    feature_name = feature + 'p'
    big_X[feature_name] = bool_vec
    big_X[feature] = big_X[feature].replace(0, -999)

alph  = {'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7,'H':8,'I':9,'J':10,'K':11,'L':12,'M':13,'N':14,'O':15,'P':16,'Q':17,'R':18,'S':19,'T':20,'U':21,'V':22,'W':23,'X':24,'Y':25,'Z':26}
big_X.fillna(-999, inplace=True)
new22 = np.zeros(big_X['v22'].shape[0])
for ind,feature in enumerate(big_X['v22']):
    if(type(feature) is str):
        val=0
        chars = list(feature)[::-1]
        for i,char in enumerate(chars):
            val+=pow(26,i)*alph[char]
        new22[ind]=val
big_X['v22']=new22

train_X = big_X[0:train_features.shape[0]].as_matrix()
test_X = big_X[train_features.shape[0]::].as_matrix()


clf = ensemble.RandomForestClassifier(n_estimators=1500, n_jobs=4).fit(train_X, labels)
print(clf.feature_importances_)
indices = np.argsort(clf.feature_importances_)[::-1]




DX = xgb.DMatrix(train_X, label=labels)
params = {'booster': 'gbtree',
          'max_depth': 11,
          'eta': 0.01,
          'silent': 1,
          'objective': 'binary:logistic',
          'nthread': 4,
          'subsample': 0.96,
          'colsample_bytree': 0.2,
          'min_child_weight': 1,
          'eval_metric': 'logloss'
          }

xgb.cv(params=params, dtrain=DX, show_progress=True, nfold=5, num_boost_round=1500)

bst = xgb.Booster(params, [DX])
for i in range(1500):
    bst.update(DX, i)
    print("iteration: ", i)


preds = bst.predict(xgb.DMatrix(test_X))

d = {'ID': id, 'PredictedProb': preds}

df = pd.DataFrame(d)
df.to_csv('bnp_submission', index=False)
