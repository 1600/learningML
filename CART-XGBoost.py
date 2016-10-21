
# coding: utf-8

# In[ ]:


import pandas as pd
import subprocess
import pydotplus
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import matplotlib.pylab as plt

# 数据导入
xls_file = pd.ExcelFile('all.xls')
# print(xls_file.sheet_names)
df = xls_file.parse('sheet1')
df_target = df['result']


# In[ ]:

# 对范畴类变量进行编码
for column in ['uristem','uriquery','http_status']:
    dummies = pd.get_dummies(df[column])
    df[dummies.columns]=dummies

# 去掉已编码的原属性
df.drop(['domain','uristem','uriquery','http_status'], axis=1, inplace=True)
#df.ix[:,df.columns!=999]


# In[ ]:

train, test = train_test_split(df, test_size = 0.2)   # doing split of training and testing
train_data, train_target= train.ix[:, train.columns != 'result'], train['result']
test_data, test_target = test.ix[:, test.columns != 'result'], test['result']
print(train_data.shape)
print(train_target.shape)


# In[ ]:

# XGBoost

cv_params = {'max_depth': [3,5,7], 'min_child_weight': [1,3,5]}
ind_params = {'learning_rate': 0.1, 'n_estimators': 1000, 'seed':0, 'subsample': 0.8, 'colsample_bytree': 0.8, 
             'objective': 'binary:logistic'}
optimized_GBM = GridSearchCV(xgb.XGBClassifier(**ind_params), 
                            cv_params, 
                             scoring = 'accuracy', cv = 5, n_jobs = -1)

optimized_GBM.fit(train_data, train_target)


# In[ ]:

# 交叉验证
train_scores = cross_val_score(optimized_GBM , test_data, test_target, cv=7,scoring='accuracy')
print(train_scores)
#test_scores = cross_val_score(clf, test_data, test_target, cv=5,scoring='accuracy')
#print(scores)


# In[ ]:



