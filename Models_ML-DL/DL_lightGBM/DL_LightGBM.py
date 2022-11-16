import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.style as style
import lightgbm as lgb
import random

from matplotlib import pyplot
from matplotlib.ticker import ScalarFormatter
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from lightgbm import LGBMClassifier

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

train_data = pd.get_dummies(train_data, columns=["color"], prefix = ["color"])
map_type = {"Ghoul":1, "Goblin":2, "Ghost":0}
train_data.loc[:, "type"] = train_data.type.map(map_type)
train_data = train_data.set_index('id')

X = train_data.drop(["type"],axis = 1)
y = train_data.type

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 42)

param_grid = {
    'class_weight': [None, 'balanced'],
    'boosting_type': ['gbdt', 'goss', 'dart'],
    'num_leaves': list(range(30, 150)),
    'learning_rate': [0.01,0.1,0.5],
    'subsample_for_bin': [20000,50000,100000,120000,150000],
    'min_child_samples': [20,50,100,200,500],
    'colsample_bytree': [0.6,0.8,1],
    "max_depth": [5,10,50,100]
}

lgbm = LGBMClassifier()
lgbm.fit(X_train, y_train)

lgbm_tuned = LGBMClassifier(
    boosting_type = 'gbdt',
    class_weight = None,
    min_child_samples = 20,
    num_leaves = 30,
    subsample_for_bin = 20000,
    learning_rate=0.01, 
    max_depth=10, 
    n_estimators=40, 
    colsample_bytree=0.6)

lgbm_tuned.fit(X_train, y_train)

y_test_pred = lgbm_tuned.predict(X_test)
score = round(accuracy_score(y_test, y_test_pred), 3)
print(score)

sns.set_context("talk")
style.use('fivethirtyeight')

fi = pd.DataFrame()
fi['features'] = X.columns.values.tolist()
fi['importance'] = lgbm_tuned.booster_.feature_importance(importance_type = 'gain')

sns.barplot(x = 'importance', y = 'features', data = fi.sort_values(by = 'importance', ascending = True))