# coding: utf-8
import lightgbm as lgb
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


print('Loading data...')

data = load_boston()

X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.25, random_state=42)

print('X_train: {}'.format(X_train))
print('y_train: {}'.format(y_train))

trainX = pd.DataFrame(X_train, columns=data.feature_names)
trainX['target'] = y_train

testX = pd.DataFrame(X_test, columns=data.feature_names)
testX['target'] = y_test

local_train = './data/train/boston_train_new.csv'
local_test = './data/test/boston_test_new.csv'

trainX.to_csv(local_train, header=None, index=False)
testX.to_csv(local_test, header=None, index=False)

train_x = pd.read_csv(local_train).iloc[:, :-1]
# train_x = train_x.values
print('train_x: {}'.format(train_x))

train_y = pd.read_csv(local_train).iloc[ :, -1:]
# train_y = train_y.values
print('train_y: {}'.format(train_y))

test_x = pd.read_csv(local_test).iloc[:, :-1]
#test_x = test_x.values
print('test_x: {}'.format(test_x))

test_y = pd.read_csv(local_test).iloc[ :, -1:]
#test_y = test_y.values
print('test_y: {}'.format(test_y))

# create dataset for lightgbm
lgb_train = lgb.Dataset(train_x, train_y)
lgb_eval = lgb.Dataset(test_x, test_y, reference=lgb_train)
# lgb_train = lgb.Dataset(X_train, y_train)
# lgb_eval = lgb.Dataset(X_train, y_train, reference=lgb_train)

# specify your configurations as a dict
params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'l2', 'l1'},
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

print('Starting training...')
# train
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=20,
                valid_sets=lgb_eval,
                early_stopping_rounds=5)

print('Saving model...')
# save model to file
gbm.save_model('model.txt')

print('Starting predicting...')
# predict
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
# eval
print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)

saved_model = lgb.Booster(model_file='model.txt')
print('Starting saved_model predicting...')
# predict
y_saved_model_pred = saved_model.predict(X_test)
# eval
print('The rmse of prediction is:', mean_squared_error(y_test, y_saved_model_pred) ** 0.5)
