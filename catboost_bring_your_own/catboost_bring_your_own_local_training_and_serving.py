# This is a sample training Python program that trains a simple CatBoost Regressor tree model.
# This implementation will work on your *local computer*.
#
# Prerequisites:
#   1. Install required Python packages:
#       pip install boto3 sagemaker 'sagemaker[local]' pandas scikit-learn
#   2. Docker Desktop has to be installed on your computer, and running.
#   3. Open terminal and run the following commands:
#       docker build  -t sagemaker-catboost-regressor-local container/.

import boto3
from sagemaker.local import LocalSession
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sagemaker.estimator import Estimator
from sagemaker.predictor import csv_serializer


sagemaker_session = LocalSession()
sagemaker_session.config = {'local': {'local_code': True}}

iam = boto3.client('iam', region_name='us-east-1')
role = iam.get_role(RoleName='AmazonSageMaker-ExecutionRole-20190829T190746')['Role']['Arn']
region = sagemaker_session.boto_session.region_name

data = load_boston()

X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.25, random_state=42)

trainX = pd.DataFrame(X_train, columns=data.feature_names)
trainX['target'] = y_train

testX = pd.DataFrame(X_test, columns=data.feature_names)
testX['target'] = y_test

local_train = './data/train/boston_train.csv'
local_test = './data/test/boston_test.csv'

trainX.to_csv(local_train)
testX.to_csv(local_test)

account = sagemaker_session.boto_session.client('sts').get_caller_identity()['Account']
region = sagemaker_session.boto_session.region_name
image = 'sagemaker-catboost-regressor-local'

local_regressor = Estimator(
    image,
    role,
    train_instance_count=1,
    train_instance_type="local",
    hyperparameters={'features': 'CRIM ZN INDUS CHAS NOX RM AGE DIS RAD TAX PTRATIO B LSTAT',
                     'target': 'target'})

train_location = 'file://./data/train/boston_train.csv'
test_location = 'file://./data/test/boston_test.csv'
local_regressor.fit({'train':train_location, 'test': test_location}, logs=True)

predictor = local_regressor.deploy(1, 'local', serializer=csv_serializer)

csv_predictions_file = './data/boston_test_set.csv'
testX.drop(['target'], axis=1).to_csv(csv_predictions_file, header=False)

with open(csv_predictions_file, 'r') as f:
    payload = f.read().strip()

predicted = predictor.predict(payload).decode('utf-8')
print(predicted)

predictor.delete_endpoint(predictor.endpoint)
predictor.delete_model()