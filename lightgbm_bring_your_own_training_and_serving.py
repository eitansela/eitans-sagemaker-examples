import boto3
import pandas as pd
import sagemaker
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sagemaker.estimator import Estimator
from sagemaker.predictor import csv_serializer


sagemaker_session = sagemaker.Session()

iam = boto3.client('iam', region_name='us-east-1')
role = iam.get_role(RoleName='AmazonSageMaker-ExecutionRole-20190829T190746')['Role']['Arn']
region = sagemaker_session.boto_session.region_name

prefix = 'DEMO-lightgbm-regression-boston'
print(sagemaker.__version__)

sess = sagemaker.session.Session()
bucket = sess.default_bucket()

print(bucket)

data = load_boston()

X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.25, random_state=42)

trainX = pd.DataFrame(X_train, columns=data.feature_names)
trainX['target'] = y_train

testX = pd.DataFrame(X_test, columns=data.feature_names)
testX['target'] = y_test

local_train = './data/boston_train.csv'
local_test = './data/boston_test.csv'

trainX.to_csv(local_train, header=None, index=False)
testX.to_csv(local_test, header=None, index=False)

# send data to S3. SageMaker will take training data from S3
train_location = sess.upload_data(
    path=local_train,
    bucket=bucket,
    key_prefix=prefix)
print(train_location)

test_location = sess.upload_data(
    path=local_test,
    bucket=bucket,
    key_prefix=prefix)
print(test_location)

account = sagemaker_session.boto_session.client('sts').get_caller_identity()['Account']
region = sagemaker_session.boto_session.region_name
image = '{}.dkr.ecr.{}.amazonaws.com/sagemaker-lightgbm-regression:latest'.format(account, region)
print(image)

local_regressor = Estimator(
    image,
    role,
    train_instance_count=1,
    train_instance_type="ml.m5.xlarge")

local_regressor.fit({'train':train_location, 'test': test_location}, logs=True)

predictor = local_regressor.deploy(1, 'ml.m5.xlarge', serializer=csv_serializer)

csv_predictions_file = './data/boston_test_no_target_feature.csv'
testX.drop(['target'], axis=1).to_csv(csv_predictions_file, header=False, index=False)

with open(csv_predictions_file, 'r') as f:
    payload = f.read().strip()

predicted = predictor.predict(payload).decode('utf-8')
print(predicted)

predictor.delete_endpoint(predictor.endpoint)
predictor.delete_model()