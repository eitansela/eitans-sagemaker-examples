import boto3
import numpy as np
import sagemaker
from sagemaker.tensorflow import TensorFlow


sagemaker_session = sagemaker.Session()

iam = boto3.client('iam', region_name='us-east-1')
role = iam.get_role(RoleName='AmazonSageMaker-ExecutionRole-20190829T190746')['Role']['Arn']

region = sagemaker_session.boto_session.region_name

training_data_uri = 's3://sagemaker-sample-data-{}/tensorflow/mnist'.format(region)
print("training_data_uri: {}".format(training_data_uri))

mnist_estimator2 = TensorFlow(entry_point='mnist_tf2.py',
                             role=role,
                             train_instance_count=1,
                             train_instance_type='ml.p3.2xlarge',
                             framework_version='2.1.0',
                             py_version='py3',
                             distributions={'parameter_server': {'enabled': True}})

mnist_estimator2.fit(training_data_uri)

predictor2 = mnist_estimator2.deploy(initial_instance_count=1, instance_type='ml.p3.2xlarge')

#!aws --region us-east-1 s3 cp s3://sagemaker-sample-data-us-east-1/tensorflow/mnist/train_data.npy train_data.npy
#!aws --region us-east-1 s3 cp s3://sagemaker-sample-data-us-east-1/tensorflow/mnist/train_labels.npy train_labels.npy

train_data = np.load('train_data.npy')
train_labels = np.load('train_labels.npy')

predictions2 = predictor2.predict(train_data[:50])
for i in range(0, 50):
    prediction = predictions2['predictions'][i]
    label = train_labels[i]
    print('prediction is {}, label is {}, matched: {}'.format(prediction, label, prediction == label))

sagemaker.Session().delete_endpoint(predictor2.endpoint)