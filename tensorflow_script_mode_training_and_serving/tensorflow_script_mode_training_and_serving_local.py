import boto3
import numpy as np
import sagemaker
from sagemaker.tensorflow import TensorFlow
from sagemaker.local import LocalSession


sagemaker_session = LocalSession()
sagemaker_session.config = {'local': {'local_code': True}}

iam = boto3.client('iam', region_name='us-east-1')
role = iam.get_role(RoleName='AmazonSageMaker-ExecutionRole-20190829T190746')['Role']['Arn']

region = sagemaker_session.boto_session.region_name

mnist_estimator2 = TensorFlow(entry_point='mnist_tf2.py',
                             role=role,
                             train_instance_count=1,
                             train_instance_type='local',
                             framework_version='2.1.0',
                             py_version='py3',
                             distributions={'parameter_server': {'enabled': True}})

mnist_estimator2.fit("file:////Users/eitans/WorkDocs/dev/PycharmProjects/amazon-sagemaker-examples/mnist")

predictor2 = mnist_estimator2.deploy(initial_instance_count=1, instance_type='local')

train_data = np.load('train_data.npy')
train_labels = np.load('train_labels.npy')

predictions2 = predictor2.predict(train_data[:50])
for i in range(0, 50):
    prediction = predictions2['predictions'][i]
    label = train_labels[i]
    print('prediction is {}, label is {}, matched: {}'.format(prediction, label, prediction == label))

predictor2.delete_endpoint(predictor2.endpoint)
predictor2.delete_model()
