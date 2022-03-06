import boto3
import pandas as pd
import numpy as np
import base64
import math


class FeatureStoreUtils:

    def __init__(self, **kwargs):
        boto_session = boto3.Session()
        sagemaker_client = boto3.client('sagemaker')
        self.featurestore_runtime = boto_session.client(service_name='sagemaker-featurestore-runtime')
        self.feature_group_name = feature_group_name = kwargs["FeatureGroupName"]
        self.feature_definitions = sagemaker_client.describe_feature_group(FeatureGroupName=feature_group_name)[
            "FeatureDefinitions"]

    @staticmethod
    def map_feature_name_value(record):
        result_dict = {}
        for feature in record:
            result_dict[feature["FeatureName"]] = [feature["ValueAsString"]]
        return result_dict

    @staticmethod
    def map_batch_feature_name_value(records, feature_definitions):
        result_dict = {}
        for feature in feature_definitions:
            result_dict[feature["FeatureName"]] = []

        for record in records:
            for feature in record["Record"]:
                result_dict[feature["FeatureName"]].append(feature["ValueAsString"])
        return result_dict

    def get_record_to_df(self, **kwargs):
        """Retrieves the latest records stored in the Online Store and returns a DataFrame."""
        record_identifier_value_as_string = str(kwargs["RecordIdentifierValueAsString"])
        response = self.featurestore_runtime.get_record(FeatureGroupName=self.feature_group_name,
                                                        RecordIdentifierValueAsString=record_identifier_value_as_string)
        if "Record" in response:
            record = response["Record"]
            record_as_dict = self.map_feature_name_value(record)
            df = pd.DataFrame(data=record_as_dict)
            return (df)
        else:
            return None

    def batch_get_records_to_df(self, **kwargs):
        """Retrieves a batch of Records from a single FeatureGroup OnlineStore and returns a DataFrame."""
        record_identifiers_value_as_string = kwargs["RecordIdentifiersValueAsString"]

        identifiers = []
        item = {}
        item["FeatureGroupName"] = self.feature_group_name
        item["RecordIdentifiersValueAsString"] = record_identifiers_value_as_string
        identifiers.append(item)

        batch_get_record_response = self.featurestore_runtime.batch_get_record(Identifiers=identifiers)
        if "Records" in batch_get_record_response:
            records = batch_get_record_response["Records"]
            records_as_dict = self.map_batch_feature_name_value(records, self.feature_definitions)
            df = pd.DataFrame(data=records_as_dict)
            return (df)
        else:
            return None


def cast_object_to_string(data_frame):
    """Cast object dtype to string. The SageMaker FeatureStore Python SDK will then map the string dtype to String feature type."""
    for label in data_frame.columns:
        if data_frame.dtypes[label] == "object":
            data_frame[label] = data_frame[label].astype("str").astype("string")

def encode_vector_features(df, vector_features_list):
    """"Encode vector features to base64."""
    for vector_feature in vector_features_list:
        df[vector_feature] = df[vector_feature].apply(lambda embeddings: base64.b64encode(embeddings))

def decode_vector_feature(feature):
    if not isinstance(feature, str) and math.isnan(feature):
        return None
    else:
        decoded_feature = base64.decodebytes(bytes(feature[2:-1].encode()))
        return (decoded_feature)

def decode_vector_features(df, vector_features_list):
    """"Decode vector features from base64."""
    for vector_feature in vector_features_list:
        df[vector_feature['name']] = df[vector_feature['name']].apply(decode_vector_feature)
        df[vector_feature['name']] = df[vector_feature['name']].apply(lambda x: np.frombuffer(x, dtype=vector_feature['type']) if x is not None else None)
