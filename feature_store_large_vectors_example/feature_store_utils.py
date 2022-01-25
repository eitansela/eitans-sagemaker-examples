import boto3
import pandas as pd


boto_session = boto3.Session()
sagemaker_client = boto3.client('sagemaker')
featurestore_runtime = boto_session.client(service_name='sagemaker-featurestore-runtime')


def map_feature_name_value(record):
    result_dict = {}
    for feature in record:
        result_dict[feature["FeatureName"]] = [feature["ValueAsString"]]
    return result_dict


def map_batch_feature_name_value(records, feature_definitions):
    result_dict = {}
    for feature in feature_definitions:
        result_dict[feature["FeatureName"]] = []

    for record in records:
        for feature in record["Record"]:
            result_dict[feature["FeatureName"]].append(feature["ValueAsString"])
    return result_dict


# Retrieves the latest records stored in the OnlineStore and returns a DataFrame.
def get_record_to_df(**kwargs):
    feature_group_name = kwargs["FeatureGroupName"]
    record_identifier_value_as_string = str(kwargs["RecordIdentifierValueAsString"])
    response = featurestore_runtime.get_record(FeatureGroupName=feature_group_name, RecordIdentifierValueAsString=record_identifier_value_as_string)
    if "Record" in response:
        record = response["Record"]
        record_as_dict = map_feature_name_value(record)
        df = pd.DataFrame(data=record_as_dict)
        return(df)
    else:
        return None

    
# Retrieves a batch of Records from a single FeatureGroup OnlineStore and returns a DataFrame.
def batch_get_records_to_df(**kwargs):
    feature_group_name = kwargs["FeatureGroupName"]
    record_identifiers_value_as_string = kwargs["RecordIdentifiersValueAsString"]
    
    describe_feature_group_response = sagemaker_client.describe_feature_group(
        FeatureGroupName=feature_group_name,
    )
    feature_definitions = describe_feature_group_response["FeatureDefinitions"]
    
    identifiers = []
    item = {}
    item["FeatureGroupName"] = feature_group_name
    item["RecordIdentifiersValueAsString"] = record_identifiers_value_as_string
    identifiers.append(item)
    
    batch_get_record_response = featurestore_runtime.batch_get_record(Identifiers=identifiers)
    if "Records" in batch_get_record_response:
        records = batch_get_record_response["Records"]
        records_as_dict = map_batch_feature_name_value(records, feature_definitions)
        df = pd.DataFrame(data=records_as_dict)
        return(df)
    else:
        return None