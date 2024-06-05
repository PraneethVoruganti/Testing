"""**************************************************************************
 Import necessary Modules and libraries
**************************************************************************"""
import joblib
import boto3
import os
import tarfile
import shutil
import json
import pandas as pd
import yaml

"""**************************************************************************"""
def package_and_upload_artifacts(s3_base_url, target_columns, config):
    """
    Packages base and fine-tuned models into a tarball and uploads to S3.

    Parameters:
    s3_base_url (str): The base URL of the S3 location where the tarball will be uploaded.
    feeder_ids (list): A list of feeder IDs for which fine-tuned models have been created.
    target_columns (list): A list of target column names for which models have been created.

    """
    # Create a base directory to store the packaged models
    base_directory = config['directories']['base_directory']
    os.makedirs(base_directory, exist_ok=True)
    tarball_path = os.path.join(base_directory, config['directories']['tarball_name'])

    with tarfile.open(tarball_path, 'w:gz') as tar:
        # Package base models
        for target in target_columns:
            base_model_path = os.path.join(base_directory, config['directories']['base_model_subdir'], target, config['directories']['version'])
            if os.path.exists(base_model_path):
                tar.add(base_model_path, arcname=os.path.join('base', target, config['directories']['version']))
            else:
                print(f"Base model path does not exist: {base_model_path}")

        # Package fine-tuned models
        for feeder_id in config['feeders']['fine_tune_feeder_ids']:
            for target in target_columns:
                ft_model_path = os.path.join(base_directory, config['directories']['fine_tuned_model_subdir'], f"{target}_{feeder_id}", config['directories']['version'])
                if os.path.exists(ft_model_path):
                    tar.add(ft_model_path, arcname=os.path.join('fine_tuned', f"{target}_{feeder_id}", config['directories']['version']))
                else:
                    print(f"Fine-tuned model path does not exist: {ft_model_path}")

    # Upload the tar.gz file to S3
    s3_url = s3_base_url + 'artifacts/' + config['directories']['tarball_name']
    upload_to_s3(tarball_path, s3_url)

"""**************************************************************************"""
def upload_to_s3(local_path, s3_url):
    """
    Uploads a file from local path to the specified S3 URL.

    Parameters:
    local_path (str): The local path of the file to upload.
    s3_url (str): The full S3 URL where the file will be uploaded.

    """
    s3_client = boto3.client('s3')
    bucket = s3_url.split('/')[2]
    key = '/'.join(s3_url.split('/')[3:])
    # Perform the upload
    s3_client.upload_file(local_path, bucket, key)
    print(f"Uploaded {local_path} to {s3_url}")

"""**************************************************************************"""
def save_model_and_artifacts(model,
artifacts,
target_col,
model_type,
base_directory,
feeder_id=None,
is_xgb=False):
    """
    Saves the model and its associated artifacts to disk.

    Parameters:
    model (Model): The trained model to save.
    artifacts (dict): A dictionary of additional artifacts to save alongside the model.
    target_col (str): The target variable the model is predicting.
    model_type (str): The type of model ('base' or 'fine_tuned').
    base_directory (str): The base directory where the model and artifacts will be saved.
    feeder_id (str, optional): The feeder ID associated with the fine-tuned model.

    """
    # Determine the directory path based on the model type and target
    model_directory = os.path.join(base_directory, model_type, f"{target_col}_{feeder_id}" if feeder_id else target_col)
    
    # Create the directory including the version subdirectory
    model_path = os.path.join(model_directory, '1')
    os.makedirs(model_path, exist_ok=True)

    # Save model
    if is_xgb:
        joblib.dump(model, os.path.join(model_path, 'model.pkl'))

    else:
        # Save the TensorFlow model using the Keras API
        model.save(model_path, save_format='tf')

    # Save other artifacts such as encoders or scalers
    for artifact_name, artifact in artifacts.items():
        artifact_path = os.path.join(model_path, f'{artifact_name}.pkl')
        # Dump the artifact using joblib
        joblib.dump(artifact, artifact_path)

"""**************************************************************************"""
def save_baseline_data_and_predictions(X_val, y_val, y_pred, target, model_type='base', feeder_id=None):
    """
    Saves validation features, actual values, and predictions to a CSV file.

    Parameters:
    X_val (pd.DataFrame): The validation features.
    y_val (pd.Series): The actual target values for the validation data.
    y_pred (np.ndarray): The predicted values from the model.
    target (str): The target variable name.
    model_type (str): The type of model ('base' or 'fine_tuned').
    feeder_id (str, optional): The feeder ID for the fine-tuned model.

    Returns:
    str: The file path of the saved CSV.
    
    """
    # Create a metrics directory within the multi-model directory
    base_directory = './multi/metrics'
    os.makedirs(base_directory, exist_ok=True)
    
    # Define the filename based on the model type and target
    filename = f"{base_directory}/{model_type}_data_predictions_{target}{f'_{feeder_id}' if feeder_id else ''}.csv"
    
    # Combine the validation features, actual values, and predictions into a DataFrame
    df_val = pd.DataFrame(X_val)
    df_val['Actual'] = y_val
    df_val['Predicted'] = y_pred.flatten()  # Flatten the predictions if they are in an array
    
    # Save the DataFrame to a CSV file
    df_val.to_csv(filename, index=False)
    print(f"Saved {model_type} data and predictions for {target} to {filename}")
    
    return filename


"""**************************************************************************"""
def create_manifest(s3_files, s3_base_url):
    manifest = {
        'fileLocations': [
            {
                'URIs': s3_files
            }
        ],
        'globalUploadSettings': {
            'format': 'CSV',
            'delimiter': ',',
            'textqualifier': '"',
            'containsHeader': 'true'
        }
    }
    manifest_path = f'{s3_base_url}manifest/manifest.json'
    with open('manifest.json', 'w') as f:
        json.dump(manifest, f)
    return manifest_path

"""**************************************************************************"""
def list_files_in_s3_folder(s3_path):
    """
    This function list all files within a folder in a specific bucket in s3

    :param  s3_path: the s3 path including bucket name and prefixes (folders)

    :return return all files inside given folder_name
    """
    import boto3
    # Parse the S3 path to extract the bucket name and folder name
    s3_path_parts = s3_path.replace("s3://", "").split('/')
    bucket_name = s3_path_parts[0]
    folder_name = '/'.join(s3_path_parts[1:])

    # Create a boto3 client for S3
    s3_client = boto3.client('s3')
    
    # List objects in the specified folder
    response = s3_client.list_objects_v2(
        Bucket=bucket_name,
        Prefix=folder_name
    )
    
    # Extract file names from the response
    file_names = []
    for obj in response.get('Contents', []):
        file_names.append(f"s3://{bucket_name}/" + obj['Key'])
    
    return file_names
"""**************************************************************************"""

