import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler


def binarize_columns(df):
    # binarizr region identifier as 0 or 1
    df[df.columns[0]] = df.iloc[:, 0].apply(lambda x: 1 if x == 'TUMOR' else 0)
    return df

def compute_median_values(data_files):
    # Read in first file to get the columns
    sample = pd.read_csv(data_files[0])
    num_cols = sample.shape[1] - 2  # Exclude the first two columns

    # Initialize array to store values for all files
    values = np.empty((0, num_cols))

    # Iterate over all files and extract values
    for data_file in data_files:
        # Read in data and skip first row (assumed to be header)
        df = pd.read_csv(data_file, skiprows=[0], usecols=range(2, num_cols+2))
        df = binarize_columns(df)
        values = np.concatenate((values, df.values), axis=0)
    
    # Compute median values for each column
    median_values = np.median(values, axis=0)
    median_dict = {sample.columns[i+2]: median_values[i] for i in range(num_cols)}
    return median_dict


def pad_missing_values(df, median_dict):
    # Replace missing values with median value for each column
    for col in df.columns:
        median_value = median_dict[col]
        df[col].fillna(median_value, inplace=True)
    return df


def min_max_normalize_features(df):

    for col in df.columns:
        min_value = df[col].min()
        max_value = df[col].max()
        
        # Check if values are not already in the range [0, 1]
        if (min_value < 0 or max_value > 1):
            df[col] = (df[col] - min_value) / (max_value - min_value)
        
    return df

def create_label_dict(label_dir):
    df = pd.read_csv(label_dir)
    column_names = df.columns  # Get the column names from the first row
    label_dict = {}
    for i in range(1, len(df)):
        name = df.iloc[i, 0]  # Get the sample name from the first column of the current row
        time = df.iloc[i, column_names.get_loc('ClinicalFeats.Survival.BCSS.YearsFromDx')]  # Get the time data from the 'ClinicalFeats.Survival.BCSS.YearsFromDx' column
        event = df.iloc[i, column_names.get_loc('ClinicalFeats.Survival.BCSS')]  # Get the event data from the 'ClinicalFeats.Survival.BCSS' column
        label_dict[name] = (time, event)
    return label_dict

def z_score_normalize_ragged_tensor(ragged_tensor, D):
    scaler = StandardScaler()

    # Store the lengths of the subtensors before flattening the ragged tensor
    lengths = [len(subtensor) for subtensor in ragged_tensor]

    # Flatten the ragged tensor and convert it to a 2D numpy array
    flat_data = ragged_tensor.to_tensor().numpy().reshape(-1, D)

    # Fit and transform the data using the StandardScaler
    normalized_data = scaler.fit_transform(flat_data)

    # Check if mean is close to 0 and standard deviation is close to 1
    mean = np.mean(normalized_data, axis=0)
    std_dev = np.std(normalized_data, axis=0)

    print("Mean:", mean)
    print("Standard Deviation:", std_dev)
    
    # Restore the original ragged structure
    normalized_data_ragged = []
    start_idx = 0
    for length in lengths:
        normalized_data_ragged.append(normalized_data[start_idx:start_idx+length])
        start_idx += length

    return tf.ragged.constant(normalized_data_ragged, ragged_rank=1, dtype=tf.float32)
