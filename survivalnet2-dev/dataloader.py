import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import sys
 
import tensorflow as tf
import survivalnet2
from survivalnet2.data.labels import stack_labels, unstack_labels
from preprocessing import binarize_columns, pad_missing_values, compute_median_values


def read_data(data_files, label_dict):
    """
    Reads in the data files, binarizes the columns, pads missing values, and normalizes the features.

    Args:
        data_files (list): List of file paths to the data files.
        label_dict (dict): Dictionary mapping file names to (time, event) tuples.

    Returns:
        A tuple containing:
            - rows_tensor (tf.RaggedTensor): A ragged tensor containing the feature vectors for each sample.
            - labels_tensor (tuple): A tuple of two tensors, containing the time and event labels for each sample.
    """
    rows_list = []
    time_list = []
    event_list = []
    empty_count = 0

    # Calculate median values for each feature across all data files
    median_values = compute_median_values(data_files)

    for data_file in data_files:
        # Get the name of the file without the extension
        name = os.path.splitext(os.path.basename(data_file))[0]

        # Skip the file if it is not in the label dictionary
        if name not in label_dict:
            empty_count += 1
            continue

        # Read in the data file
        df = pd.read_csv(data_file)
        

        # Skip the file if it has no rows
        if df.shape[0] < 1:
            empty_count += 1
            print(name)
            continue

        # Binarize the columns and pad missing values
        df = df.iloc[:, 2:]  # Drop the first two columns
        df = binarize_columns(df)
        df = pad_missing_values(df, median_values)

        # Normalize the features
        # df = min_max_normalize_features(df)
    
        # Add the feature vector and labels to the lists
        rows_list.append(df.values)
        time, event = label_dict[name]
        time_list.append(time)
        event_list.append(event)
    
    # Convert the lists to tensors
    rows_tensor = tf.ragged.constant(rows_list, ragged_rank=1, dtype=tf.float32)
    labels_tensor = stack_labels(tf.convert_to_tensor(time_list, dtype=tf.float32),
                                 tf.convert_to_tensor(event_list, dtype=tf.float32))

    print(f"Number of samples: {len(rows_list)}")
    print(f"Number of empty data files: {empty_count}")

    return rows_tensor, labels_tensor

