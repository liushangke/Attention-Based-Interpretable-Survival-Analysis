import numpy as np
import os
import random
import tensorflow as tf

from preprocessing import create_label_dict, z_score_normalize_ragged_tensor
from dataloader import read_data
from utils import perform_k_fold_cross_validation, plot_combined_training_metrics

# Set random seeds for reproducibility
np.random.seed(51)
tf.random.set_seed(51)


# define parameters
D = 49
batch_size = 64


data_dir = '/Users/shangke/Desktop/pathology/raw_data/perSlideRegionFeatures/CPSII_40X'
label_dir = '/Users/shangke/Desktop/pathology/raw_data/perSlideRegionFeatures/FusedData_CPSII_40X.csv'
csv_names = os.listdir(data_dir)
null_count = 0
label_dict = create_label_dict(label_dir)
valid_csv_names = []

for name in csv_names:
    if name.rstrip('.csv') in list(label_dict.keys()):
        valid_csv_names.append(name)

    else:
        null_count += 1
        
# Shuffle the list of valid CSV names
random.shuffle(valid_csv_names)

data_files = [os.path.join(data_dir, str(csv_name)) for csv_name in valid_csv_names]

print(f"Number of samples with missing label data: {null_count}")

data, labels = read_data(data_files, label_dict)

data = z_score_normalize_ragged_tensor(data, D)


# After this thread, the dataset size now is reduced from 1655 to 1654
indices = []
for i, subject in enumerate(data):
    if not np.sum(np.isnan(subject)):
        indices.append(i)
data = tf.gather(data, np.array(indices), axis=0)
labels = tf.gather(labels, np.array(indices), axis=0)

# visualize_data_distribution(data, labels)

histories = perform_k_fold_cross_validation(data, labels, batch_size=64, n_splits=5, model_name='model_att', D=D)
plot_combined_training_metrics(histories, model_name='model_att', n_splits=5)

histories = perform_k_fold_cross_validation(data, labels, batch_size=64, n_splits=5, model_name='model_avg', D=D)
plot_combined_training_metrics(histories, model_name='model_avg', n_splits=5)
