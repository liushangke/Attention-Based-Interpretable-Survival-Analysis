import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import KFold

import plotly.graph_objects as go
import plotly.subplots as sp
from plotly.subplots import make_subplots

from model import build_model_att, build_model_avg
from survivalnet2.losses import cox
from survivalnet2.metrics.concordance import HarrellsC
from survivalnet2.visualization import km_plot


def visualize_data_distribution(data, labels):
    # Flatten the data and create a DataFrame
    flattened_data = data.flat_values.numpy()
    data_df = pd.DataFrame(flattened_data)
    
    # Visualize the distribution of each feature
    n_features = data_df.shape[1]
    fig, axes = plt.subplots(n_features, 1, figsize=(10, 4 * n_features))

    for i in range(n_features):
        sns.kdeplot(data_df.iloc[:, i], ax=axes[i])
        axes[i].set_title(f"Feature {i+1} Distribution")
        axes[i].set_xlabel(f"Feature {i+1}")
        axes[i].set_ylabel("Density")

    plt.tight_layout()
    plt.show()

    # Visualize the distribution of labels
    labels_df = pd.DataFrame(labels.numpy(), columns=["Time", "Event"])
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    sns.histplot(labels_df["Time"], kde=True, ax=axes[0], bins=50)
    axes[0].set_title("Time Distribution")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Frequency")

    sns.histplot(labels_df["Event"], discrete=True, ax=axes[1], binwidth=1)
    axes[1].set_title("Event Distribution")
    axes[1].set_xlabel("Event")
    axes[1].set_ylabel("Frequency")
    axes[1].set_xticks([0, 1])

    plt.tight_layout()
    plt.show()


def perform_k_fold_cross_validation(data, labels, batch_size, n_splits, model_name, D):
    # Initialize KFold
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # List to store history objects
    histories = []

    for fold, (train_indices, val_indices) in enumerate(kfold.split(data)):
        print(f'Fold {fold + 1}/{n_splits}')

        if model_name == 'model_att':
            model = build_model_att(D)
        elif model_name == 'model_avg':
            model = build_model_avg(D)
        else:
            raise ValueError("Invalid model name.")

        model.compile(
            loss={"risk": cox},
            metrics={"risk": HarrellsC()},
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        )

        ds_train = create_dataset(data, labels, train_indices, batch_size)
        ds_val = create_dataset(data, labels, val_indices, batch_size)

        # Fit the model with early stopping
        history = model.fit(
            ds_train,
            validation_data=ds_val,
            epochs=100,
            verbose=0
        )

        # Append the history object to the list
        histories.append(history)

        # Evaluate and plot Kaplan-Meier curve after each fold
        val_data = tf.gather(data, val_indices, axis=0)
        val_labels = tf.gather(labels, val_indices, axis=0)
        evaluate_and_plot_kaplan_meier(model, val_data, val_labels, model_name)

        
    # Return the list of history objects
    return histories


def create_dataset(data, labels, indices, batch_size):
    # Create dataset from given indices
    ds_data = tf.data.Dataset.from_tensor_slices(tf.gather(data, indices, axis=0))
    ds_labels = tf.data.Dataset.from_tensor_slices(tf.gather(labels, indices, axis=0))
    ds = tf.data.Dataset.zip((ds_data, ds_labels))

    # ds = ds.shuffle(10000)
    ds = ds.batch(batch_size, drop_remainder=False)

    for i, batch in enumerate(ds):
        _, l = batch
        events = sum(l[:,1])
        if events < 1.:
            print(f"Warning, 0 events in batch {i}.")
    
    return ds

def evaluate_and_plot_kaplan_meier(model, test_data, test_labels, model_name):
    if model_name == 'model_att':
        risks, _ = model.predict(test_data)
    elif model_name == 'model_avg':
        risks = model.predict(test_data)
    else:
        raise ValueError("Invalid model_name. Choose from 'model_att' and 'model_avg'.")

    cindex = HarrellsC()
    print(f"{model_name} Testing c-index: {cindex(test_labels, risks):.3f}")

    risk_groups = np.squeeze(np.array(risks > np.median(risks), int)) + 1
    km_plot(
        np.array(test_labels),
        groups=risk_groups,
        xlabel="Time",
        ylabel="Survival probability",
        legend=["predicted low risk", "predicted high risk"],
    )


def plot_combined_training_metrics(histories, model_name, n_splits):
    fig, axs = plt.subplots(n_splits, 2, figsize=(12, n_splits*4))  # Adjust the size as per your requirements

    for i, history in enumerate(histories):
        epochs = list(range(1, len(history.history['loss']) + 1))

        # Set the metric key based on the model name
        metric_key = 'risk_harrellsc' if model_name == 'model_att' else 'harrellsc'

        # Plot training and validation loss
        axs[i, 0].plot(epochs, history.history['loss'], label='Train Loss')
        axs[i, 0].plot(epochs, history.history['val_loss'], label='Validation Loss')
        axs[i, 0].set_xlabel('Epoch')
        axs[i, 0].set_ylabel('Loss')
        axs[i, 0].set_title(f'{model_name} Fold {i+1} Training Loss')
        axs[i, 0].legend()

        # Plot Harrell's C for training and validation
        axs[i, 1].plot(epochs, history.history[metric_key], label=f"Train Harrell's C")
        axs[i, 1].plot(epochs, history.history[f'val_{metric_key}'], label=f"Validation Harrell's C")
        axs[i, 1].set_xlabel('Epoch')
        axs[i, 1].set_ylabel("Harrell's C")
        axs[i, 1].set_title(f'{model_name} Fold {i+1} Training Harrell\'s C')
        axs[i, 1].legend()

    plt.tight_layout()  # Adjusts the spaces between plots
    plt.show()
