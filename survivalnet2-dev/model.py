import tensorflow as tf


def build_model_att(D):
    # Input layer
    inputs = tf.keras.layers.Input(shape=(None, D), ragged=True)

    # Attention weights
    att = tf.keras.layers.Dense(units=1, activation="relu", name="att")(inputs)

    # Normalize weights to sum to 1
    totals = tf.reduce_sum(att, axis=1, name="att_total")
    normalized = tf.math.divide_no_nan(att, tf.expand_dims(totals, axis=1), name="normalized")

    # Use attention weights to calculate weighted sum of regions
    pooled = tf.linalg.matmul(normalized, inputs, transpose_a=True)

    # Remove the ragged dimension and reshape pooled tensor
    pooled = tf.squeeze(pooled.to_tensor(), axis=1)

    # Apply a linear layer to the pooled vector to generate the time and event risk values
    risk = tf.keras.layers.Dense(units=1, activation="linear", name="risk")(pooled)

    # Build the model
    model = tf.keras.models.Model(inputs=inputs, outputs=[risk, normalized])

    print(f"The input shape of model is: {model.input_shape}")
    print(f"The output shape of model is: {model.output_shape}")

    return model


def build_model_avg(D):
    # Input layer
    inputs = tf.keras.layers.Input(shape=(None, D), ragged=True)

    # Average feature vectors along the ragged dimension
    averaged = tf.reduce_mean(inputs, axis=1)

    # Apply a linear layer to the averaged vector to generate the time and event risk values
    risk = tf.keras.layers.Dense(units=1, activation="linear", name="risk")(averaged)

    # Build the model
    model = tf.keras.models.Model(inputs=inputs, outputs=risk)

    print(f"The input shape of model is: {model.input_shape}")
    print(f"The output shape of model is: {model.output_shape}")

    return model