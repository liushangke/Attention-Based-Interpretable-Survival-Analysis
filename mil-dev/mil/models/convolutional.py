import tensorflow as tf


def dense_layer(x, units, activation, dropout, name, index):
    # single dense layer
    x = tf.keras.layers.Dense(
        units, activation=activation, name=name + "_dns" + str(index)
    )(x)

    # add dropout if necessary
    if dropout > 0.0:
        x = tf.keras.layers.Dropout(dropout + "_dp" + str(index))(x)

    return x


def conv_block(
    input,
    units,
    activation,
    dropout,
    kernelsize,
    name,
    index,
    transform=None,
    batchnorm=False,
    ragged=False,
):
    kwargs = {"use_bias": True, "kernel_initializer": "he_uniform"}
    # single or split dense layer
    if isinstance(activation, (tuple, list)):
        if len(activation) == 2:
            x = tf.cond(
                ragged,
                lambda: tf.keras.layers.Conv2D(
                    units,
                    kernel_size=(kernelsize, kernelsize),
                    strides=(1, 1),
                    padding="same",
                    name=name + "_conv_top" + str(index),
                    **kwargs,
                )(input.to_tensor()),
                lambda: tf.keras.layers.Conv2D(
                    units,
                    kernel_size=(kernelsize, kernelsize),
                    strides=(1, 1),
                    padding="same",
                    name=name + "_conv_top" + str(index),
                    **kwargs,
                )(input),
            )
            # add batch normalization if necessary
            if batchnorm:
                x = tf.keras.layers.BatchNormalization(
                    name=name + "_bn_top" + str(index)
                )(x)

            top = tf.keras.layers.Activation(
                activation[0], name=name + "_act_top" + str(index)
            )(x)

            x = tf.cond(
                ragged,
                lambda: tf.keras.layers.Conv2D(
                    units,
                    kernel_size=(kernelsize, kernelsize),
                    strides=(1, 1),
                    padding="same",
                    name=name + "_conv_bottom" + str(index),
                    **kwargs,
                )(input.to_tensor()),
                lambda: tf.keras.layers.Conv2D(
                    units,
                    kernel_size=(kernelsize, kernelsize),
                    strides=(1, 1),
                    padding="same",
                    name=name + "_conv_bottom" + str(index),
                    **kwargs,
                )(input),
            )
            # add batch normalization if necessary
            if batchnorm:
                x = tf.keras.layers.BatchNormalization(
                    name=name + "_bn_bottom" + str(index)
                )(x)

            bottom = tf.keras.layers.Activation(
                activation[1], name=name + "_act_bottom" + str(index)
            )(x)
            x = tf.keras.layers.Multiply()([top, bottom])
        else:
            raise ValueError("Split layers requires two activations.")
    else:
        x = tf.cond(
            ragged,
            lambda: tf.keras.layers.Conv2D(
                units,
                kernel_size=(kernelsize, kernelsize),
                strides=(1, 1),
                padding="same",
                name=name + "_conv" + str(index),
                **kwargs,
            )(input.to_tensor()),
            lambda: tf.keras.layers.Conv2D(
                units,
                kernel_size=(kernelsize, kernelsize),
                strides=(1, 1),
                padding="same",
                name=name + "_conv" + str(index),
                **kwargs,
            )(input),
        )

        # add batch normalization if necessary
        if batchnorm:
            x = tf.keras.layers.BatchNormalization(name=name + "_bn" + str(index))(x)

        x = tf.keras.layers.Activation(activation, name=name + "_act" + str(index))(x)

    # add dropout if necessary
    if dropout > 0.0:
        x = tf.keras.layers.Dropout(dropout, name=name + "_dp" + str(index))(x)

    if transform is not None:
        x = transform(x)

    return x


def convolutional_model(
    D,
    config={
        "backbone": {
            "units": [512, 256],  # backbone configuration
            "activations": ["relu", ("tanh", "sigmoid")],
            "dropout": [0.0, 0.0],
            "kernel_size": [
                1,
                1,
            ],  # When using a kernel size larger than 1, attention will be slightly diffused, and attention weights will no longer correspond to the original data since the neighbors are also taken into consideration.
        },
        "attention": 2
        * [
            {
                "units": [1],  # attention configuration
                "activations": ["relu"],
                "dropout": [0.0],
                "kernel_size": [3],
                "transform": tf.exp,
            }
        ],
        "task": {
            "units": [1],  # task configuration
            "activations": ["linear"],
            "dropout": [0.0],
        },
        "tap": 2,  # output of this layer feeds attention network (one-index)
        "pool": 1,  # attention weights used to pool this layer (one-index)
    },
    ragged=False,
):
    # create input layer
    inputs = tf.cond(
        ragged,
        lambda: tf.keras.Input([None, None, D], ragged=True),
        lambda: tf.keras.Input([None, None, D]),
    )

    # create lists to store inputs / outputs of backbone layers
    layers = [inputs]

    # build the backbone up to attention tap layer (output size: [1, h, w, channel])
    for i, (u, a, d, k) in enumerate(
        zip(
            config["backbone"]["units"][: config["tap"]],
            config["backbone"]["activations"][: config["tap"]],
            config["backbone"]["dropout"][: config["tap"]],
            config["backbone"]["kernel_size"][: config["tap"]],
        )
    ):
        if i == 0:
            x = conv_block(inputs, u, a, d, k, "backbone", i, ragged=ragged)
        else:
            x = conv_block(x, u, a, d, k, "backbone", i)
        layers.append(x)

    # build the attention networks (output size: [1, h, w, num_class])
    weights = []
    for i, att in enumerate(config["attention"]):
        for j, (u, a, d, k) in enumerate(
            zip(att["units"], att["activations"], att["dropout"], att["kernel_size"])
        ):
            if j == 0:
                x = conv_block(layers[-1], u, a, d, k, f"attention{i}", j)
            else:
                x = conv_block(x, u, a, d, k, f"attention{i}", j)

        # capture attention network outputs
        if "transform" in att.keys():
            weights.append(att["transform"](x))
        else:
            weights.append(att["transform"](x))

    # normalize weights (output size: [1, h, w, num_class])
    weights_norm = []
    total = tf.reduce_sum(tf.concat(weights, axis=3), axis=[1, 2])
    for k in range(len(weights)):
        weights_norm.append(
            tf.math.divide_no_nan(weights[k], total[0, k], name=f"attention_branch{k}")
        )

    # weighted average pooling (output size for each branch: [1, channel])
    reshaped_x = tf.reshape(
        layers[config["pool"]], [-1, layers[config["pool"]].shape[-1]]
    )
    pooled = []
    for n in weights_norm:
        n_reduced = tf.reshape(n, [-1, 1])
        pooled.append(
            tf.tensordot(tf.reshape(n_reduced, [1, -1]), reshaped_x, axes=1)
            / tf.reduce_sum(tf.reshape(n_reduced, [-1]))
        )

    # build remaining layers for prediction - multiclass branches
    predictions = []
    for i, p in enumerate(pooled):
        for j, (u, a, d) in enumerate(
            zip(
                config["task"]["units"],
                config["task"]["activations"],
                config["task"]["dropout"],
            )
        ):
            if j == 0:
                x = dense_layer(p, u, a, d, f"pooled_{i}", j)
            else:
                x = dense_layer(x, u, a, d, f"pooled_{i}", j)

        # capture outputs
        predictions.append(x)

    # combine outputs
    scores = tf.keras.layers.Concatenate(axis=1)(predictions)
    softmax = tf.keras.layers.Activation("softmax", name="softmax")(scores)

    return tf.keras.Model(
        inputs=layers[0],
        outputs=[
            softmax,
            tf.keras.layers.Concatenate(-1, name="attention_weights", trainable=False)(
                weights_norm
            ),
        ],
    )
