import tensorflow as tf


def kmeans_plus_plus_initialization(X, num_centroids, seed):
    # set random seed
    tf.random.set_seed(seed)

    # select the first centroid at random
    X = tf.convert_to_tensor(X, dtype=tf.float32)
    first_centroid_index = tf.random.uniform(
        shape=(), minval=0, maxval=X.shape[0], dtype=tf.int32, seed=seed
    )
    first_centroid_index.numpy()
    first_centroid = tf.gather(X, indices=first_centroid_index, axis=0)
    centroids = tf.expand_dims(first_centroid, axis=0)

    for i in range(num_centroids - 1):
        # compute squared distances from all points to the centroids
        X_squares = tf.reduce_sum(X**2, axis=1)
        X_squares_expanded = tf.expand_dims(X_squares, axis=1)
        X_squares_tiled = tf.repeat(X_squares_expanded, centroids.shape[0], axis=1)
        X_squares_tiled.shape

        centroid_squares = tf.reduce_sum(centroids**2, axis=1)
        centroid_squares_expanded = tf.expand_dims(centroid_squares, axis=0)
        centroid_squares_tiled = tf.repeat(
            centroid_squares_expanded, X.shape[0], axis=0
        )

        product = tf.linalg.matmul(X, tf.transpose(centroids))
        squared_distances = X_squares_tiled - (2 * product) + centroid_squares_tiled

        # compute additional centroid probabilities proportional to squared
        # distances
        min_squared_distances = tf.reduce_min(squared_distances, axis=1)
        p = min_squared_distances / tf.reduce_sum(min_squared_distances)

        # draw additional centroids
        p_expanded = tf.expand_dims(p, axis=0)
        new_centroid_index = tf.compat.v1.multinomial(
            p_expanded, 1, seed=seed, name=None, output_dtype=tf.int32
        )
        new_centroid_index = tf.squeeze(new_centroid_index)

        # compute concatenated centroids
        new_centroid = tf.gather(X, indices=new_centroid_index, axis=0)
        new_centroid_expanded = tf.expand_dims(new_centroid, axis=0)
        centroids = tf.concat([centroids, new_centroid_expanded], axis=0)

    return centroids
