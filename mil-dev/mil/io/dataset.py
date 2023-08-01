import tensorflow as tf
from mil.io.reader import read_record


def threshold(value, key="t", cond=lambda x: x >= 2.0):
    return tf.one_hot(tf.cast(cond(value[key]), tf.int32), depth=2)


def mil_datagen(files, variables, batch_size=1, structured=False):
    # build dataset and train
    train_ds = tf.data.TFRecordDataset(files, num_parallel_reads=1).shuffle(len(files))
    train_ds = train_ds.map(lambda x: read_record(x, variables, structured=structured))
    train_ds = train_ds.map(lambda x, y, z, _: (x, threshold(y, "i")[0]))
    # train_ds = train_ds.map(lambda x, y: (set_shape(x,D), y))
    train_ds = train_ds.batch(batch_size, drop_remainder=True).prefetch(1)
    train_ds = train_ds.apply(tf.data.experimental.assert_cardinality(len(files)))
    return train_ds
