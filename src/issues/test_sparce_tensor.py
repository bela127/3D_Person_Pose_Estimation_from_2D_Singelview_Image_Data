import tensorflow as tf

indices = tf.cast([[1,1,1,1],[1,3,1,1]],dtype=tf.int64)
shape = tf.cast([4,4,4,4],dtype=tf.int64)

heat_map = tf.SparseTensor(indices = indices, values = tf.ones(tf.shape(indices)[0]), dense_shape = shape)

indices = tf.cast([[1,1,1,1],[1,3,1,1]],dtype=tf.int64)
shape = tf.cast([4,4,4,4],dtype=tf.int32)

heat_map = tf.SparseTensor(indices = indices, values = tf.ones(tf.shape(indices)[0]), dense_shape = shape)
