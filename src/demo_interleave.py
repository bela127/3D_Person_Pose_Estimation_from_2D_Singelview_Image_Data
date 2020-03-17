import tensorflow as tf

x = tf.constant(1,shape=[2,5,5,3])
y = tf.constant(2,shape=[2,5,5,3])
print(x)

conc = tf.concat([x[...,tf.newaxis],y[...,tf.newaxis]], axis=4)
print(conc)
interleaved = tf.reshape(conc, [2,5,5,-1])
print(interleaved)

x = tf.constant(1,shape=[2,5,5,3])
y = tf.constant(2,shape=[2,5,5,6])
print(x)

y1 = y[...,:3]
y2 = y[...,3:]
conc = tf.concat([x[...,tf.newaxis],y1[...,tf.newaxis],y2[...,tf.newaxis]], axis=4)
print(conc)
interleaved = tf.reshape(conc, [2,5,5,-1])
print(interleaved)

x = tf.constant(1,shape=[2,5,5,3])
y = tf.constant(2,shape=[2,5,5,6])
print(x)

resh = tf.reshape(y,[2,5,5,3,-1])
conc = tf.concat([x[...,tf.newaxis],resh], axis=4)
print(conc)
interleaved = tf.reshape(conc, [2,5,5,-1])
print(interleaved)