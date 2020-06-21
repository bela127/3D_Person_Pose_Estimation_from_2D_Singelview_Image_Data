import tensorflow as tf

def diskret_sigmoid_test(inputs):
    activated = tf.minimum(tf.maximum(inputs/8.+0.5, 0.), 1.)
    act = tf.nn.sigmoid(inputs)
    return activated, act

for x in range(-8,8):
    print(x,": ",diskret_sigmoid_test(tf.cast(x,dtype=tf.float32)))

@tf.custom_gradient
def discret_sigmoid(inputs):

    activated = tf.minimum(tf.maximum(inputs/8.+0.5, 0.), 1.)

    def grad(dy):
        with tf.GradientTape() as tape:
            tape.watch(inputs)
            act = tf.nn.sigmoid(inputs)

        return dy*tape.gradient(act, inputs)
    return activated, grad

for x in range(-8,8):
    inputs = tf.cast(x,dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(inputs)
        y = discret_sigmoid(inputs)
    g = tape.gradient(y, inputs)
    print(inputs.numpy(),": ",y.numpy()," g: ",g.numpy())

