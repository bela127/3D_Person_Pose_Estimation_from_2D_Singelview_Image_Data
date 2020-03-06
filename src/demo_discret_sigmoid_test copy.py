import time

import tensorflow as tf

def diskret_sigmoid_test(inputs):
    activated = tf.minimum(tf.maximum(inputs/8.+0.5, 0.), 1.)
    act = tf.nn.sigmoid(inputs)
    return activated, act

for x in range(-8,8):
    print(x,": ",diskret_sigmoid_test(tf.cast(x,dtype=tf.float32)))


class DiscretSigmoid(tf.keras.layers.Layer):
    def __init__(self, name = "DiscretSigmoid", **kwargs):
        super().__init__(name = name, **kwargs)
    
    @tf.function
    def call(self, inputs, training=None, **kwargs):
        super().call(inputs, training=training, **kwargs)
        if training:
            activated = self.discret_sigmoid_training(inputs)
        else:
            activated = self.discret_sigmoid_infer(inputs)
        return activated

    @tf.custom_gradient
    def discret_sigmoid_training(self, inputs):
        activated = tf.minimum(tf.maximum(inputs/10.+0.5, 0.), 1.)

        def grad(dy):
            with tf.GradientTape() as tape:
                tape.watch(inputs)
                act = tf.nn.sigmoid(inputs)

            return dy*tape.gradient(act, inputs)
        return activated, grad

    @tf.custom_gradient
    def discret_sigmoid_infer(self, inputs):
        activated = tf.minimum(tf.maximum(inputs/6.+0.5, 0.), 1.)

        def grad(dy):
            with tf.GradientTape() as tape:
                tape.watch(inputs)
                act = tf.nn.sigmoid(inputs)

            return dy*tape.gradient(act, inputs)
        return activated, grad

discret_sigmoid = DiscretSigmoid()

@tf.function
def singel_step(inputs):
    with tf.GradientTape() as tape:
        tape.watch(inputs)
        yt = discret_sigmoid(inputs, training = True)
    gt = tape.gradient(yt, inputs)
    with tf.GradientTape() as tape:
        tape.watch(inputs)
        yi = discret_sigmoid(inputs, training = False)
    gi = tape.gradient(yi, inputs)
    return inputs, yt, gt, yi, gi

def main():
    time_start = time.time()
    timing_loop()
    time_end = time.time()
    print(time_end - time_start)

    test_loop()

@tf.function
def timing_loop():
    for i in tf.range(200):
        test_loop()
        
@tf.function
def test_loop():
    for x in tf.range(-8,8):
        inputs = tf.cast(x,dtype=tf.float32)
        x, yt, gt, yi, gi = singel_step(inputs)
        tf.print("IN: ",inputs," OUT: Training: ",yt," grad: ",gt," Inferenz: ",yi," grad: ",gi)



if __name__ == "__main__":
    main()
    