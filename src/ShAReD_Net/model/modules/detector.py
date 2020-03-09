import time

import numpy as np
import tensorflow as tf
keras = tf.keras


class PersonDetector(keras.layers.Layer):
    
    def __init__(self, name = "PersonDetector", **kwargs):
        super().__init__(name = name, **kwargs)
        
    def build(self, input_shape):
        self.compositions11 = keras.layers.Convolution2D(self.texture_compositions, [1,9], name="comp11", padding='SAME', activation=tf.nn.leaky_relu, kernel_initializer=tf.initializers.he_normal(), bias_initializer=tf.initializers.he_uniform())
        super().build(input_shape)
    
    def call(self, inputs):
        feature = self.extractor(inputs)
        feature = list(zip(feature,feature))
        detection_feature = self.detector(feature)
        return detection_feature
        
    def get_config(self):
        config = super().get_config()
        return config



def test(op, optimizer, **kwargs):
    def run(inputs):
        with tf.GradientTape() as tape:
            outputs = op(inputs, **kwargs)
        g = tape.gradient(outputs, op.trainable_variables)
        optimizer.apply_gradients(zip(g, op.trainable_variables))
        return outputs, g
    return run
    

def main():
    #tf.config.experimental_run_functions_eagerly(True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    print("MultiScaleFeatureModel")
    images = tf.constant(100.,shape=[1,100,100,3])
    focal_length = tf.cast(25., dtype=tf.float32)
    crop_factor = tf.cast(4.8, dtype=tf.float32)
    inputs = [images, focal_length, crop_factor]
    msf = MultiScaleFeatureModel()    
    test_msf= test(msf, optimizer, training = True)
    out = test_msf(inputs)
    print(msf.count_params())
    for image in out[0]:
        print(image.shape)
    print("MultiScaleFeatureModel")
    
    time_start = time.time()
    for i in range(10):
        out = test_msf(inputs)
    time_end = time.time()
    print(time_end - time_start)
    
    

if __name__ == '__main__':
    main()