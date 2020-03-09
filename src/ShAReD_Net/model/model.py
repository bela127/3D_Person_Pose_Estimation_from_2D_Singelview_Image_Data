import time

import numpy as np
import tensorflow as tf
keras = tf.keras

from ShAReD_Net.model.modules.base import MultiscaleShAReD
from ShAReD_Net.model.modules.feature import ScaledFeatures

class MultiScaleFeatureModel(keras.layers.Layer):
    
    def __init__(self, name = "MultiScaleFeatureModel", **kwargs):
        super().__init__(name = name, **kwargs)
        
    def build(self, input_shape):
        self.low_level_extractor = ScaledFeatures(4)
        self.high_level_extractor = MultiscaleShAReD(2,2,16)
        super().build(input_shape)
    
    @tf.function
    def call(self, inputs):
        low_level_feature = self.low_level_extractor(inputs)
        high_level_feature = list(zip(low_level_feature,low_level_feature))
        detection_feature = self.high_level_extractor(high_level_feature)
        return detection_feature
        
    def get_config(self):
        config = super().get_config()
        return config



def test(op, optimizer, **kwargs):
    @tf.function
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
    for image1, image2 in out[0]:
        print(image1.shape, image2.shape)
    print("MultiScaleFeatureModel")
    
    time_start = time.time()
    for i in range(10):
        out = test_msf(inputs)
    time_end = time.time()
    print(time_end - time_start)
    
    

if __name__ == '__main__':
    main()