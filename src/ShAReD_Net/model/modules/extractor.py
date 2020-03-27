import time

import numpy as np
import tensorflow as tf
keras = tf.keras

import ShAReD_Net.model.modules.base as module_base
import ShAReD_Net.model.modules.feature as feature
import ShAReD_Net.model.layer.aggregation as aggregation

class MultiScaleFeatureExtractor(keras.layers.Layer):
    
    def __init__(self, stages_count = 2, dense_blocks_count = 3, dense_filter_count = 16, distance_count = 5, image_hight0 = 480., distance_steps = 100., min_dist = 100., low_level_gpu = None, high_level_gpus = None, name = "MultiScaleFeatureExtractor", dtype = tf.float32, **kwargs):
        self.dense_blocks_count = dense_blocks_count
        self.dense_filter_count = dense_filter_count
        self.stages_count = stages_count
        
        self.distance_count = tf.cast(distance_count, dtype = tf.float32)
        self.image_hight0 = tf.cast(image_hight0, dtype = tf.float32)
        self.distance_steps = tf.cast(distance_steps, dtype = tf.float32)
        self.min_dist = tf.cast(min_dist, dtype = tf.float32)
        
        self.high_level_gpus = high_level_gpus
        self.low_level_gpu = low_level_gpu
        super().__init__(name = name,dtype=dtype, **kwargs)
        
    def build(self, input_shape):
        print(self.name,input_shape)
        self.low_level_extractor = feature.ScaledFeatures(min_dist = self.min_dist, distance_count=self.distance_count, distance_steps=self.distance_steps, image_hight0 = self.image_hight0)
        self.high_level_extractor = module_base.MultiscaleShAReD(self.stages_count,self.dense_blocks_count,self.dense_filter_count,gpus = self.high_level_gpus)
        self.interleave = aggregation.Interleave()
        super().build(input_shape)
    
    @tf.function(experimental_autograph_options=tf.autograph.experimental.Feature.ALL, experimental_relax_shapes=True)
    def call(self, inputs, training=None):
        if self.low_level_gpu:
            print("low_level using", self.low_level_gpu[0])
            with tf.device(self.low_level_gpu[0]):
                low_level_feature = self.low_level_extractor(inputs)
                high_level_input = list(zip(low_level_feature,low_level_feature))
        else:
            low_level_feature = self.low_level_extractor(inputs)
            high_level_input = list(zip(low_level_feature,low_level_feature))
            
        high_level_feature = self.high_level_extractor(high_level_input, training=training)
        
        if self.low_level_gpu:
            print("low_level using", self.low_level_gpu[0])
            with tf.device(self.low_level_gpu[0]):
                combined_high_level = []
                for feature_per_size in high_level_feature:
                    combined_per_size = self.interleave(feature_per_size)
                    combined_high_level.append(combined_per_size)
                feature3d = aggregation.combine3d(combined_high_level)
        else:
            combined_high_level = []
            for feature_per_size in high_level_feature:
                combined_per_size = self.interleave(feature_per_size)
                combined_high_level.append(combined_per_size)
            feature3d = aggregation.combine3d(combined_high_level)
            
        return feature3d
        
    def get_config(self):
        config = super().get_config()
        config.update({'distance_count': self.distance_count,
                       'min_dist': self.min_dist,
                       'image_hight0': self.image_hight0,
                       'distance_steps': self.distance_steps,
                       
                       'stages_count': self.stages_count,
                       'dense_blocks_count': self.dense_blocks_count,
                       'dense_filter_count': self.dense_filter_count,
                       })
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

    print("MultiScaleFeatureExtractor")
    images = tf.constant(100.,shape=[1,100,100,3])
    focal_length = tf.cast(25., dtype=tf.float32)
    crop_factor = tf.cast(4.8, dtype=tf.float32)
    inputs = [images, focal_length, crop_factor]
    msf = MultiScaleFeatureExtractor()    
    test_msf= test(msf, optimizer, training = True)
    out, g = test_msf(inputs)
    print(msf.count_params())
    print(out.shape)
    print("MultiScaleFeatureExtractor")
    
    time_start = time.time()
    for i in range(5):
        out = test_msf(inputs)
    time_end = time.time()
    print(time_end - time_start)
    
    

if __name__ == '__main__':
    main()