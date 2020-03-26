import time

import numpy as np
import tensorflow as tf
keras = tf.keras

import ShAReD_Net.model.layer.base as base_layer


class PoseEstimator(keras.layers.Layer):
    
    def __init__(self, key_points = 15, depth_bins = 10 , xy_bins = [20,20], dense_blocks_count = 4, dense_filter_count = 32, target_gpu=None, name = "PoseEstimator", **kwargs):
        self.key_points = tf.cast(key_points, dtype = tf.int32)
        self.depth_bins = tf.cast(depth_bins, dtype = tf.int32)
        self.xy_bins = tf.cast(xy_bins, dtype = tf.int32)
        self.dense_blocks_count = dense_blocks_count
        self.dense_filter_count = dense_filter_count
        
        self.target_gpu = target_gpu
        super().__init__(name = name, **kwargs)
        
    @tf.Module.with_name_scope
    def build(self, input_shape):
        print(self.name,input_shape)
        output_depth = self.key_points + self.depth_bins
        
        self.self_shared1 = base_layer.SelfShAReD(dense_blocks_count=self.dense_blocks_count, dense_filter_count=self.dense_filter_count)
        self.scale_shc1 = base_layer.Scale(destination_channel=output_depth*3,new_shape=self.xy_bins * 2/3)
        self.scale_res1 = base_layer.Scale(destination_channel=output_depth*3/2,new_shape=self.xy_bins * 2/3)
        self.shared1 = base_layer.ShAReD(dense_blocks_count=self.dense_blocks_count, dense_filter_count=self.dense_filter_count)
        
        self.self_shared2 = base_layer.SelfShAReD(dense_blocks_count=self.dense_blocks_count, dense_filter_count=self.dense_filter_count)
        self.scale_shc2 = base_layer.Scale(destination_channel=output_depth*2,new_shape=self.xy_bins)
        self.scale_res2 = base_layer.Scale(destination_channel=output_depth,new_shape=self.xy_bins)
        self.shared2 = base_layer.ShAReD(dense_blocks_count=self.dense_blocks_count, dense_filter_count=self.dense_filter_count)
        
        self.shared3 = base_layer.ShAReD(dense_blocks_count=self.dense_blocks_count, dense_filter_count=self.dense_filter_count)
        super().build(input_shape)
    
    def call(self, inputs, training=None):
        if self.target_gpu:
            print("estimator using", self.target_gpu)
            with tf.device(self.target_gpu):
                return self._compute(inputs, training)
        else:
            return self._compute(inputs, training)
        
    @tf.function(experimental_autograph_options=tf.autograph.experimental.Feature.ALL, experimental_relax_shapes=True)
    @tf.Module.with_name_scope
    def _compute(self, inputs, training):
        self_shared1_res, self_shared1_shc = self.self_shared1([inputs,inputs], training=training)
        scaled_shc1 = self.scale_shc1(self_shared1_shc)
        scaled_res1 = self.scale_res1(self_shared1_res)
        shared1 = self.shared1([scaled_res1,scaled_shc1], training=training)
        
        self_shared2_res, self_shared2_shc = self.self_shared2(shared1, training=training)
        scaled_shc2 = self.scale_shc2(self_shared2_shc)
        scaled_res2 = self.scale_res2(self_shared2_res)
        shared2 = self.shared2([scaled_res2,scaled_shc2], training=training)
        
        shared3 = self.shared3(shared2, training=training)

        return shared3[0]
        
    def get_config(self):
        config = super().get_config()
        config.update({'dense_blocks_count': self.dense_blocks_count,
                       'dense_filter_count': self.dense_filter_count,
                       'xy_bins': self.xy_bins,
                       'depth_bins': self.depth_bins,
                       'key_points': self.key_points,
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

    print("PoseEstimator")
    inputs = tf.constant(100.,shape=[1,100,100,3])
    pe = PoseEstimator()    
    test_pe= test(pe, optimizer, training = True)
    out, g = test_pe(inputs)
    print(pe.count_params())
    print(out.shape)
    print("PoseEstimator")
    
    time_start = time.time()
    for i in range(10):
        out = test_pe(inputs)
    time_end = time.time()
    print(time_end - time_start)
    
    

if __name__ == '__main__':
    main()