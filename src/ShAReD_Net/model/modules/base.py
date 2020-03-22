import time

import numpy as np
import tensorflow as tf
keras = tf.keras

import ShAReD_Net.model.layer.base as base_layer

class ShAReDHourGlass(keras.layers.Layer):
    def __init__(self, dense_blocks_count = 2, dense_filter_count = 48, name = "ShAReDHourGlass", **kwargs):
        super().__init__(name = name, **kwargs)
        self.dense_blocks_count = dense_blocks_count
        self.dense_filter_count = dense_filter_count
        
    def build(self, input_shape):
        print(self.name,input_shape)
        res_shape, shc_shape = input_shape
        self.big_shared1 = base_layer.ShAReD(dense_blocks_count=self.dense_blocks_count, dense_filter_count=self.dense_filter_count)
        self.big_shared2 = base_layer.ShAReD(dense_blocks_count=self.dense_blocks_count, dense_filter_count=self.dense_filter_count)
        
        self.big_normal = base_layer.Scale()
        
        self.normal_shared1 = base_layer.ScaledShAReD(dense_blocks_count=self.dense_blocks_count, dense_filter_count=self.dense_filter_count)
        self.normal_shared2 = base_layer.ScaledShAReD(dense_blocks_count=self.dense_blocks_count, dense_filter_count=self.dense_filter_count)

        self.normal_medium = base_layer.Scale()
        
        self.medium_shared1 = base_layer.ScaledShAReD(dense_blocks_count=self.dense_blocks_count, dense_filter_count=self.dense_filter_count)
        self.medium_shared2 = base_layer.ScaledShAReD(dense_blocks_count=self.dense_blocks_count, dense_filter_count=self.dense_filter_count)

        self.medium_small = base_layer.Scale()

        self.small_shared1 = base_layer.ScaledShAReD(dense_blocks_count=self.dense_blocks_count, dense_filter_count=self.dense_filter_count)
        self.small_shared2 = base_layer.ScaledShAReD(dense_blocks_count=self.dense_blocks_count, dense_filter_count=self.dense_filter_count)

        self.small_medium = base_layer.Scale()

        self.medium_shared3 = base_layer.ScaledShAReD(dense_blocks_count=self.dense_blocks_count, dense_filter_count=self.dense_filter_count)
        self.medium_shared4 = base_layer.ScaledShAReD(dense_blocks_count=self.dense_blocks_count, dense_filter_count=self.dense_filter_count)

        self.medium_normal = base_layer.Scale()

        self.normal_shared3 = base_layer.ScaledShAReD(dense_blocks_count=self.dense_blocks_count, dense_filter_count=self.dense_filter_count)
        self.normal_shared4 = base_layer.ScaledShAReD(dense_blocks_count=self.dense_blocks_count, dense_filter_count=self.dense_filter_count)

        self.normal_big = base_layer.Scale()

        self.big_shared3 = base_layer.ShAReD(dense_blocks_count=self.dense_blocks_count, dense_filter_count=self.dense_filter_count)
        self.big_shared4 = base_layer.ShAReD(dense_blocks_count=self.dense_blocks_count, dense_filter_count=self.dense_filter_count)
        super().build(input_shape)
    
    @tf.function(experimental_autograph_options=tf.autograph.experimental.Feature.ALL, experimental_relax_shapes=True)
    def call(self, inputs, training=None):
        input_res, input_shc = inputs
        
        scale = tf.cast(tf.shape(input_shc)[1:3], dtype=tf.int32)
        scale_2 = tf.cast(scale/2, dtype=tf.int32)
        scale_4 = tf.cast(scale/4, dtype=tf.int32)
        scale_8 = tf.cast(scale/8, dtype=tf.int32)
        
        big_shared1 = self.big_shared1(inputs, training=training)
        big_shared2 = self.big_shared2(big_shared1, training=training)
        big_shared2_res, big_shared2_shc = big_shared2
        
        big_normal = self.big_normal(big_shared2_shc, scale_2)
        
        normal_shared1 = self.normal_shared1([big_shared2_res, big_normal], training=training)
        normal_shared2 = self.normal_shared2(normal_shared1, training=training)
        normal_shared2_res, normal_shared2_shc = normal_shared2
        
        normal_medium = self.normal_medium(normal_shared2_shc, scale_4)
        
        medium_shared1 = self.medium_shared1([normal_shared2_res, normal_medium], training=training)
        medium_shared2 = self.medium_shared2(medium_shared1, training=training)
        medium_shared2_res, medium_shared2_shc = medium_shared2
        
        medium_small = self.medium_small(medium_shared2_shc,scale_8)
        
        small_shared1 = self.small_shared1([medium_shared2_res, medium_small], training=training)
        small_shared2 = self.small_shared2(small_shared1, training=training)
        small_shared2_res, small_shared2_shc = small_shared2
        
        small_medium = self.small_medium(small_shared2_shc, scale_4)
        concat_medium = keras.layers.concatenate([small_medium, medium_shared2_shc])
                
        medium_shared3 = self.medium_shared3([small_shared2_res, concat_medium], training=training)
        medium_shared4 = self.medium_shared4(medium_shared3, training=training)
        medium_shared4_res, medium_shared4_shc = medium_shared4
        
        medium_normal = self.medium_normal(medium_shared4_shc, scale_2)
        concat_normal = keras.layers.concatenate([medium_normal, normal_shared2_shc])
        
        normal_shared3 = self.normal_shared3([medium_shared4_res, concat_normal], training=training)
        normal_shared4 = self.normal_shared4(normal_shared3, training=training)
        normal_shared4_res, normal_shared4_shc = normal_shared4
        
        normal_big = self.normal_big(normal_shared4_shc, scale)
        concat_big = keras.layers.concatenate([normal_big, big_shared2_shc])

        big_shared3 = self.big_shared3([normal_shared4_res, concat_big], training=training)
        big_shared4 = self.big_shared2(big_shared3, training=training)
        
        return big_shared4
        
        
    def get_config(self):
        config = super().get_config()
        config.update({'dense_blocks_count': self.dense_blocks_count,
                       'dense_filter_count': self.dense_filter_count,
                       })
        return config
    
class MultiscaleShAReDStage(keras.layers.Layer):
    def __init__(self, dense_blocks_count = 2, dense_filter_count = 48, name = "MultiscaleShAReDStage", **kwargs):
        super().__init__(name = name, **kwargs)
        self.dense_blocks_count = dense_blocks_count
        self.dense_filter_count = dense_filter_count
        
    def build(self, input_shape):
        print(self.name,input_shape)
        self.input_count = len(input_shape)
        self.hour_glass = ShAReDHourGlass(dense_blocks_count=self.dense_blocks_count, dense_filter_count=self.dense_filter_count)
        self.mix = base_layer.Mix()
        self.combine = list([base_layer.SelfShAReD(dense_blocks_count=self.dense_blocks_count, dense_filter_count=self.dense_filter_count) for i in range(self.input_count)])
        super().build(input_shape)
        
    @tf.function(experimental_autograph_options=tf.autograph.experimental.Feature.ALL, experimental_relax_shapes=True)
    def call(self, inputs, training=None):
        outs = []
        for ins in inputs: 
            outs.append(self.hour_glass(ins, training=training))
        outs_res = []
        for res, _ in outs: 
            outs_res.append(res)
        outs_shc = []
        for _, shc in outs: 
            outs_shc.append(shc)
            
        mixed = self.mix(outs_shc)
        packed = zip(outs_res, mixed)
        combine = zip(self.combine, packed)
        
        combined = []
        for comb, ins in combine: 
            combined.append(comb(ins, training=training))
        
        return combined
        
    def get_config(self):
        config = super().get_config()
        return config
    
class MultiscaleShAReD(keras.layers.Layer):
    def __init__(self, stages_count = 3, dense_blocks_count = 2, dense_filter_count = 48, name = "MultiscaleShAReD", **kwargs):
        super().__init__(name = name, **kwargs)
        self.dense_blocks_count = dense_blocks_count
        self.dense_filter_count = dense_filter_count
        self.stages_count = stages_count
        
    def build(self, input_shape):
        print(self.name,input_shape)
        self.stages = list([MultiscaleShAReDStage(dense_blocks_count=self.dense_blocks_count, dense_filter_count=self.dense_filter_count) for i in range(self.stages_count)])
        super().build(input_shape)
        
    @tf.function(experimental_autograph_options=tf.autograph.experimental.Feature.ALL, experimental_relax_shapes=True)
    def call(self, inputs, training=None):
        outs = inputs
        for stage in self.stages:
            outs = stage(outs, training=training)
        return outs
        
    def get_config(self):
        config = super().get_config()
        config.update({'stages_count': self.stages_count,
                       'dense_blocks_count': self.dense_blocks_count,
                       'dense_filter_count': self.dense_filter_count,
                       })
        return config

def test(op, optimizer, **kwargs):
    @tf.function
    def run(inputs):
        with tf.GradientTape() as tape:
            tape.watch(op.trainable_variables)
            outputs = op(inputs, **kwargs)
        g = tape.gradient(outputs, op.trainable_variables)
        optimizer.apply_gradients(zip(g, op.trainable_variables))
        return outputs, g
    return run
    

def main():
    #tf.config.experimental_run_functions_eagerly(True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    
    print("MultiscaleShAReDStage")
    inputs = list([(tf.constant(100.,shape=[1,i,i,20]),tf.constant(100.,shape=[1,i,i,30])) for i in range(50,125,25)])
    multiscale = MultiscaleShAReDStage()
    test_multiscale= test(multiscale, optimizer, training = True)
    out = test_multiscale(inputs)
    print(out)
    print("MultiscaleShAReDStage")
    
    
    time_start = time.time()
    for i in range(10):
        out = test_multiscale(inputs)
    time_end = time.time()
    print(time_end - time_start)
    
    print("MultiscaleShAReD")
    inputs = list([(tf.constant(100.,shape=[1,i,i,20]),tf.constant(100.,shape=[1,i,i,30])) for i in range(50,125,25)])
    multiscale = MultiscaleShAReD(2,3,16)
    test_multiscale= test(multiscale, optimizer, training = True)
    out = test_multiscale(inputs)
    print(out)
    print("MultiscaleShAReD")
    
    
    time_start = time.time()
    for i in range(10):
        out = test_multiscale(inputs)
    time_end = time.time()
    print(time_end - time_start)
    

if __name__ == '__main__':
    main()
    