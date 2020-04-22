import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

import ShAReD_Net.model.modules.base as modules_base  
import ShAReD_Net.model.layer.base as layer_base
import ShAReD_Net.model.activation.base as activation_base
 
from ShAReD_Net.model.modules.feature import LowLevelExtractor
from ShAReD_Net.configure import config

import ShAReD_Net.model.layer.base as base_layer

class ShAReDHourGlass(keras.layers.Layer):
    def __init__(self, dense_blocks_count = 2, dense_filter_count = 48, name = "ShAReDHourGlass", **kwargs):
        super().__init__(name = name, **kwargs)
        self.dense_blocks_count = dense_blocks_count
        self.dense_filter_count = dense_filter_count
        
    def build(self, input_shape):
        print(self.name,input_shape)
        res_shape, shc_shape = input_shape
        self.big_shared1 = base_layer.ScaledShAReD(dense_blocks_count=self.dense_blocks_count, dense_filter_count=self.dense_filter_count, name="big_shared1")
        
        self.big_normal = base_layer.Scale(name="big_normal")
        
        self.normal_shared1 = base_layer.ScaledShAReD(dense_blocks_count=self.dense_blocks_count, dense_filter_count=self.dense_filter_count, name="normal_shared1")

        self.normal_medium = base_layer.Scale(name="normal_medium")
        
        self.medium_shared1 = base_layer.ScaledShAReD(dense_blocks_count=self.dense_blocks_count, dense_filter_count=self.dense_filter_count, name="medium_shared1")

        self.medium_small = base_layer.Scale(name="medium_small")

        self.small_shared1 = base_layer.ScaledShAReD(dense_blocks_count=self.dense_blocks_count, dense_filter_count=self.dense_filter_count, name="small_shared1")

        self.small_medium = base_layer.Scale(name="small_medium")

        self.medium_shared3 = base_layer.ScaledShAReD(dense_blocks_count=self.dense_blocks_count, dense_filter_count=self.dense_filter_count, name="medium_shared3")

        self.medium_normal = base_layer.Scale(name="medium_normal")

        self.normal_shared3 = base_layer.ScaledShAReD(dense_blocks_count=self.dense_blocks_count, dense_filter_count=self.dense_filter_count, name="normal_shared3")

        self.normal_big = base_layer.Scale(name="normal_big")

        self.big_shared3 = base_layer.ScaledShAReD(dense_blocks_count=self.dense_blocks_count, dense_filter_count=self.dense_filter_count, name="big_shared3")
        super().build(input_shape)
    
    @tf.function(experimental_autograph_options=tf.autograph.experimental.Feature.ALL, experimental_relax_shapes=True)
    def call(self, inputs, training=None):
        input_res, input_shc = inputs
        
        scale = tf.cast(tf.shape(input_shc)[1:3], dtype=tf.int32)
        scale_2 = tf.cast(scale/2, dtype=tf.int32)
        scale_4 = tf.cast(scale/4, dtype=tf.int32)
        scale_8 = tf.cast(scale/8, dtype=tf.int32)
        
        big_shared1 = self.big_shared1(inputs, training=training)
        big_shared2_res, big_shared2_shc = big_shared1
        
        big_normal = self.big_normal(big_shared2_shc, scale_2)
        
        normal_shared1 = self.normal_shared1([big_shared2_res, big_normal], training=training)
        normal_shared2_res, normal_shared2_shc = normal_shared1
        
        normal_medium = self.normal_medium(normal_shared2_shc, scale_4)
        
        medium_shared1 = self.medium_shared1([normal_shared2_res, normal_medium], training=training)
        medium_shared2_res, medium_shared2_shc = medium_shared1
        
        medium_small = self.medium_small(medium_shared2_shc,scale_8)
        
        small_shared1 = self.small_shared1([medium_shared2_res, medium_small], training=training)
        small_shared2_res, small_shared2_shc = small_shared1
        
        small_medium = self.small_medium(small_shared2_shc, scale_4)
        concat_medium = keras.layers.concatenate([small_medium, medium_shared2_shc])
                
        medium_shared3 = self.medium_shared3([small_shared2_res, concat_medium], training=training)
        medium_shared4_res, medium_shared4_shc = medium_shared3
        
        medium_normal = self.medium_normal(medium_shared4_shc, scale_2)
        concat_normal = keras.layers.concatenate([medium_normal, normal_shared2_shc])
        
        normal_shared3 = self.normal_shared3([medium_shared4_res, concat_normal], training=training)
        normal_shared4_res, normal_shared4_shc = normal_shared3
        
        normal_big = self.normal_big(normal_shared4_shc, scale)
        concat_big = keras.layers.concatenate([normal_big, big_shared2_shc])

        big_shared3 = self.big_shared3([normal_shared4_res, concat_big], training=training)
        
        return big_shared3
        
        
    def get_config(self):
        config = super().get_config()
        config.update({'dense_blocks_count': self.dense_blocks_count,
                       'dense_filter_count': self.dense_filter_count,
                       })
        return config 
    
class Encoder(keras.layers.Layer):
    def __init__(self, dense_blocks_count = 2, dense_filter_count = 48, name = "Encoder", **kwargs):
        super().__init__(name = name, **kwargs)
        self.dense_blocks_count = dense_blocks_count
        self.dense_filter_count = dense_filter_count

    
    def build(self, input_shape):
        print(self.name,input_shape)
        self.stage1 = ShAReDHourGlass(dense_blocks_count = self.dense_blocks_count, dense_filter_count = self.dense_filter_count, name= "stage1")
        self.stage2 = ShAReDHourGlass(dense_blocks_count = self.dense_blocks_count, dense_filter_count = self.dense_filter_count, name= "stage2")
        self.stage3 = ShAReDHourGlass(dense_blocks_count = self.dense_blocks_count, dense_filter_count = self.dense_filter_count, name= "stage3")
        
        destination_channel = input_shape[-1]
        destination_channel_2 = destination_channel
        destination_channel_4 = destination_channel * 2
        destination_channel_8 = destination_channel * 2

        self.scale2_res = layer_base.Scale(destination_channel_2, name= "scale_1_2_res")
        self.scale4_res = layer_base.Scale(destination_channel_4, name= "scale_1_4_res")
        self.scale8_res = layer_base.Scale(destination_channel_8, name= "scale_1_8_res")
                
        self.scale2_shc = layer_base.Scale(name= "scale_1_2_shc")
        self.scale4_shc = layer_base.Scale(name= "scale_1_4_shc")
        self.scale8_shc = layer_base.Scale(name= "scale_1_8_shc")
        
        super().build(input_shape)

    @tf.function
    def call(self, inputs, training=None):
        image = inputs
        
        size_1 = tf.cast(tf.shape(image)[1:3], dtype=tf.int32)
        size_2 = tf.cast(size_1/2, dtype=tf.int32)
        size_4 = tf.cast(size_1/4, dtype=tf.int32)
        size_8 = tf.cast(size_1/8, dtype=tf.int32)
        
        out1_res, out1_shc = self.stage1([image,image],training = training)
        scale1_res = self.scale2_res(out1_res, size_1)
        scale1_shc = self.scale2_shc(out1_shc, size_2)
        
        out2_res, out2_shc = self.stage2([scale1_res, scale1_shc],training = training)
        scale2_res = self.scale4_res(out2_res, size_2)
        scale2_shc = self.scale4_shc(out2_shc, size_4)
        
        out3_res, out3_shc = self.stage3([scale2_res, scale2_shc],training = training)
        scale3_res = self.scale8_res(out3_res, size_4)
        scale3_shc = self.scale8_shc(out3_shc, size_4)

        pos_decoder = [scale3_res, scale3_shc]
        pose_decoder = tf.concat([out3_res, out2_shc], axis = -1)
        
        return pose_decoder, pos_decoder
        
    def get_config(self):
        config = super().get_config()
        return config
    
    
class PosDecoder(keras.layers.Layer):
    def __init__(self, dense_blocks_count = 2, dense_filter_count = 48, name = "PosDecoder", **kwargs):
        super().__init__(name = name, **kwargs)
        self.dense_blocks_count = dense_blocks_count
        self.dense_filter_count = dense_filter_count

    
    def build(self, input_shape):
        print(self.name,input_shape)
        res_shape, shc_shape = input_shape
        self.self_ShAReD_1 = layer_base.SelfShAReD(dense_blocks_count = self.dense_blocks_count, dense_filter_count = self.dense_filter_count)

        self.stage = ShAReDHourGlass(dense_blocks_count = self.dense_blocks_count, dense_filter_count = self.dense_filter_count)
        
        self.scale2_res = layer_base.Scale()
        self.scale2_shc = layer_base.Scale()
        
        self.self_ShAReD_2 = layer_base.SelfShAReD(dense_blocks_count = self.dense_blocks_count, dense_filter_count = self.dense_filter_count)
        
        
        self.compress_output = keras.layers.Convolution2D(2,
                                             kernel_size=1,
                                             padding='SAME',
                                             activation=None,
                                             kernel_initializer=tf.initializers.glorot_normal(),
                                             bias_initializer=tf.initializers.glorot_uniform(),
                                             kernel_regularizer=tf.keras.regularizers.l2(config.training.regularization_rate),
                                             dtype=self.dtype,
                                             )
        
        super().build(input_shape)

    @tf.function
    def call(self, inputs, training=None):
        input_res, input_shc = inputs
        
        size = tf.cast(tf.shape(input_res)[1:3], dtype=tf.int32)
        size_2 = tf.cast(size/2, dtype=tf.int32)
        
        att_res_1, att_shc_1 = self.self_ShAReD_1([input_res, input_shc], training = training)

        
        stage_res, stage_shc = self.stage([att_res_1, att_shc_1], training = training)
        scale_res = self.scale2_res(stage_res, size_2)
        scale_shc = self.scale2_shc(stage_shc, size_2)
                
        att_res_2, att_shc_2 = self.self_ShAReD_2([scale_res, scale_shc], training = training)
        
        concat = tf.concat([att_res_2, att_shc_2], axis = -1)
        
        out = self.compress_output(concat)
        out = activation_base.discret_sigmoid(out, training = training)
        return out
        
    def get_config(self):
        config = super().get_config()
        return config
    
class PoseDecoder(keras.layers.Layer):
    def __init__(self, keypoints = 15, z_bins = 20, dense_blocks_count = 2, dense_filter_count = 48, name = "PosDecoder", **kwargs):
        super().__init__(name = name, **kwargs)
        self.keypoints = keypoints
        self.z_bins = z_bins
        self.dense_blocks_count = dense_blocks_count
        self.dense_filter_count = dense_filter_count

    
    def build(self, input_shape):
        print(self.name, input_shape)
        self.self_ShAReD_1 = layer_base.SelfShAReD(dense_blocks_count = self.dense_blocks_count, dense_filter_count = self.dense_filter_count)

        self.stage = ShAReDHourGlass(dense_blocks_count = self.dense_blocks_count, dense_filter_count = self.dense_filter_count)
        
        self.self_ShAReD_2 = layer_base.SelfShAReD(dense_blocks_count = self.dense_blocks_count, dense_filter_count = self.dense_filter_count)
        
        self.compress_feature = keras.layers.Convolution2D((self.keypoints + self.z_bins)*2,
                                             kernel_size=1,
                                             padding='SAME',
                                             activation=tf.nn.leaky_relu,
                                             kernel_initializer=tf.initializers.he_normal(),
                                             bias_initializer=tf.initializers.he_uniform(),
                                             kernel_regularizer=tf.keras.regularizers.l2(config.training.regularization_rate),
                                             dtype=self.dtype,
                                             )
        
        self.compress_output = keras.layers.Convolution2D(self.keypoints + self.z_bins,
                                             kernel_size=3,
                                             padding='SAME',
                                             activation=None,
                                             kernel_initializer=tf.initializers.glorot_normal(),
                                             bias_initializer=tf.initializers.glorot_uniform(),
                                             kernel_regularizer=tf.keras.regularizers.l2(config.training.regularization_rate),
                                             dtype=self.dtype,
                                             )
        
        self.pos_dep1 = PositionDependency()
                
        super().build(input_shape)

    @tf.function(experimental_relax_shapes=True)
    def call(self, inputs, training=None):
        
        #inputs = tf.debugging.check_numerics(inputs, "rois are invalid")
        
        att_res_1, att_shc_1 = self.self_ShAReD_1([inputs, inputs], training = training)
        
        #att_res_1 = tf.debugging.check_numerics(att_res_1, "pose decoder shared 1 res is invalid")
        #att_shc_1 = tf.debugging.check_numerics(att_shc_1, "pose decoder shared 1 shc is invalid")
        
        stage_res, stage_shc = self.stage([att_res_1, att_shc_1], training = training)
        
        #stage_res = tf.debugging.check_numerics(stage_res, "pose decoder stage_res is invalid")
        #stage_shc = tf.debugging.check_numerics(stage_shc, "pose decoder stage_res is invalid")
        
        stage_compressed = self.compress_feature(stage_res)
        
        #stage_compressed = tf.debugging.check_numerics(stage_compressed, "pose decoder stage_compressed is invalid")
        
        pos_dep1 = self.pos_dep1(stage_compressed)
        
        #pos_dep1 = tf.debugging.check_numerics(pos_dep1, "pose decoder pos_dep1 is invalid")
        
        att_res_2, att_shc_2 = self.self_ShAReD_2([pos_dep1, stage_shc], training = training)
        
        #att_res_2 = tf.debugging.check_numerics(att_res_2, "pose decoder shared 2 res is invalid")
        #att_shc_2 = tf.debugging.check_numerics(att_shc_2, "pose decoder shared 2 res is invalid")
                
        concat = tf.concat([att_res_2, att_shc_2], axis = -1)
        
        out = self.compress_output(concat)
        
        #out = tf.debugging.check_numerics(out, "pose decoder out is invalid")
        return out
        
    def get_config(self):
        config = super().get_config()
        return config
    
class  PositionDependency(keras.layers.Layer):
    def __init__(self, name = "PositionDependency", **kwargs):
        super().__init__(name = name, **kwargs)

    
    def build(self, input_shape):
        print(self.name,input_shape)
        self.original_shape = input_shape
        self.compress = keras.layers.Convolution2D(5,
                                            kernel_size=3,
                                            strides=3,
                                            padding='VALID',
                                            activation=tf.nn.relu,
                                            kernel_initializer=tf.initializers.he_normal(),
                                            bias_initializer=tf.initializers.he_uniform(),
                                             kernel_regularizer=tf.keras.regularizers.l2(config.training.regularization_rate),
                                            dtype=self.dtype,
                                           )
                
        self.flatt = tf.keras.layers.Flatten()
        
        self.combine = tf.keras.layers.Dense(input_shape[1] * input_shape[2] * 3, activation=tf.nn.relu, kernel_initializer=tf.initializers.he_normal())

        super().build(input_shape)

    @tf.function(experimental_relax_shapes=True)
    def call(self, inputs):
        
        compressed = self.compress(inputs)
        combined = self.combine(self.flatt(compressed))
        
        
        poss_dep = tf.reshape(combined, [-1,self.original_shape[1],self.original_shape[2],3])
        
        return tf.concat([poss_dep,inputs],axis = -1)
        
    def get_config(self):
        config = super().get_config()
        return config