import time

import numpy as np
import tensorflow as tf


class LowLevelExtractor(tf.keras.layers.Layer):
    def __init__(self, color_channel = 13, texture_channel = 16, texture_compositions = 16, out_channel = 32, name = "Extractor", **kwargs):
        super().__init__(name = name, **kwargs)
        self.color_channel = color_channel
        self.texture_channel = texture_channel
        self.texture_compositions = texture_compositions
        self.out_channel = out_channel
        
    def build(self, input_shape):
        print(self.name,input_shape)
        self.colors = tf.keras.layers.Convolution2D(self.color_channel,
                                                 1,
                                                 name="colors",
                                                 padding='SAME',
                                                 activation=tf.nn.leaky_relu,
                                                 kernel_initializer=tf.initializers.he_normal(),
                                                 bias_initializer=tf.initializers.he_uniform(),
                                                 kernel_regularizer=tf.keras.regularizers.l2(0.001),
                                                 dtype = self.dtype,
                                                 )
        self.textures = tf.keras.layers.DepthwiseConv2D(kernel_size = 3,depth_multiplier = self.texture_channel, name="textures", padding='SAME', activation=tf.nn.leaky_relu, depthwise_initializer=tf.initializers.he_normal(), bias_initializer=tf.initializers.he_uniform(), dtype = self.dtype)
        self.compositions11 = tf.keras.layers.Convolution2D(self.texture_compositions,
                                                         [1,9],
                                                         name="comp11",
                                                         padding='SAME',
                                                         activation=tf.nn.leaky_relu,
                                                         kernel_initializer=tf.initializers.he_normal(),
                                                         bias_initializer=tf.initializers.he_uniform(),
                                                         kernel_regularizer=tf.keras.regularizers.l2(0.001),
                                                         dtype = self.dtype,
                                                         )
        self.compositions12 = tf.keras.layers.Convolution2D(self.texture_compositions,
                                                         [9,1],
                                                         name="comp12",
                                                         padding='SAME',
                                                         activation=tf.nn.leaky_relu,
                                                         kernel_initializer=tf.initializers.he_normal(),
                                                         bias_initializer=tf.initializers.he_uniform(),
                                                         kernel_regularizer=tf.keras.regularizers.l2(0.001),
                                                         dtype = self.dtype,
                                                         )
        self.compositions21 = tf.keras.layers.Convolution2D(self.texture_compositions,
                                                         [9,1],
                                                         name="comp21",
                                                         padding='SAME',
                                                         activation=tf.nn.leaky_relu, 
                                                         kernel_initializer=tf.initializers.he_normal(),
                                                         bias_initializer=tf.initializers.he_uniform(),
                                                         kernel_regularizer=tf.keras.regularizers.l2(0.001),
                                                         dtype = self.dtype,
                                                         )
        self.compositions22 = tf.keras.layers.Convolution2D(self.texture_compositions,
                                                         [1,9],
                                                         name="comp22",
                                                         padding='SAME',
                                                         activation=tf.nn.leaky_relu,
                                                         kernel_initializer=tf.initializers.he_normal(),
                                                         bias_initializer=tf.initializers.he_uniform(),
                                                         kernel_regularizer=tf.keras.regularizers.l2(0.001),
                                                         dtype = self.dtype,
                                                         )
        self.compositions3 = tf.keras.layers.Convolution2D(self.texture_compositions,
                                                        1,
                                                        name="comp3",
                                                        padding='SAME',
                                                        activation=tf.nn.leaky_relu,
                                                        kernel_initializer=tf.initializers.he_normal(),
                                                        bias_initializer=tf.initializers.he_uniform(),
                                                        kernel_regularizer=tf.keras.regularizers.l2(0.001),
                                                        dtype = self.dtype,
                                                        )
        self.compress = tf.keras.layers.Convolution2D(self.out_channel,
                                                   1,
                                                   name="compress",
                                                   padding='SAME',
                                                   activation=tf.nn.leaky_relu,
                                                   kernel_initializer=tf.initializers.he_normal(),
                                                   bias_initializer=tf.initializers.he_uniform(),
                                                   kernel_regularizer=tf.keras.regularizers.l2(0.001),
                                                   dtype = self.dtype,
                                                   )

        super().build(input_shape)
    
    @tf.function(experimental_autograph_options=tf.autograph.experimental.Feature.ALL, experimental_relax_shapes=True)
    def call(self, inputs):
        standardized = inputs #= tf.image.per_image_standardization(inputs)
        colors = self.colors(standardized)
        colors = tf.keras.layers.concatenate([standardized, colors])
        textures = self.textures(colors)
        compositions1 = self.compositions11(textures)
        compositions1 = self.compositions12(compositions1)
        compositions2 = self.compositions21(textures)
        compositions2 = self.compositions22(compositions2)
        compositions3 = self.compositions3(textures)
        conc = tf.keras.layers.concatenate([colors, compositions3, compositions1, compositions2])
        compressed = self.compress(conc)
        return compressed


        
    def get_config(self):
        config = super().get_config()
        config.update({'color_channel': self.color_channel,
                       'texture_channel': self.texture_channel,
                       'texture_compositions': self.texture_compositions,
                       })
        return config

class FrustumScaler(tf.keras.layers.Layer):
    
    def __init__(self, distance_count = 10, image_hight0 = 480., distance_steps = 100., min_dist = 100., name = "FrustumScaler", **kwargs):
        super().__init__(name = name, **kwargs)
        self.distance_count = tf.cast(distance_count, dtype = self.dtype)
        self.image_hight0 = tf.cast(image_hight0, dtype = self.dtype)
        self.distance_steps = tf.cast(distance_steps, dtype = self.dtype)
        self.min_dist = tf.cast(min_dist, dtype = self.dtype)
    
    def build(self, input_shape):
        print(self.name,input_shape)
        self.max_dist = self.min_dist + self.distance_steps * self.distance_count
        super().build(input_shape)
    
    @tf.function(experimental_relax_shapes=True)
    def call(self, inputs):
        images, focal_length, crop_factor = inputs
        print("tracing", self.name, images.shape, focal_length.shape, crop_factor.shape)
        print(images.dtype, focal_length.dtype, crop_factor.dtype)
        
        image_size = tf.cast(tf.shape(images)[1:3], dtype=self.dtype)
        cropped_size = image_size * crop_factor
        rel_scale = image_size * (self.image_hight0 / cropped_size[1]) / (self.max_dist / focal_length)
        
        scales_arr = tf.TensorArray(dtype =self.dtype, size=tf.cast(self.distance_count, dtype=tf.int32),dynamic_size=False)
        for i in tf.range(self.distance_count, dtype = self.dtype):
            scale = rel_scale * ((self.min_dist + self.distance_steps * tf.cast(i, dtype =self.dtype)) / focal_length)
            scales_arr = scales_arr.write(tf.cast(i, dtype=tf.int32), scale)
        scales = scales_arr.stack()
        
        scales_list = tf.unstack(scales)
        sized_images = []
        for scale in scales_list:
            scale = tf.cast(scale + 0.5, dtype = tf.int32)
            tf.print("used img scale", scale)
            sized_image = tf.image.resize(images,scale)
            sized_images.append(sized_image)
        
        return sized_images

        
    def get_config(self):
        config = super().get_config()
        config.update({'distance_count': self.distance_count,
                       'min_dist': self.min_dist,
                       'image_hight0': self.image_hight0,
                       'distance_steps': self.distance_steps,
                       })
        return config


class ScaledFeatures(tf.keras.layers.Layer):
    
    def __init__(self, distance_count = 10, image_hight0 = 480., distance_steps = 100., min_dist = 100., name = "ScaledFeatures", **kwargs):
        super().__init__(name = name, **kwargs)
        self.distance_count = tf.cast(distance_count, dtype = self.dtype)
        self.image_hight0 = tf.cast(image_hight0, dtype = self.dtype)
        self.distance_steps = tf.cast(distance_steps, dtype = self.dtype)
        self.min_dist = tf.cast(min_dist, dtype = self.dtype)
        
    
    def build(self, input_shape):
        print(self.name,input_shape)
        self.extractor = LowLevelExtractor(dtype=self.dtype)
        self.scaler = FrustumScaler(min_dist = self.min_dist, distance_count=self.distance_count, distance_steps=self.distance_steps, image_hight0 = self.image_hight0, dtype=self.dtype)
        super().build(input_shape)
    
    @tf.function(experimental_autograph_options=tf.autograph.experimental.Feature.ALL, experimental_relax_shapes=True)
    def call(self, inputs):
        images, focal_length, crop_factor = inputs
        print("tracing",self.name,images.shape, focal_length.shape, crop_factor.shape)
        print(images.dtype, focal_length.dtype, crop_factor.dtype)
        
        feature = self.extractor(images)
        multiscale_feature = self.scaler([feature, focal_length, crop_factor])
        return multiscale_feature

    def get_config(self):
        config = super().get_config()
        config.update({'distance_count': self.distance_count,
                       'min_dist': self.min_dist,
                       'image_hight0': self.image_hight0,
                       'distance_steps': self.distance_steps,
                       })
        return config


def test(op, optimizer, **kwargs):
    @tf.function
    def run(inputs):
        with tf.GradientTape() as tape:
            tape.watch(inputs)
            outputs = op(inputs, **kwargs)
        g = tape.gradient(outputs, inputs)
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
    
    print("Extractor")
    inputs = tf.constant(100.,shape=[1,100,100,20])
    extractor = LowLevelExtractor()
    test_extractor= test(extractor, optimizer, training = True)
    out, g = test_extractor(inputs)
    print(out.shape)
    print("Extractor")
    
    print("FrustumScaler")
    images = tf.constant(100.,shape=[1,100,100,20])
    focal_length = tf.cast(25.,dtype =tf.float32)
    crop_factor = tf.cast(4.8,dtype =tf.float32)
    inputs = [images, focal_length, crop_factor]
    scaler = FrustumScaler()
    test_scaler= test(scaler, optimizer, training = True)
    out = test_scaler(inputs)
    for image in out[0]:
        print(image.shape)
    print("FrustumScaler")
    
    print("ScaledFeatures")
    images = tf.constant(100.,shape=[1,100,100,3])
    focal_length = tf.cast(25.,dtype =tf.float32)
    crop_factor = tf.cast(4.8,dtype =tf.float32)
    inputs = [images, focal_length, crop_factor]
    scaler = ScaledFeatures()
    test_scaler= test(scaler, optimizer, training = True)
    out = test_scaler(inputs)
    for image in out[0]:
        print(image.shape)
    print("ScaledFeatures")
  

    print("Extractor")
    inputs = tf.constant(100.,shape=[1,100,100,20])
    time_start = time.time()
    for i in range(20):
        out = test_extractor(inputs)
    time_end = time.time()
    print(time_end - time_start)
    
    

if __name__ == '__main__':
    main()