import time

import numpy as np
import tensorflow as tf
keras = tf.keras


class LowLevelExtractor(keras.layers.Layer):
    def __init__(self, color_channel = 13, texture_channel = 32, texture_compositions = 16, name = "Extractor", **kwargs):
        super().__init__(name = name, **kwargs)
        self.color_channel = color_channel
        self.texture_channel = texture_channel
        self.texture_compositions = texture_compositions
        
    def build(self, input_shape):
        print(self.name,input_shape)
        self.colors = keras.layers.Convolution2D(self.color_channel, 1, name="colors", padding='SAME', activation=tf.nn.leaky_relu, kernel_initializer=tf.initializers.he_normal(), bias_initializer=tf.initializers.he_uniform())
        self.textures = keras.layers.DepthwiseConv2D(kernel_size = 3,depth_multiplier = self.texture_channel, name="textures", padding='SAME', activation=tf.nn.leaky_relu, depthwise_initializer=tf.initializers.he_normal(), bias_initializer=tf.initializers.he_uniform())
        self.compositions11 = keras.layers.Convolution2D(self.texture_compositions, [1,9], name="comp11", padding='SAME', activation=tf.nn.leaky_relu, kernel_initializer=tf.initializers.he_normal(), bias_initializer=tf.initializers.he_uniform())
        self.compositions12 = keras.layers.Convolution2D(self.texture_compositions, [9,1], name="comp12", padding='SAME', activation=tf.nn.leaky_relu, kernel_initializer=tf.initializers.he_normal(), bias_initializer=tf.initializers.he_uniform())
        self.compositions21 = keras.layers.Convolution2D(self.texture_compositions, [9,1], name="comp21", padding='SAME', activation=tf.nn.leaky_relu, kernel_initializer=tf.initializers.he_normal(), bias_initializer=tf.initializers.he_uniform())
        self.compositions22 = keras.layers.Convolution2D(self.texture_compositions, [1,9], name="comp22", padding='SAME', activation=tf.nn.leaky_relu, kernel_initializer=tf.initializers.he_normal(), bias_initializer=tf.initializers.he_uniform())
        self.compositions3 = keras.layers.Convolution2D(self.texture_compositions, 1, name="comp3", padding='SAME', activation=tf.nn.leaky_relu, kernel_initializer=tf.initializers.he_normal(), bias_initializer=tf.initializers.he_uniform())

        super().build(input_shape)
    
    @tf.function
    def call(self, inputs):
        standardized = tf.image.per_image_standardization(inputs)
        colors = self.colors(standardized)
        colors = keras.layers.concatenate([standardized, colors])
        textures = self.textures(colors)
        compositions1 = self.compositions11(textures)
        compositions1 = self.compositions12(compositions1)
        compositions2 = self.compositions21(textures)
        compositions2 = self.compositions22(compositions2)
        compositions3 = self.compositions3(textures)
        out = keras.layers.concatenate([colors, compositions3, compositions1, compositions2])
        return out


        
    def get_config(self):
        config = super().get_config()
        config.update({'color_channel': self.color_channel,
                       'texture_channel': self.texture_channel,
                       'texture_compositions': self.texture_compositions,
                       })
        return config

class FrustumScaler(keras.layers.Layer):
    
    def __init__(self, distance_count = 10, image_hight0 = 480., distance_steps = 100., min_dist = 100., name = "FrustumScaler", **kwargs):
        super().__init__(name = name, **kwargs)
        self.distance_count = tf.cast(distance_count, dtype = tf.float32)
        self.image_hight0 = tf.cast(image_hight0, dtype = tf.float32)
        self.distance_steps = tf.cast(distance_steps, dtype = tf.float32)
        self.min_dist = tf.cast(min_dist, dtype = tf.float32)
        
    def build(self, input_shape):
        print(self.name,input_shape)
        self.max_dist = self.min_dist + self.distance_steps * self.distance_count
        super().build(input_shape)
    
    @tf.function
    def call(self, inputs):
        images, focal_length, crop_factor = inputs
        
        image_size = tf.cast(tf.shape(images)[1:3], dtype=tf.float32)
        cropped_size = image_size * crop_factor
        rel_scale = image_size * (self.image_hight0 / cropped_size[1]) / (self.max_dist / focal_length) 
        
        scales_arr = tf.TensorArray(dtype =tf.float32, size=tf.cast(self.distance_count, dtype=tf.int32),dynamic_size=False)
        for i in tf.range(self.distance_count):
            scale = rel_scale * ((self.min_dist + self.distance_steps * tf.cast(i, dtype =tf.float32)) / focal_length)
            scales_arr = scales_arr.write(tf.cast(i, dtype=tf.int32), scale)
        scales = scales_arr.stack()
        
        scales_list = tf.unstack(scales)
        sized_images = []
        for scale in scales_list:
            sized_images.append(tf.image.resize(images,tf.cast(scale + 0.5, dtype = tf.int32)))
        
        return sized_images

        
    def get_config(self):
        config = super().get_config()
        config.update({'distance_count': self.distance_count,
                       'min_dist': self.min_dist,
                       'image_hight0': self.image_hight0,
                       'distance_steps': self.distance_steps,
                       })
        return config


class ScaledFeatures(keras.layers.Layer):
    
    def __init__(self, distance_count = 10, image_hight0 = 480., distance_steps = 100., min_dist = 100., name = "ScaledFeatures", **kwargs):
        self.distance_count = tf.cast(distance_count, dtype = tf.float32)
        self.image_hight0 = tf.cast(image_hight0, dtype = tf.float32)
        self.distance_steps = tf.cast(distance_steps, dtype = tf.float32)
        self.min_dist = tf.cast(min_dist, dtype = tf.float32)
        super().__init__(name = name, **kwargs)
        
    def build(self, input_shape):
        print(self.name,input_shape)
        self.extractor = LowLevelExtractor()
        self.scaler = FrustumScaler(min_dist = self.min_dist, distance_count=self.distance_count, distance_steps=self.distance_steps, image_hight0 = self.image_hight0)
        super().build(input_shape)
    
    @tf.function
    def call(self, inputs):
        images, focal_length, crop_factor = inputs
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
    out = test_extractor(inputs)
    print(out)
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
        print(image.shape)#TODO check output shape
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