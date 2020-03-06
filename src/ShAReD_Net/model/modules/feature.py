import time

import numpy as np
import tensorflow as tf
keras = tf.keras


class Extractor(keras.layers.Layer):
    def __init__(self, color_channel = 13, texture_channel = 32, texture_compositions = 16, name = "Extractor", **kwargs):
        super().__init__(name = name, **kwargs)
        self.color_channel = color_channel
        self.texture_channel = texture_channel
        self.texture_compositions = texture_compositions
        
    def build(self, input_shape):
        self.colors = keras.layers.Convolution2D(self.color_channel, 1, name="colors", padding='SAME', activation=None, kernel_initializer=tf.initializers.RandomNormal(), bias_initializer=tf.initializers.RandomUniform())
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
    
    def __init__(self, distance_count = 10, focal_length0 = [50.,50.], image_size0 = [480.,480.], scaling_steps = 0.1, name = "FrustumScaler", **kwargs):
        super().__init__(name = name, **kwargs)
        self.distance_count = tf.cast(distance_count, dtype = tf.float32)
        self.focal_length0 = tf.cast(focal_length0, dtype = tf.float32)
        self.image_size0 = tf.cast(image_size0, dtype = tf.float32)
        self.scaling_steps = tf.cast(scaling_steps, dtype = tf.float32)

        
    def build(self, input_shape):
        super().build(input_shape)
    
    def call(self, inputs):
        images, focal_length, crop_factor = inputs
        sized_images = []
        image_size = tf.cast(tf.shape(images)[1:3], dtype=tf.float32)
        cropped_size = image_size * crop_factor
        zero_scale = image_size * (self.image_size0 / cropped_size) * (self.focal_length0 / focal_length)
                                                                                              
        for i in tf.range(self.distance_count):
            scale = zero_scale * (1.0 - self.scaling_steps * i)
            sized_image = tf.image.resize(images, tf.cast(scale + 0.5, dtype = tf.int32))
            sized_images.append(sized_image)

        return sized_images


        
    def get_config(self):
        config = super().get_config()
        config.update({'distance_count': self.distance_count,
                       'focal_length0': self.focal_length0,
                       'image_size0': self.image_size0,
                       'scaling_steps': self.scaling_steps,
                       })
        return config

class ScaledFeatures(keras.layers.Layer):
    
    def __init__(self, distance_count = 10, scaling_steps = 0.1, name = "ScaledFeatures", **kwargs):
        self.distance_count = distance_count
        self.scaling_steps = scaling_steps
        super().__init__(name = name, **kwargs)
        
    def build(self, input_shape):
        self.extractor = Extractor()
        self.scaler = FrustumScaler(distance_count=self.distance_count, scaling_steps=self.scaling_steps)
        super().build(input_shape)
    
    def call(self, inputs):
        images, focal_length, crop_factor = inputs
        feature = self.extractor(images)
        multiscale_feature = self.scaler([feature, focal_length, crop_factor])
        return multiscale_feature


        
    def get_config(self):
        config = super().get_config()
        return config


def test(op, optimizer, **kwargs):
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

    print("FrustumScaler")
    images = tf.constant(100.,shape=[1,100,100,20])
    focal_length = 25.
    crop_factor = 4.8
    inputs = [images, focal_length, crop_factor]
    scaler = FrustumScaler()
    test_scaler= test(scaler, optimizer, training = True)
    out = test_scaler(inputs)
    for image in out[0]:
        print(image.shape)
    print("FrustumScaler")
    
    print("Extractor")
    inputs = tf.constant(100.,shape=[1,100,100,20])
    extractor = Extractor()
    test_extractor= test(extractor, optimizer, training = True)
    out = test_extractor(inputs)
    print(out)
    print("Extractor")
  
    
    time_start = time.time()
    for i in range(20):
        out = test_extractor(inputs)
    time_end = time.time()
    print(time_end - time_start)
    
    

if __name__ == '__main__':
    main()