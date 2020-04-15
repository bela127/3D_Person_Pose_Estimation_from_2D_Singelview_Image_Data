import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

class SlimInferenzModel(keras.layers.Layer):
    def __init__(self, low_level_extractor, encoder, pos_decoder, pose_decoder, name = "SlimInferenzModel", **kwargs):
        super().__init__(name = name, **kwargs)
        self.low_level_extractor = low_level_extractor
        self.encoder = encoder
        self.pos_decoder = pos_decoder
        self.pose_decoder = pose_decoder

    
    def build(self, input_shape):
        print(self.name,input_shape)
        super().build(input_shape)

    @tf.function
    def call(self, inputs, training=None):
        image = inputs

        return
        
    def get_config(self):
        config = super().get_config()
        return config