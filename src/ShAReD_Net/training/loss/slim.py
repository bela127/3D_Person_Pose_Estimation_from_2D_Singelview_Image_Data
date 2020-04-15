import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

from ShAReD_Net.training.loss.base import PoseLoss

class DetectionLoss(keras.layers.Layer):
    def __init__(self, name = "DetectionLoss", **kwargs):
        super().__init__(name = name, **kwargs)

    
    def build(self, input_shape):
        print(self.name,input_shape)

        super().build(input_shape)

    @tf.function
    def call(self, inputs, training=None):
        pos_hm, (pos_hm_gt, loss_weights) = inputs
        
        se = (pos_hm - pos_hm_gt)**2
        weighted_loss = se * loss_weights
        nr_persons = tf.reduce_sum(pos_hm_gt, axis=[1,2,3])
        detection_loss = tf.reduce_sum(weighted_loss, axis=[1,2,3]) / nr_persons
        
        return detection_loss
        
    def get_config(self):
        config = super().get_config()
        return config
