import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

import ShAReD_Net.training.slim as slim_training
import ShAReD_Net.model.modules.slim as slim_modules


from ShAReD_Net.configure import config

class SlimEvalModel(keras.layers.Layer):
    def __init__(self, low_level_extractor, encoder, pos_decoder, pose_decoder, name = "SlimEvalModel", **kwargs):
        super().__init__(name = name, **kwargs)
        self.low_level_extractor = low_level_extractor
        self.encoder = encoder
        self.pos_decoder = pos_decoder
        self.pose_decoder = pose_decoder

    
    def build(self, input_shape):
        print(self.name, input_shape)
        roi_size = np.asarray(config.model.roi_size)
        key_points = config.model.output.keypoints
        
        self.roi_extractor = slim_modules.Roi_Extractor(roi_size=roi_size)
        
        self.pose_extractor = slim_training.SlimTrainPoseExtractor()
        
        
        self.call = tf.function(self.call,input_signature=[(tf.TensorSpec(shape=[None,None,None,3], dtype=tf.float32),
                                                           tf.TensorSpec(shape=[None,3], dtype=tf.int32),
                                                           tf.TensorSpec(shape=[None,key_points,3], dtype=tf.int32))])
        
        super().build(input_shape)

    def call(self, inputs):
        image, roi_indexes, pose_indexes = inputs
        training=False
        
        low_level_feature = self.low_level_extractor(image)
            
        encoded_pose, encoded_pos = self.encoder(low_level_feature, training = training)
    
        pos_hm = self.pos_decoder(encoded_pos, training = training)
            
        roi_feature = self.roi_extractor([encoded_pose, roi_indexes])
            
        pose_hm = self.pose_decoder(roi_feature, training = training)
                
        poses_xyz, pose_prob_map_xy, pose_prob_maps_z = self.pose_extractor([pose_hm, pose_indexes])
    
        return poses_xyz, pos_hm, (pose_prob_map_xy, pose_prob_maps_z)
        
    def get_config(self):
        config = super().get_config()
        return config