import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

import ShAReD_Net.training.loss.slim as slim_loss
import ShAReD_Net.model.modules.slim as slim_modules
from ShAReD_Net.configure import config


class SlimTrainingModel(keras.layers.Layer):
    def __init__(self, low_level_extractor, encoder, pos_decoder, pose_decoder, name = "SlimInferenzModel", **kwargs):
        super().__init__(name = name, **kwargs)
        self.low_level_extractor = low_level_extractor
        self.encoder = encoder
        self.pos_decoder = pos_decoder
        self.pose_decoder = pose_decoder

    
    def build(self, input_shape):
        print(self.name, input_shape)
        roi_size = np.asarray(config.model.roi_size)
        
        self.roi_extractor = slim_modules.Roi_Extractor(roi_size=roi_size)
        
        self.call = tf.function(self.call,input_signature=[(tf.TensorSpec(shape=[None,None,None,3], dtype=tf.float32),
                                                           tf.TensorSpec(shape=[None,3], dtype=tf.int32))])
        
        super().build(input_shape)

    def call(self, inputs):
        image, roi_indexes = inputs
        training=True
        
        low_level_feature = self.low_level_extractor(image)
    
        encoded_pose, encoded_pos = self.encoder(low_level_feature, training = training)
    
        pos_hm = self.pos_decoder(encoded_pos, training = training)
            
        roi_feature = self.roi_extractor([encoded_pose, roi_indexes])
    
        pose_hm = self.pose_decoder(roi_feature, training = training)
    
        return pose_hm, pos_hm
        
    def get_config(self):
        config = super().get_config()
        return config
    
class SlimTrainingLoss(keras.layers.Layer):
    def __init__(self, name = "SlimTrainingLoss", **kwargs):
        super().__init__(name = name, **kwargs)
    
    def build(self, input_shape):
        print(self.name, input_shape)
        img_downsampling = config.model.img_downsampling
        roi_size = np.asarray(config.model.roi_size)
        person_size = roi_size * img_downsampling
        half_person_size = person_size/2
        z_size = config.model.data.cut_delta
        
        self.detection_loss = slim_loss.DetectionLoss()
        self.pose_loss = slim_loss.PoseLoss(key_points = config.model.output.keypoints,
                                            depth_bins = config.model.z_bins,
                                            xyz_min=[-half_person_size[0], -half_person_size[1], -z_size],
                                            xyz_max=[half_person_size[0], half_person_size[1], z_size])
        
        self.call = tf.function(self.call,input_signature=[(tf.TensorSpec(shape=[None,roi_size[0],roi_size[1],config.model.output.keypoints + config.model.z_bins], dtype=tf.float32),
                                                            tf.TensorSpec(shape=[None,None,None,2], dtype=tf.float32),
                                                            tf.TensorSpec(shape=[None,config.model.output.keypoints,3], dtype=tf.float32),
                                                            (tf.TensorSpec(shape=[None,None,None,2], dtype=tf.float32),
                                                             tf.TensorSpec(shape=[None,None,None,2], dtype=tf.float32),
                                                             )
                                                            )])
        
        super().build(input_shape)

    def call(self, inputs):
        pose_hm, pos_hm, roi_pose_gt, (pos_hm_gt, loss_weights) = inputs
        training=True
                
        detection_loss = self.detection_loss([pos_hm, (pos_hm_gt, loss_weights)])
                
        estimator_loss_xy, estimator_loss_z = self.pose_loss([pose_hm, roi_pose_gt])

        return detection_loss, estimator_loss_xy, estimator_loss_z
        
    def get_config(self):
        config = super().get_config()
        return config