import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

import ShAReD_Net.training.loss.slim as slim_loss
import ShAReD_Net.model.modules.slim as slim_modules

import ShAReD_Net.model.layer.slim_heatmap_1d as heatmap_1d
import ShAReD_Net.model.layer.slim_heatmap_2d as heatmap_2d

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
        key_points = config.model.output.keypoints
        
        self.roi_extractor = slim_modules.Roi_Extractor(roi_size=roi_size)
        
        self.pose_extractor = SlimTrainPoseExtractor()
        
        
        self.call = tf.function(self.call,input_signature=[(tf.TensorSpec(shape=[None,None,None,3], dtype=tf.float32),
                                                           tf.TensorSpec(shape=[None,3], dtype=tf.int32),
                                                           tf.TensorSpec(shape=[None,key_points,3], dtype=tf.int32))])
        
        super().build(input_shape)

    def call(self, inputs):
        image, roi_indexes, pose_indexes = inputs
        training=True
        
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
    
class SlimTrainPoseExtractor(keras.layers.Layer):
    def __init__(self, name = "SlimTrainPoseExtractor", **kwargs):
        super().__init__(name = name, **kwargs)

    
    def build(self, input_shape):
        print(self.name, input_shape)  
        self.key_points = config.model.output.keypoints
        z_bins = config.model.z_bins
        roi_size = np.asarray(config.model.roi_size)
        
        self.xy_extractor = SlimTrainXYExtractor()
        self.z_extractor = SlimTrainZExtractor()
        
        self.call = tf.function(self.call,input_signature=[
            (tf.TensorSpec(shape=[None,roi_size[1],roi_size[0],z_bins+self.key_points], dtype=tf.float32),
             tf.TensorSpec(shape=[None,self.key_points,3], dtype=tf.int32))])
        
        super().build(input_shape)

    def call(self, inputs):
        pose_hm, pose_indexes = inputs
        features_xy = pose_hm[:,:,:,:self.key_points]
        features_z = pose_hm[:,:,:,self.key_points:]
        
        poses_xy, pose_prob_maps_xy = self.xy_extractor(features_xy)
        poses_z, pose_prob_maps_z = self.z_extractor([features_z, pose_indexes])
        
        poses_xyz = tf.concat([poses_xy,poses_z],axis=-1)
    
        return poses_xyz, pose_prob_maps_xy, pose_prob_maps_z
        
    def get_config(self):
        config = super().get_config()
        return config
    
class SlimTrainXYExtractor(keras.layers.Layer):
    def __init__(self, name = "SlimInferenzModel", **kwargs):
        super().__init__(name = name, **kwargs)
    
    def build(self, input_shape):
        print(self.name, input_shape)
        img_downsampling = config.model.img_downsampling
        roi_size = np.asarray(config.model.roi_size)
        person_size = roi_size * img_downsampling
        half_person_size = person_size/2
        keypoints = config.model.output.keypoints
        
        min_loc = -half_person_size
        max_loc = half_person_size
        xy_bins = roi_size
        
        self.loc_map_xy = heatmap_2d.LocationMap(bins=xy_bins, min_loc=min_loc, max_loc=max_loc, dtype=self.dtype)
        
        self.call = tf.function(self.call,input_signature=[
            (tf.TensorSpec(shape=[None,roi_size[1],roi_size[0],keypoints], dtype=tf.float32))])
        
        super().build(input_shape)

    def call(self, inputs):
        features_xy  = inputs
        features_per_keypoint = tf.transpose(features_xy,[0,3,1,2])[...,None] #[b, k, y, x, p[]]
        pose_prob_maps_xy = heatmap_2d.feature_to_location_propability_map(features_per_keypoint)
        loc_map_xy = self.loc_map_xy([0.,0.])
        
        poses_xy = heatmap_2d.propability_map_to_location([pose_prob_maps_xy, loc_map_xy])

        return poses_xy, pose_prob_maps_xy
        
    def get_config(self):
        config = super().get_config()
        return config
    
    
class SlimTrainZExtractor(keras.layers.Layer):
    def __init__(self, name = "SlimInferenzModel", **kwargs):
        super().__init__(name = name, **kwargs)

    
    def build(self, input_shape):
        print(self.name, input_shape)  
        self.keypoints = config.model.output.keypoints
        z_size = config.model.data.cut_delta
        self.z_bins = config.model.z_bins
        roi_size = np.asarray(config.model.roi_size)
        
        bins = self.z_bins
        min_loc = -z_size
        max_loc = +z_size
        self.loc_map_z = heatmap_1d.LocationMap(bins = bins, min_loc=min_loc, max_loc=max_loc)
        
        
        self.call = tf.function(self.call,input_signature=[
            (tf.TensorSpec(shape=[None,roi_size[1],roi_size[0],bins], dtype=tf.float32),
             tf.TensorSpec(shape=[None,self.keypoints,3], dtype=tf.int32))])
        
        super().build(input_shape)

    def call(self, inputs):
        '''
        features_z -> [b, y, x, bins]
        pose_indexes -> [b, k, index[b, y, x]]
        '''
        features_z, pose_indexes = inputs
        relevant_features = tf.gather_nd(features_z, pose_indexes) # [b,k, bins]
        
        pose_prob_maps_z = heatmap_1d.feature_to_location_propability_map(relevant_features)
        loc_map_z = self.loc_map_z(0.)
        poses_z = heatmap_1d.propability_map_to_location([pose_prob_maps_z, loc_map_z]) # [b, k, loc[z]]
    
        return poses_z, pose_prob_maps_z
        
    def get_config(self):
        config = super().get_config()
        return config
    
    
    
class SlimTrainingLoss(keras.layers.Layer):
    def __init__(self, name = "SlimTrainingLoss", **kwargs):
        self.loss_agg = LossAggregation()
        super().__init__(name = name, **kwargs)
    
    def build(self, input_shape):
        print(self.name, input_shape)
        img_downsampling = config.model.img_downsampling
        roi_size = np.asarray(config.model.roi_size)
        person_size = roi_size * img_downsampling
        half_person_size = person_size/2
        z_size = config.model.data.cut_delta
        keypoints = config.model.output.keypoints
        z_bins = config.model.z_bins
        
        self.detection_loss = slim_loss.DetectionLoss()
        self.pose_loss = slim_loss.PoseLoss()
        
        self.call = tf.function(self.call,input_signature=[(
            (tf.TensorSpec(shape=[None,keypoints,3], dtype=tf.float32),
             tf.TensorSpec(shape=[None,keypoints,3], dtype=tf.float32),
             tf.TensorSpec(shape=[None,keypoints,roi_size[1],roi_size[0],1], dtype=tf.float32),
             tf.TensorSpec(shape=[None,keypoints,z_bins], dtype=tf.float32)),
            (tf.TensorSpec(shape=[None,None,None,2], dtype=tf.float32),
             tf.TensorSpec(shape=[None,None,None,2], dtype=tf.float32),
             tf.TensorSpec(shape=[None,None,None,2], dtype=tf.float32)))])
        
        super().build(input_shape)

    def call(self, inputs):
        '''
        
        '''
        (pose_xyz, pose_xyz_gt, pose_prob_map_xy, pose_prob_maps_z), (pos_hm, pos_hm_gt, loss_weights) = inputs
        training=True
                
        detection_loss = self.detection_loss([pos_hm, (pos_hm_gt, loss_weights)])
                
        (loss_pos_xy, loss_var_xy), (loss_pos_z, loss_var_z) = self.pose_loss([pose_xyz, pose_xyz_gt, pose_prob_map_xy, pose_prob_maps_z])
        
        losses = self.loss_agg([detection_loss, (loss_pos_xy, loss_var_xy), (loss_pos_z, loss_var_z)])

        return losses
        
    def get_config(self):
        config = super().get_config()
        return config
    
class LossAggregation(keras.layers.Layer):
    def __init__(self, name = "LossAggregation", **kwargs):
        super().__init__(name = name, **kwargs)

    
    def build(self, input_shape):
        print(self.name,input_shape)
        
        self.multi_loss_layer_detection = slim_loss.MultiLossLayer()
        self.multi_loss_layer_loc_xy = slim_loss.MultiLossLayer()
        self.multi_loss_layer_loc_z = slim_loss.MultiLossLayer()
        self.multi_loss_layer_var_xy = slim_loss.MultiLossLayer()
        self.multi_loss_layer_var_z = slim_loss.MultiLossLayer()

        super().build(input_shape)

    @tf.function(experimental_relax_shapes=True)
    def call(self, inputs):
        detection_loss, (loss_pos_xy, loss_var_xy), (loss_pos_z, loss_var_z) = inputs
        
        detection_loss_sum = tf.reduce_sum(detection_loss)
        detection = self.multi_loss_layer_detection([detection_loss_sum]) * config.training.weighting.detection
    
    
        estimator_loss_xy_list = tf.unstack(loss_pos_xy, axis=1)
        estimator_loss_z_list = tf.unstack(loss_pos_z, axis=1)
       
        estimation_loc_xy = self.multi_loss_layer_loc_xy(estimator_loss_xy_list) * config.training.weighting.xy_loc
        estimation_loc_z = self.multi_loss_layer_loc_z(estimator_loss_z_list) * config.training.weighting.z_loc
        
        
        estimator_loss_var_xy_list = tf.unstack(loss_var_xy, axis=1)
        estimator_loss_var_z_list = tf.unstack(loss_var_z, axis=1)
        
        estimation_var_xy = self.multi_loss_layer_var_xy(estimator_loss_var_xy_list) * config.training.weighting.xy_var
        estimation_var_z = self.multi_loss_layer_var_z(estimator_loss_var_z_list) * config.training.weighting.z_var
        
        loss_per_batch_sum = detection + estimation_loc_xy + estimation_var_xy + estimation_loc_z + estimation_var_z

        
        return loss_per_batch_sum, detection, (estimation_loc_xy, estimation_var_xy), (estimation_loc_z, estimation_var_z)
        
    def get_config(self):
        config = super().get_config()
        return config