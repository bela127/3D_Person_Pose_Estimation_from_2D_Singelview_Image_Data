import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

import ShAReD_Net.model.layer.slim_heatmap_1d as heatmap_1d
import ShAReD_Net.model.layer.slim_heatmap_2d as heatmap_2d

from ShAReD_Net.configure import config

class GradNormLayer(tf.keras.layers.Layer):
    def __init__(self, damping = 1, name = "GradNormLayer", **kwargs):
        self.damping = damping

        super().__init__(name = name, **kwargs)
        
    def build(self, inputs_shape):
        print(self.name, inputs_shape)
        loss_shape, grads_shape = inputs_shape
        self.loss_weigts = self.add_weight(name='loss_weigts', shape=loss_shape, initializer=tf.ones, trainable=True)
        self.inital_loss = self.add_weight(name='inital_loss', shape=loss_shape, initializer=tf.constant_initializer(value=-1), trainable=True)
        self.call = tf.function(self.call,input_signature=[(tf.TensorSpec(shape=loss_shape, dtype=self.dtype),[tf.TensorSpec(shape=None, dtype=self.dtype) for grad_shape in grads_shape])])
        super().build(inputs_shape)

    def call(self, inputs):
        loss, var = inputs
        
        weighted_loss = tf.reduce_sum(tf.multiply(loss, tf.stop_gradient(tf.nn.softmax(self.loss_weigts))))
        
        grad_norms = []
        for grad in grads:
            grad_norms.append(tf.linalg.global_norm(grad))
        grad_norms = tf.stack(grad_norms)
        
        mean_grad = tf.reduce_mean(grad_norms)
        
        if tf.reduce_sum(self.inital_loss) < 0:
            self.inital_loss.assign(weighted_loss)
        
        advance = weighted_loss / self.inital_loss
        mean_advance = tf.reduce_mean(advance)
        norm_rate = advance / mean_advance
        
        weighting_loss = reduce_sum((grad_norms - tf.stop_gradient(mean_grad * norm_rate**self.damping))**2)
                                    
        return weighted_loss, weighting_loss

class MultiLossLayer(tf.keras.layers.Layer):
    def __init__(self, name = "MultiLossLayer", **kwargs):

        super().__init__(name = name, **kwargs)
        
    def build(self, inputs_shape):
        print(self.name, inputs_shape)
        self.log_vars = []
        for i, input_shape in enumerate(inputs_shape):
            self.log_vars.append(self.add_weight(name='log_var' + str(i), shape=(),
                                              initializer=tf.ones, trainable=True))
        
        self.call = tf.function(self.call,input_signature=[list(tf.TensorSpec(shape=input_shape, dtype=self.dtype) for input_shape in inputs_shape)])
        super().build(inputs_shape)

    def multi_loss(self, in_loss, log_var):
        precision = tf.exp(-log_var)
        loss = precision * tf.reduce_sum(in_loss) + log_var
        return loss

    def call(self, inputs):
        losses = []
        for in_loss, log_var in zip(inputs, self.log_vars):
            loss = self.multi_loss(in_loss, log_var)
            losses.append(loss)
        stacked = tf.stack(losses)
        agg_loss = tf.reduce_sum(stacked)
        return agg_loss

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


class PoseLossZ(tf.keras.layers.Layer):
    
    def __init__(self, name = "PoseLossDepth", **kwargs):
        super().__init__(name = name, **kwargs)     
        
    def build(self, inputs_shape):
        print(self.name,inputs_shape)
        z_size = config.model.data.cut_delta
        self.z_bins = config.model.z_bins
        self.keypoints = config.model.output.keypoints
        
        min_loc = -z_size
        max_loc = +z_size
        self.loc_map_z = heatmap_1d.LocationMap(bins = self.z_bins, min_loc=min_loc, max_loc=max_loc)
                
        self.loss_z = heatmap_1d.VarianceLocationAndPossitionLoss(self.loc_map_z.loc_delta)
                
        self.call = tf.function(self.call,input_signature=[(tf.TensorSpec(shape=[None,self.keypoints,1], dtype=self.dtype),
                                                            tf.TensorSpec(shape=[None,self.keypoints,1], dtype=self.dtype),
                                                            tf.TensorSpec(shape=[None,self.keypoints,self.z_bins], dtype=self.dtype))])
        super().build(inputs_shape)

    # is a @tf.function with defined input shape
    def call(self, inputs):
        pose_z, pose_z_gt, pose_prob_maps_z = inputs
                        
        loc_map_z = self.loc_map_z(0.)
        loss_loc, loss_var = self.loss_z([pose_z, pose_z_gt, pose_prob_maps_z, loc_map_z])
                
        loss_z_sum = tf.reduce_sum(loss_loc) + 0.001
        loss_z_batch = tf.reduce_sum(loss_loc, axis=[0])
        loss_z_factor = loss_z_batch*self.keypoints/tf.stop_gradient(loss_z_sum)
        
        hard_kp_loss_z = loss_loc*loss_z_factor
        return hard_kp_loss_z, loss_var
    
class PoseLossXY(tf.keras.layers.Layer):
    
    def __init__(self, name = "PoseLoss2D", **kwargs):
        super().__init__(name = name, **kwargs)
        
    
    def build(self, inputs_shape):
        print(self.name,inputs_shape)
        self.keypoints = config.model.output.keypoints
        img_downsampling = config.model.img_downsampling
        roi_size = np.asarray(config.model.roi_size)
        person_size = roi_size * img_downsampling
        half_person_size = person_size/2
        
        min_loc = -half_person_size
        max_loc = half_person_size
        yx_size = roi_size[::-1]
        
        self.loc_map_xy = heatmap_2d.LocationMap(bins=roi_size, min_loc=min_loc, max_loc=max_loc, dtype=self.dtype)
                
        self.loss_xy = heatmap_2d.VarianceLocationAndPossitionLoss(self.loc_map_xy.loc_delta)
                
        self.call = tf.function(self.call,input_signature=[(tf.TensorSpec(shape=[None,self.keypoints,2], dtype=self.dtype),
                                                            tf.TensorSpec(shape=[None,self.keypoints,2], dtype=self.dtype),
                                                            tf.TensorSpec(shape=[None,self.keypoints, yx_size[0], yx_size[1], 1], dtype=self.dtype))])
        super().build(inputs_shape)

    # is a @tf.function with defined input shape
    def call(self, inputs):
        pose_xy, pose_xy_gt, pose_prob_maps_xy = inputs
                        
        loc_map_xy = self.loc_map_xy([0.,0.])
        loss_loc, loss_var = self.loss_xy([pose_xy, pose_xy_gt, pose_prob_maps_xy, loc_map_xy])
                
        loss_xy_sum = tf.reduce_sum(loss_loc) + 0.001
        loss_xy_batch = tf.reduce_sum(loss_loc, axis=[0,2])
        loss_xy_factor = loss_xy_batch*self.keypoints/tf.stop_gradient(loss_xy_sum)
        
        
        hard_kp_loss_xy = loss_loc*loss_xy_factor[...,None]
        return hard_kp_loss_xy, loss_var
    
class PoseLoss(tf.keras.layers.Layer):
    
    def __init__(self, name = "PoseLoss", **kwargs):
        super().__init__(name = name, **kwargs)
        
    
    def build(self, inputs_shape):
        print(self.name,inputs_shape)  
        self.keypoints = config.model.output.keypoints
        self.z_bins = config.model.z_bins
        xy_size = config.model.roi_size[::-1]
        
        feature_shape = inputs_shape[0]
        
        self.pose_loss_xy = PoseLossXY()
        self.pose_loss_z = PoseLossZ()
        

        self.call = tf.function(self.call,input_signature=[
            (tf.TensorSpec(shape=[None, self.keypoints,3], dtype=self.dtype),
             tf.TensorSpec(shape=[None, self.keypoints,3], dtype=self.dtype),
             tf.TensorSpec(shape=[None, self.keypoints, xy_size[0], xy_size[1], 1], dtype=self.dtype),
             tf.TensorSpec(shape=[None, self.keypoints, self.z_bins], dtype=self.dtype))])
        super().build(inputs_shape)

    # is a @tf.function with defined input shape
    def call(self, inputs):
        pose_xyz, pose_xyz_gt, pose_prob_map_xy, pose_prob_maps_z = inputs
        pose_xy = pose_xyz[...,:-1]
        pose_xy_gt = pose_xyz_gt[...,:-1]
        pose_z = pose_xyz[...,None,-1]
        pose_z_gt = pose_xyz_gt[...,None,-1]
        
        hard_kp_loss_xy, loss_var_xy = self.pose_loss_xy([pose_xy, pose_xy_gt, pose_prob_map_xy])
        hard_kp_loss_z, loss_var_z = self.pose_loss_z([pose_z, pose_z_gt, pose_prob_maps_z])
                
        #TODO transform to real space before sym loss
        #limbs = limb_length(loc_xyz)
        #sym_loss = symmetry_loss(limbs)
        
        return (hard_kp_loss_xy, loss_var_xy), (hard_kp_loss_z, loss_var_z)