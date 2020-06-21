import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from ShAReD_Net.configure import config

class FeatureToLocationPropabilityMap(tf.keras.layers.Layer):

    def __init__(self, name = "FeatureToLocationPropabilityMap", **kwargs):
        super().__init__(name=name,**kwargs)
        
    def build(self, input_shape):
        z_bins = config.model.z_bins
        keypoints = config.model.output.keypoints
        
        self.call = tf.function(self.call,input_signature=[(tf.TensorSpec(shape=[None,keypoints,z_bins], dtype=self.dtype))])
        
        super().build(input_shape)

    def call(self, feature):
        prob = tf.nn.softmax(feature)
        return prob
    
    def get_config(self):
        config = super().get_config()
        return config
        
feature_to_location_propability_map = FeatureToLocationPropabilityMap()

class LocationMap(tf.keras.layers.Layer):

    def __init__(self, min_loc=0, max_loc=3000, bins=10, name = "LocationMap", **kwargs):
        super().__init__(name=name, **kwargs)
        self.bins = tf.cast(bins, dtype = self.dtype)
        self.min_loc = tf.cast(min_loc, dtype = self.dtype)
        self.max_loc = tf.cast(max_loc, dtype = self.dtype)
        self.build(None)

    def build(self, input_shape):
        self.loc_delta = (self.max_loc - self.min_loc) / self.bins
        loc_map = np.arange(self.min_loc, self.max_loc, self.loc_delta)
        self.loc_map = tf.constant(loc_map, dtype = self.dtype)
        self.loc_map = tf.reshape(self.loc_map,[1,1,self.bins])
        super().build(input_shape)

    @tf.function
    def call(self, inputs):
        return self.loc_map + inputs
    
    def get_config(self):
        config = super().get_config()
        config.update({"bins": self.bins,
                       "min_loc": self.min_loc,
                       "max_loc": self.max_loc,
                       })
        return config

class PropabilityMapToLocation(tf.keras.layers.Layer):

    def __init__(self, name = "PropabilityMapToLocation", **kwargs):
        super().__init__(name=name, **kwargs)
        
    def build(self, input_shape):
        z_bins = config.model.z_bins
        keypoints = config.model.output.keypoints
        
        self.call = tf.function(self.call,input_signature=[(tf.TensorSpec(shape=[None,keypoints,z_bins], dtype=self.dtype),
                                                           tf.TensorSpec(shape=[1,1,z_bins], dtype=self.dtype))])
        
        super().build(input_shape)

    def call(self, inputs):
        '''
        loc_prop_map -> [b,k, p[bins]]
        loc_map -> [1,1, loc[bins]]
        '''
        loc_prop_map, loc_map = inputs
        loc = tf.reduce_sum(loc_prop_map * loc_map, axis=[-1])
        return tf.expand_dims(loc,axis=-1)
    
    def get_config(self):
        config = super().get_config()
        return config

propability_map_to_location = PropabilityMapToLocation()

class VarianceLocatonLoss(tf.keras.layers.Layer):
    
    def __init__(self, loc_delta, name = "VarianceLocatonLoss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.loc_delta = loc_delta
               
    def build(self, input_shape):
        self.variance_offset = (self.loc_delta/2)**2
        
        z_bins = config.model.z_bins
        keypoints = config.model.output.keypoints
        
        self.call = tf.function(self.call,input_signature=[(tf.TensorSpec(shape=[None,keypoints,z_bins], dtype=self.dtype),
                                                           tf.TensorSpec(shape=[1,1,z_bins], dtype=self.dtype),
                                                           tf.TensorSpec(shape=[None,keypoints,1], dtype=self.dtype))])
        
        super().build(input_shape)

    def call(self, inputs):
        '''
        pose_prob_maps_z -> [b, k, p[bins]]
        loc_map_z -> [1, 1, loc[bins]]
        pose_z_gt -> [b, k, loc[z]]
        '''
        pose_prob_maps_z, loc_map_z, pose_z_gt = inputs
        variance = tf.reduce_sum(pose_prob_maps_z * (loc_map_z - pose_z_gt)**2, axis=[-1])
        maxed_var = tf.math.maximum(variance,self.variance_offset)
        shifted_var = maxed_var - self.variance_offset
        return tf.expand_dims(shifted_var,axis=-1)

class VarianceLocationAndPossitionLoss(tf.keras.layers.Layer):
    def __init__(self, loc_delta, name = "VarianceLocationAndPossitionLoss", **kwargs):
        super().__init__(name=name,**kwargs)
        self.vll = VarianceLocatonLoss(loc_delta)
        
    def build(self, input_shape):        
        z_bins = config.model.z_bins
        keypoints = config.model.output.keypoints
        
        self.call = tf.function(self.call,input_signature=[(tf.TensorSpec(shape=[None,keypoints,1], dtype=self.dtype),
                                                            tf.TensorSpec(shape=[None,keypoints,1], dtype=self.dtype),
                                                            tf.TensorSpec(shape=[None,keypoints,z_bins], dtype=self.dtype),
                                                            tf.TensorSpec(shape=[1,1,z_bins], dtype=self.dtype))])
        
        super().build(input_shape)

    def call(self, inputs):
        '''
        pose_z -> [b, k, loc[z]]
        pose_z_gt -> [b, k, loc[z]]
        pose_prob_maps_z -> [b, k, p[bins]]
        loc_map_z -> [1, 1, loc[bins]]       
        '''
        pose_z, pose_z_gt, pose_prob_maps_z, loc_map_z = inputs
        se = (pose_z-pose_z_gt)**2
        vll = self.vll([pose_prob_maps_z, loc_map_z, pose_z_gt])
        return se, vll
    