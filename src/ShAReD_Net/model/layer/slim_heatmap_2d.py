import tensorflow as tf
import numpy as np

class FeatureToLocationPropabilityMap(tf.keras.layers.Layer):

    def __init__(self, name = "FeatureToLocationPropabilityMap", **kwargs):
        super().__init__(name=name, **kwargs)

    @tf.function
    def call(self, inputs):
        features_xy = inputs
        shape = tf.shape(features_xy)
        flat = tf.reshape(features_xy,[shape[0],shape[1], -1])
        flat = tf.nn.softmax(flat)
        return tf.reshape(flat, shape)
    
    def get_config(self):
        config = super().get_config()
        return config       

feature_to_location_propability_map = FeatureToLocationPropabilityMap()


class LocationMap(tf.keras.layers.Layer):

    def __init__(self, min_loc=[0,0], max_loc=[3000,3000], bins=[10,10], name = "LocationMap", **kwargs):
        super().__init__(name=name, **kwargs)
        self.bins = tf.cast(bins, dtype = self.dtype)
        self.min_loc = tf.cast(min_loc, dtype = self.dtype)
        self.max_loc = tf.cast(max_loc, dtype = self.dtype)
        self.build(None)

    def build(self, inputs_shape):
        print(self.name, inputs_shape)
        self.loc_delta = (self.max_loc - self.min_loc) / self.bins
        loc_map = np.meshgrid(*[np.arange(_min_loc, _max_loc, _delta) for _min_loc, _max_loc, _delta in zip(self.min_loc, self.max_loc, self.loc_delta)])
        loc_map = tf.constant(loc_map, dtype = self.dtype)
        loc_map = tf.transpose(loc_map, [1,2,0])
        self.loc_map = loc_map[None,None,...]
        super().build(inputs_shape)

    @tf.function
    def call(self, inputs):
        '''
        loc_map -> [b, k, y, x, loc[x,y]]
        '''
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

    @tf.function
    def call(self, inputs):
        '''
        in:
        pose_prop_map, loc_map = inputs
        pose_prop_map -> [b, k, y, x, p[]]
        loc_map -> [b, k, y, x, loc[x,y]]
        
        out:
        pose -> [b, k, xloc_yloc]
        '''
        pose_prop_map, loc_map = inputs
        pose = tf.reduce_sum(pose_prop_map * loc_map, axis=[2,3])
        return pose
    
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
        super().build(input_shape)

    @tf.function
    def call(self, inputs):
        '''
        pose_prob_maps_xy -> [b,k,y,x,p[]]
        loc_map_xy -> [1,1,y,x,loc[x,y]]
        pose_xy_gt -> [b,k,loc[x,y]]
        '''
        pose_prob_maps_xy, loc_map_xy, pose_xy_gt = inputs
        pose_xy_gt_shape = tf.shape(pose_xy_gt)
        gt_loc_map = pose_xy_gt[:,:,None,None,:]
        variance = tf.reduce_sum(pose_prob_maps_xy * (loc_map_xy - gt_loc_map)**2, axis=[2,3])
        maxed_var = tf.math.maximum(variance,self.variance_offset)
        shifted_var = (maxed_var - self.variance_offset)
        return shifted_var

class VarianceLocationAndPossitionLoss(tf.keras.layers.Layer):
    def __init__(self, loc_delta, name = "VarianceLocationAndPossitionLoss", **kwargs):
        super().__init__(name=name,**kwargs)
        self.vll = VarianceLocatonLoss(loc_delta)

    @tf.function
    def call(self, inputs):
        pose_xy, pose_xy_gt, pose_prob_maps_xy, loc_map_xy = inputs
        se = (pose_xy-pose_xy_gt)**2
        vll = self.vll([pose_prob_maps_xy, loc_map_xy, pose_xy_gt])
        return se, vll
