import tensorflow as tf
import numpy as np

class FeatureToLocationPropabilityMap(tf.keras.layers.Layer):

    def __init__(self, name = "FeatureToLocationPropabilityMap", **kwargs):
        super().__init__(name=name, **kwargs)

    @tf.function
    def call(self, feature):
        shape = tf.shape(feature)
        flat = tf.reshape(feature,[shape[0], -1])
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
        self.loc_map = tf.transpose(loc_map, [1,2,0])
        super().build(inputs_shape)

    @tf.function
    def call(self, inputs):
        offset = inputs
        return self.loc_map + offset
    
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
    def call(self, loc_prop_map, loc_map):
        return tf.reduce_sum(loc_prop_map * loc_map, axis=[1,2])
    
    def get_config(self):
        config = super().get_config()
        return config
    
propability_map_to_location = PropabilityMapToLocation()    
    
class PropabilityMapToIndex(tf.keras.layers.Layer):

    def __init__(self, name = "PropabilityMapToIndex", **kwargs):
        super().__init__(name=name, **kwargs)
        
    def build(self, input_shape):
        loc_x = tf.range(input_shape[1], dtype=self.dtype)
        loc_y = tf.range(input_shape[2], dtype=self.dtype)
        loc_x = tf.broadcast_to(tf.expand_dims(loc_x,axis=-1),[input_shape[1],input_shape[2]])
        loc_y = tf.broadcast_to(tf.expand_dims(loc_y,axis=0),[input_shape[1],input_shape[2]])
        self.index_map = tf.stack([loc_x,loc_y],axis=-1)
        super().build(input_shape)

    @tf.function
    def call(self, loc_prop_map):
        indexes = tf.reduce_sum(loc_prop_map * self.index_map, axis=[1,2])
        indexes = tf.minimum(indexes, tf.cast(tf.shape(loc_prop_map)[1:3]-1, dtype=self.dtype))
        return tf.cast(indexes +0.5, dtype=tf.int32)
    
    def get_config(self):
        config = super().get_config()
        return config

propability_map_to_index = PropabilityMapToIndex()

class VarianceLocatonLoss(tf.keras.layers.Layer):
    
    def __init__(self, loc_delta, name = "VarianceLocatonLoss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.loc_delta = loc_delta
    
    def build(self, input_shape):
        self.variance_offset = (self.loc_delta/2)**2
        super().build(input_shape)

    @tf.function
    def call(self, loc_prop_map, loc_map, gt_loc):
        gt_loc_shape = tf.shape(gt_loc)
        gt_loc = tf.reshape(gt_loc,[gt_loc_shape[0],1,1,gt_loc_shape[-1]])
        variance = tf.reduce_sum(loc_prop_map * (loc_map - gt_loc)**2, axis=[1,2])
        maxed_var = tf.math.maximum(variance,self.variance_offset)
        max_loss = (maxed_var-variance)**2
        shifted_var = tf.reduce_mean(maxed_var - self.variance_offset + max_loss, axis=1)
        return shifted_var

class VarianceLocationAndPossitionLoss(tf.keras.layers.Layer):
    def __init__(self, loc_delta, name = "VarianceLocationAndPossitionLoss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.mse = tf.keras.losses.MeanSquaredError(tf.keras.losses.Reduction.NONE)
        self.vll = VarianceLocatonLoss(loc_delta)

    @tf.function
    def call(self, loc_prop_map, loc, loc_map, gt_loc):
        mse = self.mse(loc, gt_loc)
        vll = self.vll(loc_prop_map, loc_map, gt_loc)
        return mse, vll

class LocationToIndex(tf.keras.layers.Layer):
    
    def __init__(self, loc_delta, min_loc, loc_bins, name = "LocationToIndex", **kwargs):
        super().__init__(name=name, **kwargs)
        self.loc_delta = tf.cast(loc_delta,dtype=self.dtype)
        self.max_index = tf.cast(loc_bins,dtype=self.dtype) - 1
        self.min_loc = tf.cast(min_loc,dtype=self.dtype)

    @tf.function
    def call(self, loc):
        indexes = (loc - self.min_loc) / self.loc_delta
        indexes = tf.maximum(indexes, 0)
        indexes = tf.minimum(indexes, self.max_index)
        return tf.cast(indexes +0.5, dtype=tf.int32)[:,::-1]

def main():
    bins = [30,15]
    bin0_half = int(bins[0]/2)
    bin1_half = int(bins[1]/2)
    loc_prop_map = np.zeros([4,bins[0],bins[1],1],dtype=np.float32)
    loc_prop_map[0,bin0_half,bin1_half,0]=0.5
    loc_prop_map[0,bin0_half+1,bin1_half+1,0]=0.5
    loc_prop_map[1,bin0_half-1,bin1_half-1,0]=0.5
    loc_prop_map[1,bin0_half+2,bin1_half+2,0]=0.5
    loc_prop_map[2,bin0_half,bin1_half,0]=1
    loc_prop_map[3,bin0_half,bin1_half,0]=1
    loc_prop_map[3,bin0_half+1,bin1_half+1,0]=1
    
    ftpm = FeatureToLocationPropabilityMap()
    loc_prop_map_test = ftpm(loc_prop_map)
    print(tf.reduce_sum(loc_prop_map_test,axis=[1,2]))
    
    loc_map_op = LocationMap(bins=bins)
    loc_map = loc_map_op([0.,0.])
    
    pmtl = PropabilityMapToLocation()
    loc = pmtl(loc_prop_map, loc_map)
    print("loc:",loc)
    
    lti = LocationToIndex(loc_map_op.loc_delta, loc_map_op.min_loc, bins)
    indexes = lti(loc)
    print("indexes loc:",indexes)
    
    pmti = PropabilityMapToIndex()
    indexes = pmti(loc_prop_map)
    print("indexes map:",indexes)
    
    gt_loc = np.asarray([[1550,1550],[1550,1550],[1500,1500],[1500,1500]],dtype=np.float32)
    
    vll = VarianceLocatonLoss(loc_map_op.loc_delta)
    loss = vll(loc_prop_map, loc_map, gt_loc)
    print("var_loss:",loss)
    
    vlapl = VarianceLocationAndPossitionLoss(loc_map_op.loc_delta)
    loss = vlapl(loc_prop_map, loc, loc_map, gt_loc)
    print("loc_loss:",loss)


if __name__ == "__main__":
    tf.profiler.experimental.start('/home/inferics/Docker/volumes/3D_Person_Pose_Estimation_from_2D_Singelview_Image_Data/logdir')
    main()
    tf.profiler.experimental.stop()