import tensorflow as tf
import numpy as np

class FeatureToLocationPropabilityMap(tf.keras.layers.Layer):

    def __init__(self):
        super().__init__()

    @tf.function
    def call(self, feature):
        shape = feature.shape
        flat = tf.reshape(feature,[shape[0], -1])
        flat = tf.nn.softmax(flat)
        return tf.reshape(flat, shape)
    
    def get_config(self):
        config = super().get_config()
        return config       

feature_to_location_propability_map = FeatureToLocationPropabilityMap()

class LocationMap(tf.keras.layers.Layer):

    def __init__(self, min_loc=[0,0], max_loc=[3000,3000], bins=[10,10]):
        super().__init__()
        self.bins = tf.constant(bins, dtype = tf.float32)
        self.min_loc = tf.constant(min_loc, dtype = tf.float32)
        self.max_loc = tf.constant(max_loc, dtype = tf.float32)
        self.build(None)

    def build(self, input_shape):
        self.loc_delta = (self.max_loc - self.min_loc) / self.bins
        loc_map = np.meshgrid(*[np.arange(_min_loc, _max_loc, _delta) for _min_loc, _max_loc, _delta in  zip(self.min_loc, self.max_loc, self.loc_delta)])
        loc_map = tf.constant(loc_map, dtype = tf.float32)
        self.loc_map = tf.transpose(loc_map, [2,1,0])

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

    def __init__(self):
        super().__init__()

    @tf.function
    def call(self, loc_prop_map, loc_map):
        return tf.reduce_sum(loc_prop_map * loc_map, axis=[1,2])
    
    def get_config(self):
        config = super().get_config()
        return config
    
propability_map_to_location = PropabilityMapToLocation()    
    
class PropabilityMapToIndex(tf.keras.layers.Layer):

    def __init__(self):
        super().__init__()
        
    def build(self, input_shape):
        loc_x = tf.range(input_shape[1], dtype=tf.float32)
        loc_y = tf.range(input_shape[2], dtype=tf.float32)
        loc_x = tf.broadcast_to(tf.expand_dims(loc_x,axis=-1),[input_shape[1],input_shape[2]])
        loc_y = tf.broadcast_to(tf.expand_dims(loc_y,axis=0),[input_shape[1],input_shape[2]])
        self.index_map = tf.stack([loc_x,loc_y],axis=-1)

    @tf.function
    def call(self, loc_prop_map):
        indexes = tf.reduce_sum(loc_prop_map * self.index_map, axis=[1,2])
        indexes = tf.minimum(indexes, tf.cast(tf.shape(loc_prop_map)[1:3]-1, dtype=tf.float32))
        return tf.cast(indexes +0.5, dtype=tf.int32)
    
    def get_config(self):
        config = super().get_config()
        return config

propability_map_to_index = PropabilityMapToIndex()

class VarianceLocatonLoss(tf.keras.layers.Layer):
    
    def __init__(self, loc_delta):
        super().__init__()
        self.loc_delta = loc_delta
               
    def build(self, input_shape):
        self.variance_offset = (self.loc_delta/2)**2

    @tf.function
    def call(self, loc_prop_map, loc_map, gt_loc):
        gt_loc = tf.reshape(gt_loc,[gt_loc.shape[0],1,1,gt_loc.shape[-1]])
        variance = tf.reduce_sum(loc_prop_map * (loc_map - gt_loc)**2, axis=[1,2])
        shifted_var = tf.reduce_mean((tf.math.maximum(variance,self.variance_offset) - self.variance_offset), axis=1)
        return shifted_var

class VarianceLocationAndPossitionLoss(tf.keras.layers.Layer):
    def __init__(self, loc_delta):
        super().__init__()
        self.mse = tf.keras.losses.MeanSquaredError(tf.keras.losses.Reduction.NONE)
        self.vll = VarianceLocatonLoss(loc_delta)

    @tf.function
    def call(self, loc_prop_map, loc, loc_map, gt_loc):
        return self.mse(loc, gt_loc) + self.vll(loc_prop_map, loc_map, gt_loc)

class LocationToIndex(tf.keras.layers.Layer):
    
    def __init__(self, loc_delta, max_index):
        super().__init__()
        self.loc_delta = tf.cast(loc_delta,dtype=tf.float32)
        self.max_index = tf.cast(max_index,dtype=tf.float32) - 1

    @tf.function
    def call(self, loc):
        indexes = loc / self.loc_delta
        indexes = tf.minimum(indexes, self.max_index)
        return tf.cast(indexes +0.5, dtype=tf.int32)


if __name__ == "__main__":
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
    
    lti = LocationToIndex(loc_map_op.loc_delta, bins)
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


    
    