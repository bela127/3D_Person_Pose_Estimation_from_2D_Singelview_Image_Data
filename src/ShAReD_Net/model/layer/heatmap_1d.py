import tensorflow as tf
import numpy as np

class FeatureToLocationPropabilityMap(tf.keras.layers.Layer):

    def __init__(self):
        super().__init__()

    @tf.function
    def call(self, feature):
        prob = tf.nn.softmax(feature)
        return prob
    
    def get_config(self):
        config = super().get_config()
        return config
        
feature_to_location_propability_map = FeatureToLocationPropabilityMap()

class LocationMap(tf.keras.layers.Layer):

    def __init__(self, min_loc=0, max_loc=3000, bins=10):
        super().__init__()
        self.bins = tf.cast(bins, dtype = tf.float32)
        self.min_loc = tf.cast(min_loc, dtype = tf.float32)
        self.max_loc = tf.cast(max_loc, dtype = tf.float32)
        self.build(None)

    def build(self, input_shape):
        self.loc_delta = (self.max_loc - self.min_loc) / self.bins
        loc_map = np.arange(self.min_loc, self.max_loc, self.loc_delta)
        self.loc_map = tf.constant(loc_map, dtype = tf.float32)
        self.loc_map = tf.reshape(self.loc_map,[1,1,1,self.bins])
        super().build(input_shape)

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
        loc = tf.reduce_sum(loc_prop_map * loc_map, axis=[-1])
        return tf.expand_dims(loc,axis=-1)
    
    def get_config(self):
        config = super().get_config()
        return config

propability_map_to_location = PropabilityMapToLocation()

class VarianceLocatonLoss(tf.keras.layers.Layer):
    
    def __init__(self, loc_delta):
        super().__init__()
        self.loc_delta = loc_delta
               
    def build(self, input_shape):
        self.variance_offset = (self.loc_delta/2)**2
        super().build(input_shape)

    @tf.function
    def call(self, loc_prop_map, loc_map, gt_loc):
        variance = tf.reduce_sum(loc_prop_map * (loc_map - gt_loc)**2, axis=[3])
        shifted_var = (tf.math.maximum(variance,self.variance_offset) - self.variance_offset)
        return tf.expand_dims(shifted_var,axis=-1)

class VarianceLocationAndPossitionLoss(tf.keras.layers.Layer):
    def __init__(self, loc_delta):
        super().__init__()
        self.vll = VarianceLocatonLoss(loc_delta)

    @tf.function
    def call(self, loc_prop_map, loc, loc_map, gt_loc):
        se = (loc-gt_loc)**2
        vll = self.vll(loc_prop_map, loc_map, gt_loc)
        return se + vll

class MaskFromIndex(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    @tf.function
    def call(self, index, dest_shape):
        indices = tf.cast(index, dtype =tf.int64)
        dest_shape = tf.cast(dest_shape, dtype =tf.int64)
        mask = tf.SparseTensor(indices = indices, values = tf.ones(tf.shape(index)[0]), dense_shape = dest_shape)
        mask = tf.sparse.to_dense(mask, validate_indices=False)
        return tf.expand_dims(mask,axis=-1)
    
mask_from_index = MaskFromIndex()
    
class MaskPropabilityMap(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    @tf.function
    def call(self, loc_prop_map, mask):
        loc_prop_map = loc_prop_map * mask
        return loc_prop_map

mask_propability_map = MaskPropabilityMap()
    
class ExpandGt(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    @tf.function
    def call(self, gt_index, gt_loc, dest_shape):
        indices = tf.cast(gt_index, dtype =tf.int64)
        dest_shape = tf.cast(dest_shape, dtype =tf.int64)
        gt_loc = tf.SparseTensor(indices = indices, values = gt_loc, dense_shape = dest_shape)
        gt_loc = tf.sparse.expand_dims(gt_loc)
        gt_loc = tf.sparse.to_dense(gt_loc, validate_indices=False)
        return gt_loc

expand_gt = ExpandGt()

    
def main():
    bins = 30
    bin_half = int(bins/2)
    loc_prop_map = np.zeros([2,5,5,bins],dtype=np.float32)
    loc_prop_map[0,1,1,bin_half]=0.5
    loc_prop_map[0,1,1,bin_half+1]=0.5
    loc_prop_map[0,2,2,bin_half-1]=0.5
    loc_prop_map[0,2,2,bin_half+2]=0.5
    loc_prop_map[0,4,4,bin_half]=1
    loc_prop_map[1,1,1,bin_half]=1
    loc_prop_map[1,1,1,bin_half+1]=1

    ftpm = FeatureToLocationPropabilityMap()
    loc_prop_map_test = ftpm(loc_prop_map)
    print(tf.reduce_sum(loc_prop_map_test,axis=[-1]))
    
    loc_map_op = LocationMap(bins=bins)
    loc_map = loc_map_op(0.)
    
    gt_index = np.asarray([[0,1,1],[0,2,2],[0,4,4],[1,1,1]],dtype=np.float32)
    gt_loc = np.asarray([1550,1550,1500,1500],dtype=np.float32)
    mask = mask_from_index(gt_index, loc_prop_map.shape[0:3])

    loc_prop_map = mask_propability_map(loc_prop_map, mask)
    loc_prop_map_test = mask_propability_map(loc_prop_map_test, mask)
    print(tf.reduce_sum(loc_prop_map_test,axis=[-1]))

    gt_loc = expand_gt(gt_index, gt_loc, tf.shape(loc_prop_map)[0:3])
    
    pmtl = PropabilityMapToLocation()
    loc = pmtl(loc_prop_map, loc_map)
    print("loc:",loc)
    
    
    vll = VarianceLocatonLoss(loc_map_op.loc_delta)
    loss = vll(loc_prop_map, loc_map, gt_loc)
    print("var_loss:",loss)
    
    vlapl = VarianceLocationAndPossitionLoss(loc_map_op.loc_delta)
    loss = vlapl(loc_prop_map, loc, loc_map, gt_loc)
    print("loc_loss:",loss)


    

if __name__ == "__main__":
    main()