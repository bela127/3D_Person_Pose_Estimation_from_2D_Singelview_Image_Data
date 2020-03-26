import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


import ShAReD_Net.model.layer.heatmap_1d as heatmap_1d
import ShAReD_Net.model.layer.heatmap_2d as heatmap_2d

class LimbLength(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        
    @tf.Module.with_name_scope
    def build(self, inputs_shape):
        print(self.name,inputs_shape)
        self.limb_connection = tf.constant([0,
                                            0,
                                            1,
                                            1,
                                            1,
                                            3,
                                            4,
                                            5,
                                            6,
                                            2,
                                            2,
                                            9,
                                            10,
                                            11,
                                            12,
                                            ],dtype=tf.int32)
        super().build(inputs_shape)
    
    @tf.function
    @tf.Module.with_name_scope
    def call(self, inputs):
        inputs_shape = tf.shape(inputs)
        size = inputs_shape[0]*inputs_shape[1]
        limb_arr = tf.TensorArray(dtype=tf.float32, size=size, dynamic_size=False)
        for b in tf.range(inputs_shape[0]):
            for kp in tf.range(inputs_shape[1]):
                limb_length = tf.sqrt(tf.reduce_sum((inputs[b,kp] - inputs[b,self.limb_connection[kp]])**2))
                index = b*inputs_shape[1]+kp
                limb_arr = limb_arr.write(index,limb_length)
        limbs = limb_arr.stack()
        limbs = tf.reshape(limbs, [inputs_shape[0],inputs_shape[1],1])
        return limbs

limb_length = LimbLength()

class SymmetryLoss(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
    
    @tf.Module.with_name_scope
    def build(self, inputs_shape):
        print(self.name,inputs_shape)
        self.sym_connection = tf.constant([0,
                                          1,
                                          2,
                                          4,
                                          3,
                                          6,
                                          5,
                                          8,
                                          7,
                                          10,
                                          9,
                                          12,
                                          11,
                                          14,
                                          13,
                                          ],dtype=tf.int32)
        super().build(inputs_shape)
    
    @tf.function
    @tf.Module.with_name_scope
    def call(self, inputs):
        inputs_shape = tf.shape(inputs)
        size = inputs_shape[0]*inputs_shape[1]
        sym_loss_arr = tf.TensorArray(dtype=tf.float32, size=size, dynamic_size=False)
        for b in tf.range(inputs_shape[0]):
            for kp in tf.range(inputs_shape[1]):
                per_limb_loss = (inputs[b,kp] - inputs[b,self.sym_connection[kp]])**2
                sym_loss_arr = sym_loss_arr.write(b*inputs_shape[1]+kp,per_limb_loss)
        sym_loss = sym_loss_arr.stack()
        sym_loss = tf.reshape(sym_loss, [inputs_shape[0],inputs_shape[1],1])
        return sym_loss

symmetry_loss = SymmetryLoss()

class KeypointBatchToPoseGT(tf.keras.layers.Layer):
    def __init__(self, loc_delta_xy, min_loc_xy, max_idx_xy, loc_delta_z, min_loc_z, max_idx_z):
        super().__init__()
        self.loc_delta_xy = loc_delta_xy
        self.min_loc_xy = min_loc_xy
        self.max_idx_xy = tf.cast(max_idx_xy,dtype=tf.float32)
        self.min_loc_z = min_loc_z
        self.max_idx_z = max_idx_z
        self.loc_delta_z = loc_delta_z
        
    @tf.Module.with_name_scope
    def build(self, inputs_shape):
        print(self.name,inputs_shape)
        self.lti = heatmap_2d.LocationToIndex(self.loc_delta_xy, self.min_loc_xy, self.max_idx_xy)
        self.max_loc_xy = (self.max_idx_xy - 1) * self.loc_delta_xy + self.min_loc_xy
        self.max_loc_z = (self.max_idx_z - 1) * self.loc_delta_z + self.min_loc_z
        super().build(inputs_shape)

    @tf.function(experimental_autograph_options=tf.autograph.experimental.Feature.ALL, experimental_relax_shapes=True)
    @tf.Module.with_name_scope
    def call(self, inputs):
        gt_xy = inputs[:,:,0:2]
        gt_xy = tf.minimum(gt_xy,self.max_loc_xy)
        gt_xy = tf.maximum(gt_xy,self.min_loc_xy)
        
        indexes_xy = self.lti(gt_xy)
        
        inputs_shape = tf.shape(inputs)
        size = inputs_shape[0]*inputs_shape[1]
        loc_arr = tf.TensorArray(dtype=tf.float32, size=size, dynamic_size=False)
        index_arr = tf.TensorArray(dtype=tf.int32, size=size, dynamic_size=False)
        for b in tf.range(inputs_shape[0]):
            for kp in tf.range(inputs_shape[1]):
                arr_index = b*inputs_shape[1]+kp
                loc_value = inputs[b,kp,2]
                loc_arr = loc_arr.write(arr_index, loc_value)
                index = tf.stack([b,indexes_xy[b,kp,0],indexes_xy[b,kp,1]])
                index_arr = index_arr.write(arr_index, index)
        gt_loc_z = loc_arr.stack()
        gt_loc_z = tf.maximum(gt_loc_z, self.min_loc_z)
        gt_loc_z = tf.minimum(gt_loc_z, self.max_loc_z)
        gt_index_z = index_arr.stack()
        return gt_xy, gt_loc_z, gt_index_z

class PoseLossDepth(tf.keras.layers.Layer):
    
    def __init__(self, depth_bins):
        self.depth_bins = tf.cast(depth_bins, tf.int32)
        self.loc_map_z = heatmap_1d.LocationMap(bins = self.depth_bins, min_loc=-150,max_loc=150)
        super().__init__()
        
    @tf.Module.with_name_scope
    def build(self, inputs_shape):
        print(self.name,inputs_shape)
        feature_z_shape, gt_loc_z_shape, gt_index_z_shape = inputs_shape
        self.loss_z = heatmap_1d.VarianceLocationAndPossitionLoss(self.loc_map_z.loc_delta)
                
        self.call = tf.function(self.call,input_signature=[(tf.TensorSpec(shape=[None,feature_z_shape[1],feature_z_shape[2],self.depth_bins], dtype=tf.float32),
                                                            tf.TensorSpec(shape=[None], dtype=tf.float32),
                                                            tf.TensorSpec(shape=[None,3], dtype=tf.int32))])
        super().build(inputs_shape)

    # is a @tf.function with defined input shape
    @tf.Module.with_name_scope
    def call(self, inputs):
        features_z, gt_loc_z, gt_index_z = inputs

                        
        features_z = heatmap_1d.feature_to_location_propability_map(features_z)
        loc_map_z = self.loc_map_z(0.)
        mask = heatmap_1d.mask_from_index(gt_index_z, tf.shape(features_z)[0:3])
        prop_map_z = heatmap_1d.mask_propability_map(features_z, mask)
        gt_loc_exp_z = heatmap_1d.expand_gt(gt_index_z, gt_loc_z, tf.shape(prop_map_z)[0:3])
        loc_z = heatmap_1d.propability_map_to_location(prop_map_z, loc_map_z)
        loss_z = self.loss_z(prop_map_z, loc_z, loc_map_z, gt_loc_exp_z)
        
        feature_shape = tf.shape(features_z)

        loss_z = tf.concat(loss_z, axis=-1)
        loss_z = tf.gather_nd(loss_z,gt_index_z)
        loss_z = tf.reshape(loss_z, [feature_shape[0],-1,2])
        
        loc_z_point = tf.gather_nd(loc_z,gt_index_z)
        loc_z_point = tf.reshape(loc_z_point, [feature_shape[0],-1,1])
        
        loc_z_point_shape = tf.shape(loc_z_point)
        
        loss_z_sum = tf.reduce_sum(loss_z) + 0.001
        loss_z_batch = tf.reduce_sum(loss_z, axis=0)
        loss_z_factor = loss_z_batch*tf.cast(loc_z_point_shape[1], dtype = tf.float32)/loss_z_sum/ self.loc_map_z.max_loc
        
        hard_kp_loss_z = loss_z*loss_z_factor
        return hard_kp_loss_z, loc_z_point
    
class PoseLoss2D(tf.keras.layers.Layer):
    
    def __init__(self, xy_bins):
        self.loc_map_xy = heatmap_2d.LocationMap(bins=xy_bins, min_loc=[-150,-150],max_loc=[150, 150])
        super().__init__()
    
    @tf.Module.with_name_scope
    def build(self, inputs_shape):
        print(self.name,inputs_shape)
        features_xy_shape, gt_xy_shape = inputs_shape
        self.key_points = features_xy_shape[-1]
        
        self.loss_xy = heatmap_2d.VarianceLocationAndPossitionLoss(self.loc_map_xy.loc_delta)
                
        self.call = tf.function(self.call,input_signature=[(tf.TensorSpec(shape=[None,features_xy_shape[1],features_xy_shape[2],self.key_points], dtype=tf.float32),
                                                            tf.TensorSpec(shape=[None,self.key_points,2], dtype=tf.float32))])
        super().build(inputs_shape)

    # is a @tf.function with defined input shape
    @tf.Module.with_name_scope
    def call(self, inputs):
        features_xy, gt_xy = inputs
        features_xy = tf.transpose(features_xy, [3,0,1,2])
        gt_xy_per_keypoint = tf.transpose(gt_xy, [1,0,2])
        
        
        keypoint_loss_arr = tf.TensorArray(dtype=tf.float32, size=self.key_points, dynamic_size=False)
        loc_xy_arr = tf.TensorArray(dtype=tf.float32, size=self.key_points, dynamic_size=False)
        
        for kp in tf.range(self.key_points):
            feature_xy = features_xy[kp]
            gt_loc_xy = gt_xy_per_keypoint[kp]
            
            feature_xy = tf.expand_dims(feature_xy,axis=-1)
            feature_xy = heatmap_2d.feature_to_location_propability_map(feature_xy)
            loc_map_xy = self.loc_map_xy([0.,0.])
            loc_xy = heatmap_2d.propability_map_to_location(feature_xy,loc_map_xy)
            loss_xy = self.loss_xy(feature_xy, loc_xy, loc_map_xy, gt_loc_xy)
            keypoint_loss_arr = keypoint_loss_arr.write(kp,loss_xy)
            loc_xy_arr = loc_xy_arr.write(kp,loc_xy)
        loss_xy = keypoint_loss_arr.stack()
        loss_xy = tf.transpose(loss_xy,[2,0,1])
        loc_xy = loc_xy_arr.stack()
        loc_xy = tf.transpose(loc_xy,[1,0,2])
                
        loss_xy_sum = (tf.reduce_sum(loss_xy)+0.001)
        loss_xy_batch = tf.reduce_sum(loss_xy, axis=0)
        loss_xy_factor = loss_xy_batch*tf.cast(self.key_points, dtype = tf.float32)/loss_xy_sum / self.loc_map_xy.max_loc
        
        hard_kp_loss_xy = loss_xy*loss_xy_factor
        
        return hard_kp_loss_xy, loc_xy

class PoseLoss(tf.keras.layers.Layer):
    
    def __init__(self, key_points = 15, depth_bins = 10):
        super().__init__()
        self.key_points = tf.cast(key_points, dtype = tf.int32)
        self.depth_bins = tf.cast(depth_bins, dtype = tf.float32)
    
    @tf.Module.with_name_scope
    def build(self, inputs_shape):
        print(self.name,inputs_shape)
        feature_shape = inputs_shape[0]
        print(feature_shape)
        self.pose_loss_xy = PoseLoss2D(feature_shape[1:3])
        self.pose_loss_z = PoseLossDepth(self.depth_bins) #TODO key_points noetig?
        
        self.xy_loc_delta = self.pose_loss_xy.loc_map_xy.loc_delta
        self.xy_min_loc = self.pose_loss_xy.loc_map_xy.min_loc
        self.xy_bins = self.pose_loss_xy.loc_map_xy.bins
        
        self.z_loc_delta = self.pose_loss_z.loc_map_z.loc_delta
        self.z_min_loc = self.pose_loss_z.loc_map_z.min_loc
        self.z_bins = self.pose_loss_z.loc_map_z.bins
        
        self.kp_to_gt = KeypointBatchToPoseGT(self.xy_loc_delta, self.xy_min_loc, self.xy_bins,
                                              self.z_loc_delta, self.z_min_loc, self.z_bins)
        
        self.call = tf.function(self.call,input_signature=[(tf.TensorSpec(shape=[None,feature_shape[1],feature_shape[2],self.key_points+tf.cast(self.depth_bins, dtype = tf.int32)], dtype=tf.float32),
                                                            tf.TensorSpec(shape=[None,self.key_points,3], dtype=tf.float32))])
        super().build(inputs_shape)

    # is a @tf.function with defined input shape
    @tf.Module.with_name_scope
    def call(self, inputs):
        feature, gt_kp = inputs
        features_xy = feature[:,:,:,:self.key_points]
        features_z = feature[:,:,:,self.key_points:]


        gt_xy, gt_loc_z, gt_index_z = self.kp_to_gt(gt_kp)
        
        hard_kp_loss_xy, loc_xy = self.pose_loss_xy([features_xy, gt_xy])
        hard_kp_loss_z, loc_z = self.pose_loss_z([features_z, gt_loc_z, gt_index_z])
                
        loc_xyz = tf.concat([loc_xy,loc_z],axis=-1)
        
        limbs = limb_length(loc_xyz)
        sym_loss = symmetry_loss(limbs)
        
        tf.print("loss", hard_kp_loss_xy, hard_kp_loss_z)
                     
        return hard_kp_loss_xy, hard_kp_loss_z#, sym_loss

class PersonPosFromPose(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
    
    @tf.Module.with_name_scope
    def build(self, inputs_shape):
        print(self.name,inputs_shape)   
        super().build(inputs_shape)
        self.call = tf.function(self.call,input_signature=[(tf.TensorSpec([None, 15, 3],dtype=tf.float32))])
    
    # is a @tf.function with defined input shape
    @tf.Module.with_name_scope
    def call(self, inputs):
        person_poses = inputs
        print("tracing", self.name,person_poses.shape)
        
        person_pos = tf.reduce_mean(person_poses,axis=1)
        
        return person_pos
    
person_pos_from_pose = PersonPosFromPose()

class PersonPosToIndexes(tf.keras.layers.Layer): #TODO triggeres retracing
    def __init__(self):
        super().__init__()
    
    @tf.Module.with_name_scope
    def build(self, inputs_shape):
        print(self.name,inputs_shape)
        super().build(inputs_shape)
        self.call = tf.function(self.call,input_signature=[(tf.TensorSpec([None],dtype=tf.float32),tf.TensorSpec([None, 3],dtype=tf.float32)),tf.TensorSpec([3],dtype=tf.int32),tf.TensorSpec([3],dtype=tf.float32),tf.TensorSpec([3],dtype=tf.float32)])
     
    @tf.Module.with_name_scope
    def call(self, inputs, max_indexes, min_loc_xyz, loc_delta_xyz):
        batch_index, person_pos = inputs
        print("tracing", self.name, batch_index.shape, person_pos.shape)
        print("and params", max_indexes.shape, min_loc_xyz.shape, loc_delta_xyz.shape)
        
        indices = (person_pos - min_loc_xyz) / loc_delta_xyz
        indices = tf.maximum(indices, 0)
        max_index = tf.cast(max_indexes, dtype=tf.float32)
        indices = tf.minimum(indices, max_index)
        batch_index = tf.expand_dims(batch_index,axis=1)
        indices = tf.concat([batch_index,indices],axis=-1)
        indices = tf.cast(indices + 0.5, dtype=tf.int64)
        return indices

person_pos_to_indexes = PersonPosToIndexes()

class PersonPosToHeatMap(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
    
    @tf.Module.with_name_scope
    def build(self, inputs_shape):
        print(self.name,inputs_shape)
        super().build(inputs_shape)
        
    @tf.function(experimental_autograph_options=tf.autograph.experimental.Feature.ALL, experimental_relax_shapes=True)
    @tf.Module.with_name_scope
    def call(self, inputs, feature_shape, min_loc_xyz, loc_delta_xyz):
        batch_index, person_pos = inputs
        max_indexes = feature_shape[1:] - 1
        indices = person_pos_to_indexes([batch_index, person_pos],max_indexes, min_loc_xyz, loc_delta_xyz)
        heat_map = tf.SparseTensor(indices = indices, values = tf.ones(tf.shape(indices)[0]), dense_shape = tf.cast(feature_shape, dtype =tf.int64))
        heat_map = tf.sparse.to_dense(heat_map, validate_indices=False)
        return heat_map

person_pos_to_heat_map = PersonPosToHeatMap()

class HeatMapToWeights(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
    
    @tf.Module.with_name_scope
    def build(self, inputs_shape):
        print(self.name,inputs_shape)
        super().build(inputs_shape)
        
    @tf.function(experimental_autograph_options=tf.autograph.experimental.Feature.ALL, experimental_relax_shapes=True)
    @tf.Module.with_name_scope
    def call(self, inputs):
        heatmap = inputs
        heatmap_shape = tf.shape(heatmap)
        ones = tf.ones(heatmap_shape)
        
        heatmap_3d = tf.expand_dims(heatmap,axis=-1)
        dilated_heatmap_3d = tf.nn.max_pool3d(heatmap_3d, ksize=3, strides=1, padding="SAME", name='dilation')
        dilated_heatmap = dilated_heatmap_3d[...,0]
        
        nr_persons = tf.reduce_sum(heatmap, axis=[1,2,3])
        nr_non_zeros = nr_persons * 3*3*3
        nr_positions = tf.cast(tf.reduce_prod(heatmap_shape[1:]),dtype=tf.float32)
        scale_negative = nr_non_zeros / nr_positions
        scale_positive = 1 - nr_persons / nr_positions
        scale_negative = tf.reshape(scale_negative,[-1,1,1,1])
        scale_positive = tf.reshape(scale_positive,[-1,1,1,1])
        
        negative_weights = (ones - dilated_heatmap) * scale_negative
        positive_weights = heatmap * scale_positive
        weights = negative_weights + positive_weights
        return weights

heat_map_to_weights = HeatMapToWeights()

class PersonLoss(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
    
    @tf.Module.with_name_scope
    def build(self, inputs_shape):
        print(self.name,inputs_shape)
        feature_shape, (batch_index_shape, pos_shape) = inputs_shape
        self.call = tf.function(self.call,input_signature=[(tf.TensorSpec(shape=[None,feature_shape[1],feature_shape[2],feature_shape[3]], dtype=tf.float32),
                                                            (tf.TensorSpec(shape=[None], dtype=tf.float32),
                                                             tf.TensorSpec(shape=[None,pos_shape[1]], dtype=tf.float32))),
                                                           tf.TensorSpec(shape=[3], dtype=tf.float32),
                                                           tf.TensorSpec(shape=[3], dtype=tf.float32)])
        
        super().build(inputs_shape)
        
        
    # is a @tf.function with defined input shape
    @tf.Module.with_name_scope
    def call(self, inputs, min_loc_xyz, loc_delta_xyz):
        feature, (batch_index, person_pos) = inputs
        feature_shape = tf.shape(feature)
        heat_map = person_pos_to_heat_map((batch_index, person_pos), feature_shape, min_loc_xyz, loc_delta_xyz)
        loss_weights = heat_map_to_weights(heat_map)
        se = (feature - heat_map)**2
        weighted_loss = se * loss_weights
        nr_persons = tf.reduce_sum(heat_map, axis=[1,2,3])
        loss = tf.reduce_sum(weighted_loss, axis=[1,2,3]) / nr_persons
        return loss

person_loss = PersonLoss()

def main():
    #tf.config.experimental_run_functions_eagerly(True)
    #test_pose_loss()
    test_pos_loss()
    
def test_pos_loss():
    min_loc_xyz=tf.constant([0,0,50],dtype=np.float32)
    loc_delta_xyz=tf.constant([150,150,150],dtype=np.float32)
    batches = 4
    feature = np.zeros([batches,10,10,10],dtype=np.float32)
    
    pos_01=tf.constant([5,5,5],dtype=np.float32)
    pos_02=tf.constant([2,2,8],dtype=np.float32)
    pos_11=tf.constant([8,1,0],dtype=np.float32)
    pos_12=tf.constant([3,7,1],dtype=np.float32)
    pos_21=tf.constant([5,5,0],dtype=np.float32)
    pos_31=tf.constant([5,5,0],dtype=np.float32)
    pos_32=tf.constant([7,7,0],dtype=np.float32)
    
    feature[0,5,5,5]=1
    feature[0,2,2,8]=1
    feature[1,8,1,0]=1
    feature[1,3,7,1]=0.8
    feature[2,3,3,0]=1
    feature[3,5,5,0]=1
    feature[3,7,7,0]=1
    
    feature = tf.cast(feature, dtype=tf.float32)
    
    person_poses = [[min_loc_xyz+loc_delta_xyz*pos_01 for _ in range(15)],
                    [min_loc_xyz+loc_delta_xyz*pos_02 for _ in range(15)],
                    [min_loc_xyz+loc_delta_xyz*pos_11 for _ in range(15)],
                    [min_loc_xyz+loc_delta_xyz*pos_12 for _ in range(15)],
                    [min_loc_xyz+loc_delta_xyz*pos_21 for _ in range(15)],
                    [min_loc_xyz+loc_delta_xyz*pos_31 for _ in range(15)],
                    [min_loc_xyz+loc_delta_xyz*pos_32 for _ in range(15)]]
    
    person_poses = tf.cast(person_poses, dtype=tf.float32)
    
    batch_indexes = tf.cast([0,0,1,1,2,3,3], dtype=tf.float32)
    
    person_pos = person_pos_from_pose(person_poses)
    
    pos_loss = person_loss([feature,(batch_indexes, person_pos)],min_loc_xyz,loc_delta_xyz)
    print(pos_loss)
    
    feature = tf.Variable(np.ones([batches,10,10,10]), trainable=True, dtype=tf.float32)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)

    @tf.function
    def train_loop(epochs):
        for epoch in tf.range(epochs):
            with tf.GradientTape() as tape:
                loss = person_loss([feature,(batch_indexes, person_pos)],min_loc_xyz,loc_delta_xyz)
            
            gradients = tape.gradient(loss, feature)
            [capped_gradients], _ = tf.clip_by_global_norm([gradients], 10.)
            optimizer.apply_gradients([(capped_gradients,feature)])
            tf.print(loss)
            
    train_loop(500)
    
    heat_map = person_pos_to_heat_map((batch_indexes, person_pos), tf.shape(feature),min_loc_xyz, loc_delta_xyz)
    pos_gt = tf.unstack(heat_map,axis=0)
    pos_maps = tf.unstack(feature,axis=0)
    for pos_batch, gt_batch in zip(pos_maps, pos_gt):
        for pos, gt in zip(pos_batch, gt_batch):
            plt.imshow(tf.concat([pos, gt],axis=-1))
            plt.show()


def test_pose_loss():
    batches = 4
    keypoints = 15
    feature = np.zeros([batches,10,10,10+keypoints],dtype=np.float32)
    #x
    feature[0,5,5,0]=0.5
    feature[0,6,6,0]=0.5
    feature[1,4,4,0]=0.5
    feature[1,7,7,0]=0.5
    feature[2,5,5,0]=1
    feature[3,5,5,0]=1
    feature[3,6,6,0]=1
    
    #z
    feature[0,6,6,keypoints+5]=0.5
    feature[0,6,6,keypoints+6]=0.5
    feature[1,6,6,keypoints+4]=0.5
    feature[1,6,6,keypoints+7]=0.5
    feature[2,5,5,keypoints+5]=1
    feature[3,6,6,keypoints+5]=1
    feature[3,6,6,keypoints+6]=1
    
    x = y = z = 1500 - 1500
    kp_gt = tf.constant([[ [x,y,z] for kp in range(keypoints)] for b in range(batches)],dtype = tf.float32)
    
    tl = PoseLoss()
    
    loss = tl([feature,kp_gt])
    loss = tl([feature,kp_gt])
    
    batches = 2
    x = y = z = 250 - 1500
    kp_gt = tf.constant([[ [x+545*kp,y+504*kp,z+200*kp] for kp in range(keypoints)] for b in range(batches)],dtype = tf.float32)
    
    
    feature = tf.Variable(np.ones([batches,10,10,10+keypoints]), trainable=True, dtype=tf.float32)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)

    @tf.function
    def train_loop(epochs):
        for epoch in tf.range(epochs):
            with tf.GradientTape() as tape:
                loss = tl([feature,kp_gt])
            
            gradients = tape.gradient(loss, feature)
            [capped_gradients], _ = tf.clip_by_global_norm([gradients], 10.)
            optimizer.apply_gradients([(capped_gradients,feature)])
            tf.print(loss)
            
    train_loop(100)
    
    kp_to_gt = KeypointBatchToPoseGT(tl.xy_loc_delta, tl.xy_min_loc, tl.xy_bins, tl.z_loc_delta, tl.z_min_loc, tl.z_bins)
    gt_xy, gt_loc_z, gt_index_z = kp_to_gt(kp_gt)
    prob_maps = tf.unstack(feature,axis=-1)
    kps = tf.unstack(gt_xy,axis=1)
    for prob_map_batch, kp_batch in zip(prob_maps, kps):
        prob_map_batch = heatmap_2d.feature_to_location_propability_map(prob_map_batch)
        loc_map_xy = tl.pose_loss_xy.loc_map_xy([0.,0.])
        loc_xy_batch = heatmap_2d.propability_map_to_location(tf.expand_dims(prob_map_batch,axis=-1),loc_map_xy)
        for prob_map, kp, loc in zip(prob_map_batch,kp_batch,loc_xy_batch):
            print(kp, loc)
            plt.imshow(prob_map)
            plt.show()
    
    feature_z = feature[...,keypoints:]
    prop_map_z = heatmap_1d.feature_to_location_propability_map(feature_z)

    gt_loc_z = heatmap_1d.expand_gt(gt_index_z, gt_loc_z, tf.shape(prop_map_z)[0:3])
    
    loc_map_z = tl.pose_loss_z.loc_map_z(0.)
    loc_z = heatmap_1d.propability_map_to_location(prop_map_z, loc_map_z)
    for kp, loc in zip(gt_loc_z,loc_z):
        plt.imshow(tf.concat([kp[...,0],loc[...,0]],axis=1))
        plt.show()
            

class LossTestTrainingsModel(tf.keras.layers.Layer):
    def __init__(self, keypoints = 15, depth_bins = 10):
        super().__init__()
        self.keypoints = tf.cast(keypoints, dtype = tf.int32)
        self.depth_bins = tf.cast(depth_bins, dtype = tf.int32)
        
    def build(self, inputs_shape):
        feature_shape, gt_shape = inputs_shape
        self.representation = tf.Variable(tf.ones([1, 10, 10, self.depth_bins+self.keypoints]), trainable=True, dtype = tf.float32)
        self.loss = PoseLoss(self.keypoints, self.depth_bins)
        self.call = tf.function(self.call,input_signature=[(tf.TensorSpec(shape=None, dtype=tf.float32), tf.TensorSpec(shape=[None,self.keypoints,3], dtype=tf.float32))])
        super().build(inputs_shape)

    def call(self, inputs):
        feature, gt_target = inputs
        print("Tracing with", feature, gt_target)
        batched_repr = tf.repeat(self.representation, repeats = tf.shape(feature)[0], axis=0)
        return self.loss([batched_repr, gt_target])
    

if __name__ == "__main__":
    main()