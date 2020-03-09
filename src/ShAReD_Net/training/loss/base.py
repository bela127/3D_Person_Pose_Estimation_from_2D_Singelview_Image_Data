import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


import ShAReD_Net.model.layer.heatmap_1d as heatmap_1d
import ShAReD_Net.model.layer.heatmap_2d as heatmap_2d

class LimbLength(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        
    def build(self, inputs_shape):
        size = inputs_shape[0]*inputs_shape[1]
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
    def call(self, inputs):
        size = inputs.shape[0]*inputs.shape[1]
        limb_arr = tf.TensorArray(dtype=tf.float32, size=size, dynamic_size=False)
        for b in tf.range(inputs.shape[0]):
            for kp in tf.range(inputs.shape[1]):
                limb_length = tf.sqrt(tf.reduce_sum((inputs[b,kp] - inputs[b,self.limb_connection[kp]])**2))
                index = b*inputs.shape[1]+kp
                limb_arr = limb_arr.write(index,limb_length)
        limbs = limb_arr.stack()
        limbs = tf.reshape(limbs, [inputs.shape[0],inputs.shape[1],1])
        return limbs

limb_length = LimbLength()

class SymmetryLoss(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        
    def build(self, inputs_shape):
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
    def call(self, inputs):
        size = inputs.shape[0]*inputs.shape[1]
        sym_loss_arr = tf.TensorArray(dtype=tf.float32, size=size, dynamic_size=False)
        for b in tf.range(inputs.shape[0]):
            for kp in tf.range(inputs.shape[1]):
                per_limb_loss = (inputs[b,kp] - inputs[b,self.sym_connection[kp]])**2
                sym_loss_arr = sym_loss_arr.write(b*inputs.shape[1]+kp,per_limb_loss)
        sym_loss = sym_loss_arr.stack()
        sym_loss = tf.reshape(sym_loss, [inputs.shape[0],inputs.shape[1],1])
        return sym_loss

symmetry_loss = SymmetryLoss()

class KeypointBatchToGT(tf.keras.layers.Layer):
    def __init__(self, loc_delta, max_indexes_xy, max_value_z):
        super().__init__()
        self.loc_delta = loc_delta
        self.max_index = tf.cast(max_indexes_xy,dtype=tf.float32)
        self.max_value = max_value_z
        
    def build(self, inputs_shape):
        self.lti = heatmap_2d.LocationToIndex(self.loc_delta, self.max_index)
        self.max_loc = (self.max_index - 1) * self.loc_delta
        super().build(inputs_shape)

    @tf.function 
    def call(self, inputs):
        gt_xy = inputs[:,:,0:2]
        gt_xy = tf.minimum(gt_xy,self.max_loc)
        
        indexes_xy = self.lti(gt_xy)
        
        size = inputs.shape[0]*inputs.shape[1]
        loc_arr = tf.TensorArray(dtype=tf.float32, size=size, dynamic_size=False)
        index_arr = tf.TensorArray(dtype=tf.int32, size=size, dynamic_size=False)
        for b in tf.range(inputs.shape[0]):
            for kp in tf.range(inputs.shape[1]):
                arr_index = b*inputs.shape[1]+kp
                loc_value = inputs[b,kp,2]
                loc_arr = loc_arr.write(arr_index, loc_value)
                index = tf.stack([b,indexes_xy[b,kp,0],indexes_xy[b,kp,1]])
                index_arr = index_arr.write(arr_index, index)
        gt_loc_z = loc_arr.stack()
        gt_loc_z = tf.minimum(gt_loc_z, self.max_value)
        gt_index_z = index_arr.stack()
        return gt_xy, gt_loc_z, gt_index_z

class TrainingsLoss(tf.keras.layers.Layer):
    
    def __init__(self, key_points = 15, depth_bins = 10):
        super().__init__()
        self.key_points = key_points
        self.depth_bins = depth_bins
        
    def build(self, inputs_shape):
        feature_shape = inputs_shape[0]
        self.loc_map_z = heatmap_1d.LocationMap(bins = self.depth_bins)
        self.loss_z = heatmap_1d.VarianceLocationAndPossitionLoss(self.loc_map_z.loc_delta)
        
        self.loc_map_xy = heatmap_2d.LocationMap(bins=feature_shape[1:3])
        self.loss_xy = heatmap_2d.VarianceLocationAndPossitionLoss(self.loc_map_xy.loc_delta)
        
        max_loc_z = self.loc_map_z.loc_delta * (self.depth_bins - 1)
        self.kp_to_gt = KeypointBatchToGT(self.loc_map_xy.loc_delta,feature_shape[1:3], max_loc_z)
        super().build(inputs_shape)

    @tf.function
    def call(self, inputs):
        feature, gt_kp = inputs
        features_xy = feature[:,:,:,0:self.key_points]
        features_xy = tf.transpose(features_xy, [3,0,1,2])
        
        gt_xy, gt_loc_z, gt_index_z = self.kp_to_gt(gt_kp)
        
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
        loss_xy = tf.transpose(loss_xy,[1,0])
        loc_xy = loc_xy_arr.stack()
        loc_xy = tf.transpose(loc_xy,[1,0,2])
        
        features_z = feature[:,:,:,self.key_points:]
        features_z = heatmap_1d.feature_to_location_propability_map(features_z)
        loc_map_z = self.loc_map_z(0.)
        mask = heatmap_1d.mask_from_index(gt_index_z, tf.shape(features_z)[0:3])
        prop_map_z = heatmap_1d.mask_propability_map(features_z, mask)
        gt_loc_z = heatmap_1d.expand_gt(gt_index_z, gt_loc_z, tf.shape(prop_map_z)[0:3])
        loc_z = heatmap_1d.propability_map_to_location(prop_map_z, loc_map_z)
        loss_z = self.loss_z(prop_map_z, loc_z, loc_map_z, gt_loc_z)
        
        
        loss_z = tf.gather_nd(loss_z,gt_index_z)
        loss_z = tf.reshape(loss_z, [feature.shape[0],-1])
        
        loc_z_point = tf.gather_nd(loc_z,gt_index_z)
        loc_z_point = tf.reshape(loc_z_point, [feature.shape[0],-1,1])
        
        loss_z_sum = (tf.reduce_sum(loss_z)+0.001)/self.key_points
        loss_z_batch = tf.reduce_sum(loss_z, axis=0)
        loss_z_factor = loss_z_batch/loss_z_sum
        
        loss_xy_sum = (tf.reduce_sum(loss_xy)+0.001)/self.key_points
        loss_xy_batch = tf.reduce_sum(loss_xy, axis=0)
        loss_xy_factor = loss_xy_batch/loss_xy_sum
        
        hard_kp_loss_z = loss_z_batch*loss_z_factor
        hard_kp_loss_xy = loss_xy_batch*loss_xy_factor
        
        loc_xyz = tf.concat([loc_xy,loc_z_point],axis=-1)
        
        limbs = limb_length(loc_xyz)
        sym_loss = symmetry_loss(limbs)
        sym_loss = tf.reduce_sum(sym_loss)
        
        loss = tf.reduce_sum(hard_kp_loss_xy + hard_kp_loss_z) #+ sym_loss
        
        return loss

        

def main():
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
    
    x = y = z = 250
    kp_gt = tf.constant([[ [x+245*kp,y+204*kp,z+200*kp] for kp in range(keypoints)] for b in range(batches)],dtype = tf.float32)
    
    tl = TrainingsLoss()
    
    loss = tl([feature,kp_gt])
    loss = tl([feature,kp_gt])
    
    feature = tf.Variable(np.ones([batches,10,10,10+keypoints]), trainable=True)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.5)

    @tf.function
    def train_loop(epochs):
        for epoch in tf.range(epochs):
            with tf.GradientTape() as tape:
                loss = tl([feature,kp_gt])
            
            gradients = tape.gradient(loss, feature)
            [capped_gradients], _ = tf.clip_by_global_norm([gradients], 10.)
            optimizer.apply_gradients([(capped_gradients,feature)])
            tf.print(loss)
            
    train_loop(400)
    prob_maps = tf.unstack(feature,axis=-1)
    for prob_map_batch in prob_maps:
        prob_map_batch = heatmap_2d.feature_to_location_propability_map(prob_map_batch)
        loc_map_xy = tl.loc_map_xy([0.,0.])
        loc_xy = heatmap_2d.propability_map_to_location(tf.expand_dims(prob_map_batch,axis=-1),loc_map_xy)
        print(loc_xy)
        for prob_map in prob_map_batch:
            plt.imshow(prob_map)
            plt.show()
            

class LossTestTrainingsModel(tf.keras.Model):
    def __init__(self, keypoints = 15, depth_bins = 10):
        super().__init__()
        self.keypoints = tf.cast(keypoints, dtype = tf.float32)
        self.depth_bins = tf.cast(depth_bins, dtype = tf.float32)

        
    def build(self, inputs_shape):
        feature_shape, gt_shape = inputs_shape
        self.representation = tf.Variable(np.ones([feature_shape[0], 10, 10, self.depth_bins+self.keypoints]), trainable=True, dtype = tf.float32)
        self.loss = TrainingsLoss(self.keypoints, self.depth_bins)
        super().build(inputs_shape)

    
    @tf.function
    def call(self, inputs):
        feature, gt_target = inputs
        return self.loss([self.representation, gt_target])
    

if __name__ == "__main__":
    main()