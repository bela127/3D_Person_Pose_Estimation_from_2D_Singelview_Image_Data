import time

import numpy as np
import tensorflow as tf

import ShAReD_Net.model.base as base
import ShAReD_Net.training.loss.base as loss_base
import ShAReD_Net.model.layer.aggregation as aggregation

class TrainingModel(tf.keras.layers.Layer):
    
    def __init__(self, max_loc_xy, name = "TrainingModel", dtype = tf.float32, **kwargs):  
        super().__init__(name = name,dtype=dtype, **kwargs)
        self.max_loc_xy = tf.cast(max_loc_xy,dtype=tf.float32)
        
        self.low_level_gpu = "/device:GPU:2"
        self.high_level_gpus = ["/device:GPU:0","/device:GPU:1"]
        self.target_gpu = "/device:GPU:2"
        
        self.loss_gpu = "/device:CPU:0"
    
    def build(self, input_shape):
        super().build(input_shape)
        print(self.name,input_shape)
        images_shape, (batch_indexes_shape, gt_poses_shape), focal_length_shape, crop_factor_shape = input_shape
        
        self.base_model = base.BaseModel(low_level_gpu = self.low_level_gpu,
                                         high_level_gpus = self.high_level_gpus,
                                         target_gpu = self.target_gpu)
        
        self.detection_loss = loss_base.person_loss
        self.estimator_loss = loss_base.PoseLoss(key_points = self.base_model.key_points, depth_bins = self.base_model.z_bins)
        self.roi_extractor = aggregation.CropROI3D(roi_size=[1,11,11,1])
        
        self.call = tf.function(self.call,input_signature=[(tf.TensorSpec([None, None, None, 3], dtype=tf.float32), (tf.TensorSpec([None], dtype=tf.float32), tf.TensorSpec([None, 15, 3], dtype=tf.float32)), tf.TensorSpec([], dtype=tf.float32), tf.TensorSpec([], dtype=tf.float32))])
    
    def call(self, inputs):
        training=True
        images, (batch_indexes, gt_poses), focal_length, crop_factor = inputs
        print("tracing", self.name,images.shape, (batch_indexes.shape, gt_poses.shape), focal_length.shape, crop_factor.shape)
        
        feature3d = self.base_model.extractor([images, focal_length, crop_factor], training=training)
        detection = self.base_model.detector(feature3d, training = training)
        
        with tf.device(self.loss_gpu):
            images_shape = tf.shape(images)
            xy_step = self.max_loc_xy / tf.cast(images_shape[1:3],dtype=tf.float32)
            min_loc_xyz = tf.stack([0,0,self.base_model.min_dist],axis = 0)
            loc_delta_xyz = tf.stack([xy_step[0],xy_step[1],self.base_model.dist_step],axis = 0)

            gt_person_pos = loss_base.person_pos_from_pose(gt_poses)
            detection_loss = self.detection_loss([detection, (batch_indexes, gt_person_pos)],min_loc_xyz,loc_delta_xyz)

        with tf.device(self.target_gpu):
            expanded3d = self.base_model.expand(feature3d)
            
        with tf.device(self.loss_gpu):
            roi_indexes = loss_base.person_pos_to_indexes([batch_indexes, gt_person_pos], tf.shape(detection)[1:], min_loc_xyz, loc_delta_xyz)
            roi_feature = self.roi_extractor([expanded3d, roi_indexes])
            
        estimates = self.base_model.estimator(roi_feature, training=training)
        
        with tf.device(self.loss_gpu):
            estimator_pose = gt_poses - gt_person_pos[:,None,:]
            estimator_loss = self.estimator_loss([estimates, estimator_pose])
                
        tf.print("end loss", detection_loss, estimator_loss)
        return detection_loss, estimator_loss
        
    def get_config(self):
        config = super().get_config()
        return config



def test(op, optimizer, **kwargs):
    @tf.function
    def run(inputs):
        with tf.GradientTape() as tape:
            outputs = op(inputs, **kwargs)
        g = tape.gradient(outputs, op.trainable_variables)
        optimizer.apply_gradients(zip(g, op.trainable_variables))
        return outputs, g
    return run
    

def main():
    #tf.config.experimental_run_functions_eagerly(True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    print("TrainingModel")
    images = tf.constant(100.,shape=[1,384,384,3])
    
    min_loc_xyz=tf.constant([0,0,50],dtype=tf.float32)
    loc_delta_xyz=tf.constant([150,150,150],dtype=tf.float32)
    
    pos_01=tf.constant([5,5,5],dtype=tf.float32)
    pos_02=tf.constant([2,2,8],dtype=tf.float32)
    pos_11=tf.constant([8,1,0],dtype=tf.float32)
    pos_12=tf.constant([3,7,1],dtype=tf.float32)
    pos_21=tf.constant([5,5,0],dtype=tf.float32)
    pos_31=tf.constant([5,5,0],dtype=tf.float32)
    pos_32=tf.constant([7,7,0],dtype=tf.float32)
    
    person_poses = [[min_loc_xyz+loc_delta_xyz*pos_01 for _ in range(15)],
                    [min_loc_xyz+loc_delta_xyz*pos_02 for _ in range(15)],
                    [min_loc_xyz+loc_delta_xyz*pos_11 for _ in range(15)],
                    [min_loc_xyz+loc_delta_xyz*pos_12 for _ in range(15)],
                    [min_loc_xyz+loc_delta_xyz*pos_21 for _ in range(15)],
                    [min_loc_xyz+loc_delta_xyz*pos_31 for _ in range(15)],
                    [min_loc_xyz+loc_delta_xyz*pos_32 for _ in range(15)]]
    
    person_poses = tf.cast(person_poses, dtype=tf.float32)
    
    batch_indexes = tf.cast([0,0,1,1,2,3,3], dtype=tf.float32)
    
    focal_length = tf.cast(5, dtype=tf.float32)
    crop_factor = tf.cast(0.5, dtype=tf.float32)
    
    inputs = [images, (batch_indexes, person_poses), focal_length, crop_factor]
    msf = TrainingModel(max_loc_xy = [2500,2500])    
    test_msf= test(msf, optimizer, training = True)
    out, g = test_msf(inputs)
    print(msf.count_params())
    print(out.shape)
    print("TrainingModel")
    
    time_start = time.time()
    for i in range(10):
        out = test_msf(inputs)
    time_end = time.time()
    print(time_end - time_start)
    
    

if __name__ == '__main__':
    main()