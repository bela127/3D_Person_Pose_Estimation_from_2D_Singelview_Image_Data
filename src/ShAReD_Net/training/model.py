import time

import numpy as np
import tensorflow as tf

import ShAReD_Net.model.base as base
import ShAReD_Net.training.loss.base as loss_base
import ShAReD_Net.model.layer.aggregation as aggregation

class TrainingModel(tf.keras.layers.Layer):
    
    def __init__(self, max_loc_xy, name = "TrainingModel", **kwargs):  
        self.max_loc_xy = max_loc_xy
        self.base_model = base.base_model
        super().__init__(name = name, **kwargs)
        
    def build(self, input_shape):
        print(self.name,inputs_shape)
        self.detection_loss = loss_base.person_loss
        self.estimator_loss = loss_base.PoseLoss(key_points = self.base_model.key_points, depth_bins = self.base_model.z_bins)
        self.roi_extractor = aggregation.CropROI3D(roi_size=[1,11,11,1])
        super().build(input_shape)
    
    @tf.function
    def call(self, inputs, training=None):
        images, (batch_indexes, gt_poses) = inputs
        feature3d = self.base_model.extractor(images, training=training)
        detection = self.base_model.detector(feature3d, training = training)
        
        images_shape = tf.shape(images)
        xy_step = self.max_loc_xy / images_shape[1:3]
        min_loc_xyz = tf.concat([0,0,self.base_model.min_dist],axis = 0)
        loc_delta_xyz = tf.concat([xy_step[0],xy_step[1],self.base_model.dist_step],axis = 0)
        
        gt_person_pos = loss_base.person_pos_from_pose(gt_poses)
        detection_loss = self.detection_loss([detection, (batch_indexes, gt_person_pos)],min_loc_xyz,loc_delta_xyz)
        
        expanded3d = self.base_model.expand(feature3d)
        roi_indexes = loss_base.person_pos_to_indexes([batch_indexes, gt_person_pos], tf.shape(detection)[1:], min_loc_xyz, loc_delta_xyz)
        roi_feature = self.roi_extractor([expanded3d, roi_indexes])
        
        estimates = self.base_model.estimator(roi_feature, training=training)
        
        estimator_loss = self.estimator_loss([estimates, gt_poses])
        
        loss = detection_loss + estimator_loss
        return loss
        
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
    inputs = tf.constant(100.,shape=[4,480,480,3])
    msf = TrainingModel()    
    test_msf= test(msf, optimizer, training = True)
    out = test_msf(inputs)
    print(msf.count_params())
    for image1, image2 in out[0]:
        print(image1.shape, image2.shape)
    print("TrainingModel")
    
    time_start = time.time()
    for i in range(10):
        out = test_msf(inputs)
    time_end = time.time()
    print(time_end - time_start)
    
    

if __name__ == '__main__':
    main()