import time

import numpy as np
import tensorflow as tf
keras = tf.keras

import ShAReD_Net.model.base as base
import ShAReD_Net.training.loss.base as loss_base

class EvaluationModelETE(keras.layers.Layer):
    
    def __init__(self, name = "EvaluationModelETE", **kwargs):   
        self.base_model = base.base_model
        super().__init__(name = name, **kwargs)
        
    def build(self, input_shape):
        self.estimator_loss = loss_base.PoseLoss(key_points = self.base_model.key_points, depth_bins = self.base_model.z_bins)
        super().build(input_shape)
    
    @tf.function
    def call(self, inputs, training=None):
        images, gt_poses = inputs
        feature3d = self.base_model.extractor(images)
        detection = self.base_model.detector(feature3d)
        
        person_pos = self.detect_to_pos(detection)
        gt_person_pos = self.person_pos(gt_poses)
        detection_loss = self.detection_loss([person_pos, gt_person_pos])

        expanded3d = self.base_model.expand(feature3d)
        roi_feature = self.roi_extractor([expanded3d, person_pos])
        
        estimates = self.base_model.estimator(roi_feature)
        
        poses = self.pose_from_estimate(estimates)
        gt_per_person_poses = self.pose_per_person(gt_poses)
        estimator_loss = self.estimator_loss([poses, gt_per_person_poses])
        
        return detection_loss, estimator_loss
        
    def get_config(self):
        config = super().get_config()
        return config


class EvaluationModelFGT(keras.layers.Layer):
    
    def __init__(self, name = "EvaluationModelFGT", **kwargs):   
        self.base_model = base.base_model
        super().__init__(name = name, **kwargs)
        
    def build(self, input_shape):
        self.estimator_loss = loss_base.PoseLoss(key_points = self.base_model.key_points, depth_bins = self.base_model.z_bins)
        super().build(input_shape)
    
    @tf.function
    def call(self, inputs, training=None):
        images, gt_poses = inputs
        feature3d = self.base_model.extractor(images)
        detection = self.base_model.detector(feature3d)
        
        person_pos = self.detect_to_pos(detection)
        gt_person_pos = self.person_pos(gt_poses)
        detection_loss = self.detection_loss([person_pos, gt_person_pos])

        expanded3d = self.base_model.expand(feature3d)
        roi_feature = self.roi_extractor([expanded3d, gt_person_pos])
        
        estimates = self.base_model.estimator(roi_feature)
        
        poses = self.pose_from_estimate(estimates)
        gt_per_person_poses = self.pose_per_person(gt_poses)
        estimator_loss = self.estimator_loss([poses, gt_per_person_poses])
        
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
    inputs = tf.constant(100.,shape=[1,100,100,3])
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