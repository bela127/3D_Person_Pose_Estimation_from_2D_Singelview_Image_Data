import time

import numpy as np
import tensorflow as tf
keras = tf.keras

import ShAReD_Net.model.modules.extractor as extractor
import ShAReD_Net.model.modules.detector as detector
import ShAReD_Net.model.modules.estimator as estimator
import ShAReD_Net.model.layer.aggregation as aggregation

class BaseModel(keras.Model):
    
    def __init__(self, key_points = 15,
                       xyz_bins = [20,20,10],
                       est_dbc = 4,
                       est_dfc = 8,
                       
                       stage_count = 2,
                       ext_dbc = 2,
                       ext_dfc = 8,
                       
                       min_dist=100,
                       dist_count = 5, 
                       dist_step = 500,
                       image_hight0 = 384,
                 
                       low_level_gpu = None,
                       high_level_gpus = None,
                       target_gpu = None,
                       
                       name = "PoseEstimator", **kwargs):
          
        self.key_points = tf.cast(key_points, dtype = tf.int32)
        self.z_bins = tf.cast(xyz_bins[2], dtype = tf.int32)
        self.xy_bins = tf.cast(xyz_bins[0:2], dtype = tf.int32)
        self.est_dbc = est_dbc
        self.est_dfc = est_dfc
        
        self.ext_dbc = ext_dbc
        self.ext_dfc = ext_dfc
        self.stage_count = stage_count
        
        self.min_dist = tf.cast(min_dist, dtype = tf.float32)
        self.dist_count = tf.cast(dist_count, dtype = tf.float32)
        self.dist_step = tf.cast(dist_step, dtype = tf.float32)
        self.image_hight0 = tf.cast(image_hight0, dtype = tf.float32)
        
        self.low_level_gpu = low_level_gpu,
        self.high_level_gpus = high_level_gpus
        self.target_gpu = target_gpu
        
        super().__init__(name = name, **kwargs)
        
        self.extractor = extractor.MultiScaleFeatureExtractor(stages_count = self.stage_count, dense_blocks_count = self.ext_dbc, dense_filter_count = self.ext_dfc, distance_count = self.dist_count, image_hight0 = self.image_hight0, distance_steps = self.dist_step, min_dist = self.min_dist,low_level_gpu = self.low_level_gpu, high_level_gpus=self.high_level_gpus)
        self.detector = detector.PersonDetector(target_gpu = self.target_gpu)
        self.estimator = estimator.PoseEstimator(key_points = self.key_points, depth_bins = self.z_bins , xy_bins = self.xy_bins, dense_blocks_count = self.est_dbc, dense_filter_count = self.est_dfc, target_gpu = self.target_gpu)
        self.expand = aggregation.Expand3D()
        
    def get_config(self):
        config = super().get_config()
        config.update({'est_dbc': self.est_dbc,
                       'est_dfc': self.est_dfc,
                       'xy_bins': self.xy_bins,
                       'z_bins': self.z_bins,
                       'key_points': self.key_points,
                       
                       'stage_count': self.stage_count,
                       'ext_dbc': self.ext_dbc,
                       'ext_dfc': self.ext_dfc,
                       
                       'min_dist': self.min_dist,
                       'dist_count': self.dist_count,
                       'dist_step': self.dist_step,
                       'image_hight0': self.image_hight0,
                       })
        return config
    
base_model = BaseModel()
