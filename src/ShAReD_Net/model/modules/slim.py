import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

class Roi_Extractor(keras.layers.Layer):
    
    def __init__(self, roi_size=[31,31], name = "Roi_Extractor", **kwargs):
        super().__init__(name = name, **kwargs)
        self.roi_size = np.asarray(roi_size, dtype=np.int32)
        if tf.logical_or(tf.math.mod(self.roi_size[0], 2) != 1, tf.math.mod(self.roi_size[1], 2) != 1):
            raise ValueError(f"roi_size should be odd")

    
    def build(self, input_shape):
        print(self.name,input_shape)
        encoded_poses_shape, roi_centers_batch_shape = input_shape
        self.encoded_poses_shape = encoded_poses_shape
        
        self.roi_size_half = tf.cast(self.roi_size / 2, dtype=tf.int32)
        
        super().build(input_shape)

    tf.function(experimental_relax_shapes=True)
    def call(self, inputs):
        encoded_poses, roi_centers = inputs
                    
        size = tf.shape(roi_centers)[0]
        cuts_arr = tf.TensorArray(dtype=encoded_poses.dtype, size=size, dynamic_size=False)
        for i in tf.range(size):
            roi_center = roi_centers[i]
            
            batch = roi_center[0]
            roi_l = roi_center[1] - self.roi_size_half[0]
            roi_u = roi_center[2] - self.roi_size_half[1]
            roi_r = roi_center[1] + self.roi_size_half[0] + 1
            roi_d = roi_center[2] + self.roi_size_half[1] + 1
            
            slide_l = tf.minimum(tf.maximum(roi_l,0), tf.shape(encoded_poses)[1])
            slide_u = tf.minimum(tf.maximum(roi_u,0), tf.shape(encoded_poses)[2])
            slide_r = tf.maximum(tf.minimum(roi_r, tf.shape(encoded_poses)[1]),0)
            slide_d = tf.maximum(tf.minimum(roi_d, tf.shape(encoded_poses)[2]),0)
            
            cut = encoded_poses[batch, slide_l:slide_r, slide_u:slide_d, :]

            overlap_l = tf.maximum(slide_l - roi_l, 0)
            overlap_u = tf.maximum(slide_u - roi_u, 0)
            overlap_r = tf.maximum(roi_r - slide_r, 0)
            overlap_d = tf.maximum(roi_d - slide_d, 0)
            
            cut = tf.pad(cut, [[overlap_l, overlap_r], [overlap_u, overlap_d], [0,0]])

            cut.set_shape([self.roi_size[0], self.roi_size[1], self.encoded_poses_shape[-1]])
            cuts_arr = cuts_arr.write(i,cut)
            
        roi_cuts = cuts_arr.stack()
        return roi_cuts
        
    def get_config(self):
        config = super().get_config()
        return config