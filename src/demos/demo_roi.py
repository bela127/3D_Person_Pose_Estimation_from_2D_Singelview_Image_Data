import time

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class CropROI(tf.keras.layers.Layer):
    def __init__(self, name = "CropROI", **kwargs):
        super().__init__(name = name, **kwargs)
    
    @tf.function
    def call(self, inputs, **kwargs):
        super().call(inputs, **kwargs)
        
        feature, ROIs = inputs

        def crop_roi(ROI):
            slide = tf.maximum(ROI,0)
            crop = feature[slide[0,0],slide[0,1]:slide[1,1],slide[0,2]:slide[1,2],:]
            rd_overlap = tf.maximum(ROI[1,1:] - feature.shape[1:-1],0)
            lu_overlap = tf.maximum(- ROI[0,1:],0)

            crop = tf.pad(crop, [[lu_overlap[0],rd_overlap[0]], [lu_overlap[1],rd_overlap[1]], [0,0]])
            return crop

        crops = tf.map_fn(crop_roi, ROIs, dtype=tf.float32)
        return crops
        
        
crop_ROI = CropROI()

if __name__ == "__main__":

    goal = tf.cast(np.zeros([5,5,1]),dtype=tf.float32)
    features = tf.cast(np.ones([1,20,20,1]),dtype=tf.float32)
    rois = tf.cast( [[[0,1,5],[0,6,10]],[[0,3,8],[0,8,13]],[[0,2,4],[0,7,9]]],
                    dtype=tf.int32
                    )

    time_start = time.time()
    for i in range(1000):
        roi_crop = crop_ROI([features, rois])
    time_end = time.time()
    print(time_end - time_start)

    with tf.GradientTape() as tape:
        tape.watch(features)
        roi_crop = crop_ROI([features, rois])
        loss = tf.keras.losses.mean_squared_error(roi_crop, goal)
    g = tape.gradient(loss, features)

    plt.imshow(tf.reshape(g,[20,20]))
    plt.show()

    print(tf.version.VERSION)