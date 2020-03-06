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


class MaskToROI(tf.keras.layers.Layer):
    def __init__(self, roi_size, threshold = 0.5, name = "MaskToROI", **kwargs):
        super().__init__(name = name, **kwargs)
        self.roi_size = tf.cast(roi_size, dtype=tf.int32)
        self.threshold = tf.cast(threshold, dtype=tf.float32)
    
    def build(self, input_shape):
        self.roi_half_size = tf.cast(self.roi_size / 2, dtype=tf.int32)
        self.offset = tf.cast(self.roi_size % 2, dtype=tf.int32)
        super().build(input_shape)
    
        
    
    @tf.function
    def call(self, inputs, **kwargs):
        
        mask = inputs
        shape = mask.shape
        flat_mask = tf.reshape(mask, [-1, shape[1]*shape[2]])

        indexes = tf.where(flat_mask > self.threshold)
        id_x = tf.cast(indexes[:,1] / shape[2], dtype=tf.int32)
        id_y = tf.cast(indexes[:,1] % shape[2], dtype=tf.int32)
        id_b = tf.cast(indexes[:,0], dtype=tf.int32)

        poss = tf.stack([id_x,id_y],axis=-1)

        roi_lu = poss - self.roi_half_size
        roi_rd = poss + self.roi_half_size + self.offset

        roi = tf.stack([roi_lu, roi_rd],axis = 1)

        id_b = tf.reshape(id_b,[-1,1])
        id_b = tf.stack([id_b,id_b],axis = 1)
       
        roi = tf.concat([id_b,roi], axis=-1)

        return roi
    
    def get_config(self):
        config = super().get_config()
        config.update({"roi_size": self.roi_size,
                       "threshold": self.threshold,
                       })
        return config

if __name__ == "__main__":
       
    mask_to_roi = MaskToROI([5,5])

    mask = np.zeros([2,20,15])
    mask[0,5,14] = 1
    mask[0,8,4] = 1
    mask[0,18,10] = 1
    mask[1,5,14] = 1
    mask[1,8,4] = 1
    mask = tf.cast(mask, dtype=tf.float32)


    rois = mask_to_roi(mask)
    print(rois)
    crop_ROI = CropROI()
    features = tf.reshape(mask, [*mask.shape,1])
    crops = crop_ROI([features,rois])
    print(crops)
    for crop in crops:
        plt.imshow(tf.reshape(crop,[5,5]))
        plt.show()
        
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