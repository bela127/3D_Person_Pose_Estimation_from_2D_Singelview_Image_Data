import time

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from demo_roi import crop_ROI


class MaskToROI(tf.keras.layers.Layer):
    def __init__(self, roi_size, threshold = 0.5, name = "MaskToROI", **kwargs):
        super().__init__(name = name, **kwargs)
        self.roi_size = tf.cast(roi_size, dtype=tf.int32)
        self.threshold = tf.cast(threshold, dtype=self.dtype)
        self.roi_half_size = tf.cast(self.roi_size / 2, dtype=tf.int32)
        self.offset = tf.cast(self.roi_size % 2, dtype=tf.int32)
    
    @tf.function
    def call(self, inputs, **kwargs):
        super().call(inputs, **kwargs)
        
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

features = tf.reshape(mask, [*mask.shape,1])
crops = crop_ROI([features,rois])
print(crops)
for crop in crops:
    plt.imshow(tf.reshape(crop,[5,5]))
    plt.show()