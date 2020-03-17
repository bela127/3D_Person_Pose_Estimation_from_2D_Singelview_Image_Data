import time

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import ShAReD_Net.training.loss.base as loss_base

    
class CropROI3D(tf.keras.layers.Layer):
    def __init__(self, roi_size = [1,9,9,1], name = "CropROI3D", **kwargs):
        self.roi_size = tf.cast(roi_size,dtype=tf.int32)
        super().__init__(name = name, **kwargs)
        
    def build(self, inputs_shape):
        self.roi_half_size = tf.cast(self.roi_size / 2, dtype=tf.int32)
        self.offset = tf.cast(self.roi_size % 2, dtype=tf.int32)
        super().build(inputs_shape)
    
    #@tf.function
    def call(self, inputs):
        feature3D, roi_indexes = inputs
        
        roi_lu = roi_indexes - self.roi_half_size
        roi_rd = roi_indexes + self.roi_half_size + self.offset
        
        rois = tf.stack([roi_lu, roi_rd],axis = 1)
        
        #TODO
        size = tf.shape(roi_indexes)[0]
        crops_arr = tf.TensorArray(dtype=tf.float32, size=size, dynamic_size=False)
        for i in tf.range(size):
            roi = rois[i,...]
            slide = tf.maximum(roi,0)
            crop = feature3D[slide[0,0],slide[0,3]:slide[1,3],slide[0,1]:slide[1,1],slide[0,2]:slide[1,2],:]
            feature3D_shape = tf.shape(feature3D)
            size_zxy = tf.stack([feature3D_shape[2],feature3D_shape[3],feature3D_shape[1]])
            rdb_overlap = tf.maximum(roi[1,1:] - size_zxy,0)
            luf_overlap = tf.maximum(- roi[0,1:],0)

            crop = tf.pad(crop, [ [luf_overlap[2],rdb_overlap[2]], [luf_overlap[0],rdb_overlap[0]], [luf_overlap[1],rdb_overlap[1]], [0,0]])
            crops_arr = crops_arr.write(i,crop)
        crops = crops_arr.stack()

        return crops

class CropROI2D(tf.keras.layers.Layer):
    def __init__(self, name = "CropROI2D", **kwargs):
        super().__init__(name = name, **kwargs)
    
    @tf.function
    def call(self, inputs):
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

class Interleave(tf.keras.layers.Layer):
    def __init__(self, name = "Interleave", **kwargs):
        super().__init__(name = name, **kwargs)
    
    def build(self, inputs_shape):
        res_shape, shc_shape = inputs_shape
        self.compress = tf.keras.layers.Convolution2D(res_shape[-1], 1, name="compress", padding='SAME', activation=tf.nn.leaky_relu, kernel_initializer=tf.initializers.he_normal(), bias_initializer=tf.initializers.he_uniform())
        super().build(inputs_shape)
    
    @tf.function
    def call(self, inputs):
        res, shc = inputs
        compressed = self.compress(shc)
        conc = tf.concat([res[...,tf.newaxis], compressed[...,tf.newaxis]], axis=-1)
        interleaved = tf.reshape(conc, [tf.shape(res)[0],tf.shape(res)[1],tf.shape(res)[2],-1])
        return interleaved
    
    def get_config(self):
        config = super().get_config()
        return config

class Combine3D(tf.keras.layers.Layer):
    def __init__(self, name = "Combine3D", **kwargs):
        super().__init__(name = name, **kwargs)
    
    def build(self, inputs_shape):
        super().build(inputs_shape)
    
    @tf.function
    def call(self, inputs):
        size = tf.shape(inputs[-1])[1:3]
        same_sized = []
        for feature in inputs:
            padded_feature = tf.image.resize_with_crop_or_pad(feature,size[0],size[1])
            same_sized.append(padded_feature)
        stacked = tf.stack(same_sized[...,tf.newaxis],axis=-1)
        feature_3d = tf.transpose(stacked,[0,4,1,2,3])
        return feature_3d
    
    def get_config(self):
        config = super().get_config()
        return config

combine3d = Combine3D()  
    
class Expand3D(tf.keras.layers.Layer):
    def __init__(self, name = "Expand3D", **kwargs):
        super().__init__(name = name, **kwargs)
    
    def build(self, inputs_shape):
        input_shape = inputs_shape
        self.tconf1 = tf.keras.layers.Conv3DTranspose(filters = input_shape[-1]*2/3,
                                        kernel_size = [5,3,3],
                                        strides = [2,1,1],
                                        padding='same',
                                        activation=tf.nn.leaky_relu,
                                        kernel_initializer=tf.initializers.he_uniform(),
                                        )
        self.tconf2 = tf.keras.layers.Conv3DTranspose(filters = input_shape[-1]/2,
                                        kernel_size = [3,3,3],
                                        strides = [1,1,1],
                                        padding='same',
                                        activation=tf.nn.leaky_relu,
                                        kernel_initializer=tf.initializers.he_uniform(),
                                        )
        self.conf1 = tf.keras.layers.Convolution3D(filters = input_shape[-1]/2,
                                      kernel_size = [1,1,1],
                                      strides = [1,1,1],
                                      padding='same',
                                      activation=tf.nn.leaky_relu,
                                      kernel_initializer=tf.initializers.he_uniform(),
                                      )
        super().build(inputs_shape)
    
    @tf.function
    def call(self, inputs):
        feature = inputs
        tconf1 = self.tconf1(feature)
        tconf2 = self.tconf2(tconf1)
        conf1 = self.conf1(tconf2)
        return conf1
    
    def get_config(self):
        config = super().get_config()
        return config
    
    
def main():
    #test_roi_2d()
    test_roi_3d()
    
def test_roi_3d():
    feature3d = np.zeros([2,5,30,30,1])
    
    roi_indexes = [[0,2,10,2],
                   [0,10,15,4],
                   [1,15,8,1],
                   [1,17,19,3]]
    
    for index in roi_indexes:    
        feature3d[index[0],index[3],index[1],index[2]] = 1
    
    feature3d = tf.cast(feature3d,dtype=tf.float32)
    roi_indexes = tf.cast(roi_indexes,dtype=tf.int32)
    
    crop_op = CropROI3D()
    crops = crop_op([feature3d, roi_indexes])
    
    #for crop in crops:
    #    plt.imshow(tf.reshape(crop,[9,9]))
    #    plt.show()
    
    goal = tf.cast(np.ones([4,1,9,9,1]),dtype=tf.float32) * 0.5
    with tf.GradientTape() as tape:
        tape.watch(feature3d)
        roi_crop = crop_op([feature3d, roi_indexes])
        loss = tf.keras.losses.mean_squared_error(roi_crop, goal)
    g = tape.gradient(loss, feature3d)

    for grads in g:
        for xy_grads in grads:
            plt.imshow(tf.reshape(xy_grads,[30,30]))
            plt.show()
    
    
def test_roi_2d():
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
    crop_ROI = CropROI2D()
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

if __name__ == "__main__":
    main()
    