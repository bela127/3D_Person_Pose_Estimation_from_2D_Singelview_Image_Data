import time

import numpy as np
import tensorflow as tf
keras = tf.keras

import ShAReD_Net.model.activation.base as activation_base
import ShAReD_Net.model.layer.aggregation as aggregation

class PersonDetector(keras.layers.Layer):
    
    def __init__(self,target_gpu = None, name = "PersonDetector", **kwargs):
        self.target_gpu = target_gpu

        super().__init__(name = name, **kwargs)
    
    @tf.Module.with_name_scope
    def build(self, input_shape):
        print(self.name,input_shape)
        self.expand = aggregation.Expand3D()
        self.conf2 = tf.keras.layers.Convolution3D(filters = input_shape[-1]/4,
                                      kernel_size = [3,3,3],
                                      strides = [1,1,1],
                                      padding='same',
                                      activation=tf.nn.leaky_relu,
                                      kernel_initializer=tf.initializers.he_uniform(),
                                      )
        self.conf3 = tf.keras.layers.Convolution3D(filters = input_shape[-1]/4,
                                      kernel_size = [1,1,1],
                                      strides = [1,1,1],
                                      padding='same',
                                      activation=tf.nn.leaky_relu,
                                      kernel_initializer=tf.initializers.he_uniform(),
                                      )
        self.conf_d_1 = tf.keras.layers.Convolution3D(filters = 8,
                                      kernel_size = [7,1,1],
                                      strides = [1,1,1],
                                      padding='same',
                                      activation=tf.nn.leaky_relu,
                                      kernel_initializer=tf.initializers.he_uniform(),
                                      )
        self.conf_w_1 = tf.keras.layers.Convolution3D(filters = 8,
                                      kernel_size = [1,7,1],
                                      strides = [1,1,1],
                                      padding='same',
                                      activation=tf.nn.leaky_relu,
                                      kernel_initializer=tf.initializers.he_uniform(),
                                      )
        self.conf_h_1 = tf.keras.layers.Convolution3D(filters = 8,
                                      kernel_size = [1,1,7],
                                      strides = [1,1,1],
                                      padding='same',
                                      activation=tf.nn.leaky_relu,
                                      kernel_initializer=tf.initializers.he_uniform(),
                                      )
        self.conf_d_2 = tf.keras.layers.Convolution3D(filters = 8,
                                      kernel_size = [7,1,1],
                                      strides = [1,1,1],
                                      padding='same',
                                      activation=tf.nn.leaky_relu,
                                      kernel_initializer=tf.initializers.he_uniform(),
                                      )
        self.conf_w_2 = tf.keras.layers.Convolution3D(filters = 8,
                                      kernel_size = [1,7,1],
                                      strides = [1,1,1],
                                      padding='same',
                                      activation=tf.nn.leaky_relu,
                                      kernel_initializer=tf.initializers.he_uniform(),
                                      )
        self.conf_h_2 = tf.keras.layers.Convolution3D(filters = 8,
                                      kernel_size = [1,1,7],
                                      strides = [1,1,1],
                                      padding='same',
                                      activation=tf.nn.leaky_relu,
                                      kernel_initializer=tf.initializers.he_uniform(),
                                      )
        self.conf4 = tf.keras.layers.Convolution3D(filters = 1,
                                      kernel_size = [1,1,1],
                                      strides = [1,1,1],
                                      padding='same',
                                      activation=tf.nn.leaky_relu,
                                      kernel_initializer=tf.initializers.he_uniform(),
                                      )

        super().build(input_shape)
    
    @tf.Module.with_name_scope
    def call(self, inputs, training=None):
        if self.target_gpu:
            print("detector using", self.target_gpu)
            with tf.device(self.target_gpu):
                return self._compute(inputs, training)
        else:
            return self._compute(inputs, training)
    
    @tf.function(experimental_autograph_options=tf.autograph.experimental.Feature.ALL, experimental_relax_shapes=True)
    def _compute(self, inputs, training):
        feature = inputs
        expanded = self.expand(feature)
        conf2 = self.conf2(expanded)
        conf3 = self.conf3(conf2)
        conf_d_1 = self.conf_d_1(conf3)
        conf_w_1 = self.conf_w_1(conf3)
        conf_h_1 = self.conf_h_1(conf3)
        
        conf_hd = self.conf_d_2(conf_h_1)
        conf_dw = self.conf_w_2(conf_d_1)
        conf_wh = self.conf_h_2(conf_w_1)
        
        conf_wd = self.conf_d_2(conf_w_1)
        conf_hw = self.conf_w_2(conf_h_1)
        conf_dh = self.conf_h_2(conf_d_1)
        
        conf_hdw = self.conf_w_2(conf_hd)
        conf_dwh = self.conf_h_2(conf_dw)
        conf_whd = self.conf_d_2(conf_wh)
        
        conf_wdh = self.conf_h_2(conf_wd)
        conf_hwd = self.conf_d_2(conf_hw)
        conf_dhw = self.conf_w_2(conf_dh)
        
        person_feature = tf.concat([conf_hdw,conf_dwh,conf_whd,conf_wdh,conf_hwd,conf_dhw], axis=-1)
        person_det = self.conf4(person_feature)
        person_det = person_det[...,0]
        person_det = tf.transpose(person_det,[0,2,3,1])
        person_det = activation_base.discret_sigmoid(person_det, training=training)
        return person_det
        
    def get_config(self):
        config = super().get_config()
        return config



def test(op, optimizer, **kwargs):
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

    print("PersonDetector")
    inputs = tf.constant(100.,shape=[4,5,100,100,10])
    msf = PersonDetector()    
    test_msf= test(msf, optimizer, training = True)
    out,g = test_msf(inputs)
    print(msf.count_params())
    print(out.shape)
    print("PersonDetector")
    
    time_start = time.time()
    for i in range(10):
        out = test_msf(inputs)
    time_end = time.time()
    print(time_end - time_start)
    
    

if __name__ == '__main__':
    main()