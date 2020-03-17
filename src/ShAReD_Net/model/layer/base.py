import time

import tensorflow as tf
keras = tf.keras

class BnDoConfReluConfRelu(keras.layers.Layer):
    def __init__(self, filter_count, rate = 0.15, filter_size = [3,3], name = "BnDoConfReluConfRelu", **kwargs):
        super().__init__(name = name, **kwargs)
        self.filter_count = filter_count
        self.rate = rate
        self.filter_size = filter_size
        
        
    def build(self, input_shape):
        print(self.name,input_shape)
        self.bn = keras.layers.BatchNormalization(input_shape = [None, None, input_shape[-1]])
        self.do = keras.layers.GaussianDropout(self.rate)
        self.conv1 = keras.layers.Convolution2D(self.filter_count, self.filter_size, padding='SAME', activation=tf.nn.leaky_relu, kernel_initializer=tf.initializers.he_normal(), bias_initializer=tf.initializers.he_uniform())
        self.conv2 = keras.layers.Convolution2D(self.filter_count, self.filter_size, padding='SAME', activation=tf.nn.leaky_relu, kernel_initializer=tf.initializers.he_normal(), bias_initializer=tf.initializers.he_uniform())
        super().build(input_shape)
        
    @tf.function
    def call(self, inputs, training=None):
        bn = self.bn(inputs, training=training)
        do = self.do(bn, training=training)
        conv1 = self.conv1(do)
        conv2 = self.conv2(conv1)
        return conv2
        
    def get_config(self):
        config = super().get_config()
        config.update({'filter_count': self.filter_count,
                       'rate': self.rate,
                       'filter_size': self.filter_size,
                       })
        return config

class DenseBlock(keras.layers.Layer):
    def __init__(self, filter_count, rate = 0.15, filter_size = [3,3], name = "DenseBlock", **kwargs):
        super().__init__(name = name, **kwargs)
        self.filter_count = filter_count
        self.rate = rate
        self.filter_size = filter_size

    def build(self, input_shape):
        print(self.name,input_shape)
        self.block = BnDoConfReluConfRelu(filter_count = self.filter_count, rate = self.rate, filter_size = self.filter_size)
        super().build(input_shape)

    @tf.function
    def call(self, inputs, training=None):
        block = self.block(inputs, training=training)
        concat = keras.layers.concatenate([inputs, block])
        return concat
        
    def get_config(self):
        config = super().get_config()
        return config
        
class DenseModule(keras.layers.Layer):
    def __init__(self, blocks_count, filter_count, rate = 0.15, filter_size = [3,3], name = "DenseModule", **kwargs):
        super().__init__(name = name, **kwargs)
        self.blocks_count = blocks_count
        self.filter_count = filter_count
        self.rate = rate
        self.filter_size = filter_size
        
    def build(self, input_shape):
        print(self.name,input_shape)
        self.blocks = [DenseBlock(filter_count = self.filter_count, rate = self.rate, filter_size = self.filter_size) for i in range(self.blocks_count)]
        super().build(input_shape)

    @tf.function
    def call(self, inputs, training=None):
        prev_block = inputs
        for next_block in self.blocks:
            prev_block = next_block(prev_block, training=training)
        return prev_block
        
    def get_config(self):
        config = super().get_config()
        config.update({'blocks_count': self.blocks_count})
        return config

class ShReD(keras.layers.Layer):
    def __init__(self, dense_filter_count, dense_blocks_count = 2, do_rate = 0.15, dense_filter_size = [3,3], name = "ShReD", **kwargs):
        super().__init__(name = name, **kwargs)
        self.dense_blocks_count = dense_blocks_count
        self.dense_filter_count = dense_filter_count
        self.do_rate = do_rate
        self.dense_filter_size = dense_filter_size
        
        
    def build(self, input_shape):
        print(self.name,input_shape)
        res_shape, shc_shape = input_shape
        self.dense_m = DenseModule(blocks_count = self.dense_blocks_count, rate = self.do_rate, filter_size = self.dense_filter_size, filter_count = self.dense_filter_count)
        self.write_res = keras.layers.Convolution2D(res_shape[-1], kernel_size=1, padding='SAME', activation=tf.nn.leaky_relu, kernel_initializer=tf.initializers.he_normal(), bias_initializer=tf.initializers.he_uniform())
        super().build(input_shape)

    @tf.function
    def call(self, inputs, training=None):
        inputs_res, inputs_shc = inputs
        concat = keras.layers.concatenate(inputs)
        dense_m = self.dense_m(concat, training=training)
        write_res = self.write_res(dense_m)
        add_res = keras.layers.add([inputs_res, write_res])
        return add_res, dense_m
        
    def get_config(self):
        config = super().get_config()
        return config
    
class Attention(keras.layers.Layer):
    def __init__(self, name = "Attention", **kwargs):
        super().__init__(name = name, **kwargs)     
        
    def build(self, input_shape):
        print(self.name,input_shape)
        ins_shape, att_shape = input_shape
        self.attention = keras.layers.Convolution2D(ins_shape[-1], kernel_size=1, padding='SAME', activation=tf.nn.sigmoid, kernel_initializer=tf.initializers.glorot_normal(), bias_initializer=tf.initializers.glorot_uniform())
        super().build(input_shape)
        
    @tf.function
    def call(self, inputs):
        ins, att = inputs
        attention = self.attention(att)
        salient = keras.layers.multiply([ins, attention])
        concat = keras.layers.concatenate([ins, salient])
        return concat
        
    def get_config(self):
        config = super().get_config()
        return config
    
class ResAttention(keras.layers.Layer):
    def __init__(self, name = "ResAttention", **kwargs):
        super().__init__(name = name, **kwargs)     
        
    def build(self, input_shape):
        print(self.name,input_shape)
        res_shape, shc_shape = input_shape
        self.attention_res = Attention()
        self.read_shc = keras.layers.Convolution2D(res_shape[-1], kernel_size=1, padding='SAME', activation=tf.nn.leaky_relu, kernel_initializer=tf.initializers.he_normal(), bias_initializer=tf.initializers.he_uniform())
        super().build(input_shape)
        
    @tf.function
    def call(self, inputs):
        inputs_res, inputs_shc = inputs
        attention_res = self.attention_res(inputs)
        read_shc = self.read_shc(inputs_shc)
        concat = keras.layers.concatenate([attention_res, read_shc])
        return concat
        
    def get_config(self):
        config = super().get_config()
        return config

class ShAReD(keras.layers.Layer):
    def __init__(self, dense_filter_count, dense_blocks_count = 2, do_rate = 0.15, dense_filter_size = [3,3], name = "ShAReD", **kwargs):
        super().__init__(name = name, **kwargs)
        self.dense_blocks_count = dense_blocks_count
        self.dense_filter_count = dense_filter_count
        self.do_rate = do_rate
        self.dense_filter_size = dense_filter_size
        
        
    def build(self, input_shape):
        print(self.name,input_shape)
        res_shape, shc_shape = input_shape
        self.shred = ShReD(dense_filter_count= self.dense_filter_count, dense_blocks_count= self.dense_blocks_count, do_rate=self.do_rate, dense_filter_size= self.dense_filter_size)
        self.attention = ResAttention()
        super().build(input_shape)

    @tf.function
    def call(self, inputs, training=None):
        inputs_res, inputs_shc = inputs
        attention = self.attention(inputs)
        output = self.shred([inputs_res, attention], training=training)
        return output
        
    def get_config(self):
        config = super().get_config()
        return config
    
class SelfShAReD(keras.layers.Layer):
    def __init__(self, dense_filter_count, dense_blocks_count = 2, do_rate = 0.15, dense_filter_size = [3,3], name = "SelfShAReD", **kwargs):
        super().__init__(name = name, **kwargs)
        self.dense_blocks_count = dense_blocks_count
        self.dense_filter_count = dense_filter_count
        self.do_rate = do_rate
        self.dense_filter_size = dense_filter_size
        
        
    def build(self, input_shape):
        print(self.name,input_shape)
        res_shape, shc_shape = input_shape
        self.shared1 = ShAReD(dense_filter_count= self.dense_filter_count, dense_blocks_count= self.dense_blocks_count, do_rate=self.do_rate, dense_filter_size= self.dense_filter_size)
        self.attention = Attention()
        self.shared2 = ShAReD(dense_filter_count= self.dense_filter_count, dense_blocks_count= self.dense_blocks_count, do_rate=self.do_rate, dense_filter_size= self.dense_filter_size)
        super().build(input_shape)
        
    @tf.function
    def call(self, inputs, training=None):
        inputs_res, inputs_shc = inputs
        outputs_res, outputs_shc = self.shared1([inputs_res, inputs_shc], training=training)
        attention = self.attention([inputs_shc, outputs_shc])
        output = self.shared2([outputs_res, attention], training=training)
        return output
        
    def get_config(self):
        config = super().get_config()
        return config
    
class Scale(keras.layers.Layer):
    def __init__(self, destination_channel = None, name = "Scale", **kwargs):
        super().__init__(name = name, **kwargs)
        self.destination_channel = destination_channel
        
    def build(self, input_shape):
        print(self.name,input_shape)
        if self.destination_channel is None:
            self.destination_channel = input_shape[-1]
        self.compress_input = keras.layers.Convolution2D(int(input_shape[-1]/2), kernel_size=1, padding='SAME', activation=tf.nn.leaky_relu, kernel_initializer=tf.initializers.he_normal(), bias_initializer=tf.initializers.he_uniform())
        self.conv = keras.layers.Convolution2D(input_shape[-1], kernel_size=3, padding='SAME', activation=tf.nn.leaky_relu, kernel_initializer=tf.initializers.he_normal(), bias_initializer=tf.initializers.he_uniform())
        self.pool = keras.layers.MaxPool2D(pool_size=3,strides=1,padding="SAME")
        self.compress_output = keras.layers.Convolution2D(self.destination_channel, kernel_size=1, padding='SAME', activation=tf.nn.leaky_relu, kernel_initializer=tf.initializers.he_normal(), bias_initializer=tf.initializers.he_uniform())
        super().build(input_shape)

    @tf.function
    def call(self, inputs, destination_size):
        
        compressed_input = self.compress_input(inputs)
        conv = self.conv(compressed_input)
        pool = self.pool(inputs)
        
        scaled_conv = tf.image.resize(conv, destination_size, preserve_aspect_ratio=True, antialias=True)
        scaled_pool = tf.image.resize(pool, destination_size, preserve_aspect_ratio=True, antialias=True)
        
        concat = keras.layers.concatenate([scaled_pool, scaled_conv])
        compressed_output = self.compress_output(concat)
        return compressed_output
    
    def get_config(self):
        config = super().get_config()
        config.update({'destination_channel': self.destination_channel,
                       })
        return config
    
class ScaledShAReD(keras.layers.Layer):
    def __init__(self, dense_filter_count, dense_blocks_count = 2, do_rate = 0.15, dense_filter_size = [3,3], name = "ScaledShAReD", **kwargs):
        super().__init__(name = name, **kwargs)
        self.dense_blocks_count = dense_blocks_count
        self.dense_filter_count = dense_filter_count
        self.do_rate = do_rate
        self.dense_filter_size = dense_filter_size
        
    def build(self, input_shape):
        print(self.name,input_shape)
        res_shape, shc_shape = input_shape
        self.dense_m = DenseModule(blocks_count = self.dense_blocks_count, rate = self.do_rate, filter_size = self.dense_filter_size, filter_count = self.dense_filter_count)
        self.attention = ResAttention()
        self.scale_up = Scale(destination_channel = res_shape[-1])
        self.scale_down = Scale()
        super().build(input_shape)
        
    @tf.function
    def call(self, inputs, training=None):
        inputs_res, inputs_shc = inputs
        inputs_shc_shape = tf.shape(inputs_shc)
        inputs_res_shape = tf.shape(inputs_res)
        scaled_res = self.scale_down(inputs_res, inputs_shc_shape[1:3])
        attention = self.attention([scaled_res, inputs_shc])
        concat = keras.layers.concatenate([scaled_res, attention])
        dense_m = self.dense_m(concat, training=training)
        scaled_dense = self.scale_up(dense_m, inputs_res_shape[1:3])
        add_res = keras.layers.add([inputs_res, scaled_dense])
        return add_res, dense_m
        
    def get_config(self):
        config = super().get_config()
        return config

class Merge(keras.layers.Layer):
    def __init__(self, name = "Merge", **kwargs):
        super().__init__(name = name, **kwargs)
        
    def build(self, input_shape):
        print(self.name,input_shape)
        super().build(input_shape)

    @tf.function
    def call(self, inputs):
        small, normal, big = inputs
        normal_shape = tf.shape(normal)
        if small is not None and big is not None:
            resized_small = tf.image.resize_with_crop_or_pad(small, normal_shape[1], normal_shape[2])
            resized_big = tf.image.resize_with_crop_or_pad(big, normal_shape[1], normal_shape[2])
            concat = tf.concat([resized_small, normal, resized_big],axis=-1)
        elif small is not None:
            resized_small = tf.image.resize_with_crop_or_pad(small, normal_shape[1], normal_shape[2])
            concat = tf.concat([resized_small, normal],axis=-1)
        elif big is not None:
            resized_big = tf.image.resize_with_crop_or_pad(big, normal_shape[1], normal_shape[2])
            concat = tf.concat([normal, resized_big],axis=-1)
        else:
            concat = normal
        
        return concat
        
    def get_config(self):
        config = super().get_config()
        config.update({'merge_smaller': self.merge_smaller,
                       'merge_biger': self.merge_biger,
                       })
        return config
    
class Mix(keras.layers.Layer):
    def __init__(self, name = "Mix", **kwargs):
        super().__init__(name = name, **kwargs)     
        
    def build(self, input_shape):
        print(self.name,input_shape)
        input_count = len(input_shape)
        self.merge = list([Merge() for i in range(input_count)])
        super().build(input_shape)

    @tf.function
    def call(self, inputs):
        inputs_u = [None] + inputs[:-1]
        inputs_m = inputs
        inputs_d = inputs[1:] + [None]
        mixed = list(zip(inputs_u, inputs_m, inputs_d))
        combined = list(zip(self.merge, mixed))
        merge = list([merge(ins) for merge, ins in combined])
        return merge
        
    def get_config(self):
        config = super().get_config()
        return config



def test(op, optimizer, **kwargs):
    @tf.function
    def run(inputs):
        with tf.GradientTape() as tape:
            tape.watch(op.trainable_variables)
            outputs = op(inputs, **kwargs)
        g = tape.gradient(outputs, op.trainable_variables)
        optimizer.apply_gradients(zip(g, op.trainable_variables))
        return outputs, g
    return run
    

def main():
    #tf.config.experimental_run_functions_eagerly(True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    print("SelfShAReD")
    inputs = tf.constant(100.,shape=[1,100,100,20])
    self_shared = SelfShAReD(dense_filter_count=64)
    test_self_shared = test(self_shared, optimizer, training = True)
    out = test_self_shared([inputs,inputs])
    print(out)
    print("SelfShAReD")
    
    print("ScaledShAReD")
    inputs_small = tf.constant(100.,shape=[1,80,80,20])
    scaled_shared = ScaledShAReD(dense_filter_count=64)
    test_scaled_shared = test(scaled_shared, optimizer, training = True)
    out = test_scaled_shared([inputs,inputs_small])
    print(out)
    print("ScaledShAReD")
    
    print("Mix")
    mix = Mix()
    test_mix = test(mix, optimizer, training = True)
    out = test_mix([inputs_small,inputs,inputs])
    [print(output.shape) for output in out[0]]
    print("Mix")

    time_start = time.time()
    for i in range(20):
        out = test_scaled_shared([inputs,inputs_small])
    time_end = time.time()
    print(out)
    print(time_end - time_start)
    
    
    time_start = time.time()
    for i in range(20):
        out = test_self_shared([inputs,inputs])
    time_end = time.time()
    print(out)
    print(time_end - time_start)
    

if __name__ == '__main__':
    main()
    