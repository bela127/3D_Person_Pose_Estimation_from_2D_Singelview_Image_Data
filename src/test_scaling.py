import tensorflow as tf
keras = tf.keras


def main():
    eager = True                    ### please change to FALSE in eager mode all 3 tests are fine
    test_nr = 1 # 1 or 2 or 3       ### please test 1 and 2 and 3 -> diffrent errors
                                    ### error 3 ist clear, TensorShape is not tf.function compatible
                                    ### error 1,2 has somthing todo with the image.resize implementation
                                    ### runtime tensor is not evaluated and so the value is None
    
    tf.config.experimental_run_functions_eagerly(eager)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    inputs = tf.constant(100.,shape=[1,100,100,20])
    inputs_small = tf.constant(100.,shape=[1,80,80,20])
    
    if eager or test_nr == 1:
        print("Scaled1")
        scaled_shared = Scaled1()
        test_scaled_shared = test(scaled_shared, optimizer, training = True)
        out = test_scaled_shared([inputs,inputs_small])
        print("Scaled1")
    
    if eager or test_nr == 2:
        print("Scaled2")
        scaled_shared = Scaled2()
        test_scaled_shared = test(scaled_shared, optimizer, training = True)
        out = test_scaled_shared([inputs,inputs_small])
        print("Scaled2")
        
    if eager or test_nr == 3:
        print("Scaled3")
        scaled_shared = Scaled3()
        test_scaled_shared = test(scaled_shared, optimizer, training = True)
        out = test_scaled_shared([inputs,inputs_small])
        print("Scaled3")

def test(op, optimizer, **kwargs):
    def run(inputs):
        with tf.GradientTape() as tape:
            tape.watch(op.trainable_variables)
            outputs = op(inputs, **kwargs)
        g = tape.gradient(outputs, op.trainable_variables)
        optimizer.apply_gradients(zip(g, op.trainable_variables))
        return outputs, g
    return run
  
class Scale(keras.layers.Layer):
    def __init__(self, destination_channel = None, name = "Scale", **kwargs):
        super().__init__(name = name, **kwargs)
        self.destination_channel = destination_channel
        
    def build(self, input_shape):
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


class Scaled1(keras.layers.Layer):
    def __init__(self, name = "Scaled1", **kwargs):
        super().__init__(name = name, **kwargs)

        
    def build(self, input_shape):
        res_shape, shc_shape = input_shape
        self.scale_up = Scale(destination_channel = res_shape[-1])
        self.scale_down = Scale()
        super().build(input_shape)
        
    def call(self, inputs):
        inputs_res, inputs_shc = inputs
        shape1 = tf.shape(inputs_shc)[1:3]
        shape2 = tf.shape(inputs_shc)[1:3]
        
        scaled_res = self.scale_down(inputs_res, shape1)
        scaled_dense = self.scale_up(scaled_res, shape2)
        return scaled_dense      
    
class Scaled2(keras.layers.Layer):
    def __init__(self, name = "Scaled2", **kwargs):
        super().__init__(name = name, **kwargs)

        
    def build(self, input_shape):
        res_shape, shc_shape = input_shape
        self.scale_up = Scale(destination_channel = res_shape[-1])
        self.scale_down = Scale()
        super().build(input_shape)
        
    def call(self, inputs):
        inputs_res, inputs_shc = inputs
        
        shape1 = tf.cast(tf.shape(inputs_shc)[1:3], dtype = tf.int32)
        shape2 = tf.cast(tf.shape(inputs_shc)[1:3], dtype = tf.int32)
        
        scaled_res = self.scale_down(inputs_res, shape1)
        scaled_dense = self.scale_up(scaled_res, shape2)
        return scaled_dense
        
class Scaled3(keras.layers.Layer):
    def __init__(self, name = "Scaled2", **kwargs):
        super().__init__(name = name, **kwargs)

        
    def build(self, input_shape):
        res_shape, shc_shape = input_shape
        self.scale_up = Scale(destination_channel = res_shape[-1])
        self.scale_down = Scale()
        super().build(input_shape)
        
    def call(self, inputs):
        inputs_res, inputs_shc = inputs
        
        shape1 = inputs_shc.shape[1:3]
        shape2 = inputs_shc.shape[1:3]
        
        scaled_res = self.scale_down(inputs_res, shape1)
        scaled_dense = self.scale_up(scaled_res, shape2)
        return scaled_dense

  
if __name__ == '__main__':
    main()
    