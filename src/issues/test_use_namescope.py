import tensorflow as tf

class DummyModel(tf.keras.layers.Layer):
    
    def __init__(self, name = "DummyModel", **kwargs):  
        super().__init__(name = name, **kwargs)

    @tf.Module.with_name_scope
    def build(self, inputs_shape):
        super().build(inputs_shape)
    
    @tf.Module.with_name_scope
    def call(self, inputs):
        return inputs
    
dm = DummyModel()


print(dm([1]))