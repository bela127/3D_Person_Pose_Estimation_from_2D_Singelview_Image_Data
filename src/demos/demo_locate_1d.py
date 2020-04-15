import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

def main():
    locator = Locator1D()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
    loss_func = locator.loss

    variance_loss = tf.keras.metrics.Mean(name='variance_loss')
    location_loss = tf.keras.metrics.Mean(name='location_loss')


    def train_step(observations, goals):
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = locator(observations, training=True)
            loss = loss_func(predictions, goals)
        gradients = tape.gradient(loss, locator.trainable_variables)
        optimizer.apply_gradients(zip(gradients, locator.trainable_variables))

        variance_loss(loss_func.dll(predictions, goals))
        location_loss(loss_func.vlr(predictions))

    def train_loop(epochs):
        observations = [15]
        goals = [15]

        for epoch in range(epochs):
            # Reset the metrics at the start of the next epoch
            variance_loss.reset_states()
            location_loss.reset_states()

            train_step(observations, goals)


            template = 'Epoch {}, variance_loss: {}, location_loss: {}'
            print(template.format(  epoch+1,
                                    variance_loss.result(),
                                    location_loss.result()))
    train_loop(100)
    representation = locator(2)
    print(representation)
    plt.bar(locator.dtcl.loc_map,representation)
    plt.show()





class Locator1D(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.dl = DiscretLocation(10)
        self.dtcl = DiscretToContinuousLocation(0,20,10)
        self.loss = RegularizedDiscretLocationLoss(self.dtcl)

    def call(self, inputs):
        return self.dl(inputs)



    

class DiscretLocation(tf.keras.layers.Layer):
    def __init__(self, bins):
        super().__init__()
        self.bins = bins
        self.representation = self.add_weight(trainable=True, shape = [bins], initializer = tf.initializers.he_uniform())
    
    def call(self,inputs):
        return tf.nn.softmax(self.representation)

class DiscretToContinuousLocation(tf.keras.layers.Layer):

    def __init__(self, min_loc=0, max_loc=10, bins=10):
        super().__init__()
        self.loc_delta = (max_loc - min_loc) / bins
        self.loc_map = tf.constant(np.arange(min_loc,max_loc,self.loc_delta), dtype = tf.float32)#, name='Const')

    def call(self, discret_loc):
        return tf.reduce_sum(discret_loc * self.loc_map)


class VarianceLocatonRegularization(tf.keras.layers.Layer):

    def __init__(self, loc_converter):
        super().__init__()
        self.loc_converter = loc_converter

    def call(self, discret_loc):
        loc = self.loc_converter(discret_loc)
        variance = tf.reduce_sum(discret_loc * (self.loc_converter.loc_map - loc)**2)
        variance_offset = (self.loc_converter.loc_delta/2)**2
        return tf.math.maximum(variance,variance_offset) - variance_offset
    
class DiscretLocationLoss(tf.keras.layers.Layer):
    def __init__(self, loc_converter):
        super().__init__()
        self.loc_converter = loc_converter

    def call(self, discret_loc, loc):
        return tf.reduce_sum(tf.math.squared_difference(self.loc_converter(discret_loc),loc))
    

class RegularizedDiscretLocationLoss(tf.keras.layers.Layer):
    def __init__(self, loc_converter):
        super().__init__()
        self.loc_converter = loc_converter
        self.dll = DiscretLocationLoss(loc_converter)
        self.vlr = VarianceLocatonRegularization(loc_converter)

    def call(self, discret_loc, loc):
        return self.dll(discret_loc, loc) + self.vlr(discret_loc)

if __name__ == '__main__':
    main()
    