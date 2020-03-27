import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

def main():
    locator = Locator2D()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
    loss_func = locator.loss

    variance_loss = tf.keras.metrics.Mean(name='variance_loss')
    location_loss = tf.keras.metrics.Mean(name='location_loss')

    @tf.function
    def train_step(observations, goals):
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = locator(observations, training=True)
            loss = loss_func(predictions, goals)
        gradients = tape.gradient(loss, locator.trainable_variables)
        optimizer.apply_gradients(zip(gradients, locator.trainable_variables))

        #variance_loss(loss_func.vlr(predictions))
        location_loss(loss)
        return loss
        
    
    def train_loop(epochs):
        observations = np.array([15,5])
        goals = np.array([6,5], dtype=tf.float32)

        for epoch in range(epochs):
            # Reset the metrics at the start of the next epoch
            variance_loss.reset_states()
            location_loss.reset_states()

            loss = train_step(observations, goals)
            if loss < 0.0001: break

            template = 'Epoch {}, variance_loss: {}, location_loss: {}'
            print(template.format(  epoch+1,
                                    variance_loss.result(),
                                    location_loss.result()))

    train_loop(1500)
    representation = locator(2)
    print(representation)
    plt.imshow(representation)
    plt.show()




class Locator2D(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.dl = DiscretLocation2D()
        self.dtcl = DiscretToContinuousLocation([0,0],[20,20])
        self.loss = RegularizedDiscretLocationLoss(self.dtcl)
        #self.loss = VarianceLocatonLoss(self.dtcl)

    def call(self, inputs):
        return self.dl(inputs)



    

class DiscretLocation2D(tf.keras.layers.Layer):
    def __init__(self, bins=[10,10]):
        super().__init__()
        self.bins = np.array(bins)
        self.parameter = self.add_weight(trainable=True, shape = self.bins, initializer = tf.initializers.he_uniform())
        #self.filter = self.add_weight(trainable=True, shape = [1,1,1,1], initializer = tf.initializers.he_uniform())

    def call(self, inputs):
        shape = self.parameter.shape
        blow = tf.reshape(self.parameter,[1,*self.bins,1])
        #maxed = tf.nn.convolution(blow,self.filter,padding="SAME")
        flat = tf.reshape(self.parameter,[-1])
        flat = flat**2
        sum_flat = tf.reduce_sum(flat)
        flat = flat / sum_flat
        #flat = tf.nn.softmax(flat)
        return tf.reshape(flat, shape)

class DiscretToContinuousLocation(tf.keras.layers.Layer):

    def __init__(self, min_loc=[0,0], max_loc=[10,10], bins=[10,10]):
        super().__init__()
        self.bins = tf.constant(bins, dtype = tf.float32)
        self.min_loc = tf.constant(min_loc, dtype = tf.float32)
        self.max_loc = tf.constant(max_loc, dtype = tf.float32)
        self.loc_delta = (self.max_loc - self.min_loc) / self.bins
        loc_map = np.meshgrid(*[np.arange(_min_loc, _max_loc, _delta) for _min_loc, _max_loc, _delta in  zip(self.min_loc, self.max_loc, self.loc_delta)])
        self.loc_map = tf.constant(loc_map, dtype = tf.float32)

    def call(self, discret_loc):
        return tf.reduce_sum(discret_loc * self.loc_map, axis=range(1,len(discret_loc.shape) + 1))


class VarianceLocatonRegularization(tf.keras.layers.Layer):

    def __init__(self, loc_converter):
        super().__init__()
        self.loc_converter = loc_converter
        self.variance_offset = (self.loc_converter.loc_delta/2)**2
        self.varianz_max = ((self.loc_converter.max_loc-self.loc_converter.min_loc)*0.5)**2.0 - self.variance_offset
        

    def call(self, discret_loc):
        loc = self.loc_converter(discret_loc)
        loc = tf.reshape(loc,[loc.shape[0],*[1 for i in range(len(discret_loc.shape))]])
        variance = tf.reduce_sum(discret_loc * (self.loc_converter.loc_map - loc)**2, axis=range(1,len(discret_loc.shape) + 1))
        return tf.reduce_mean((tf.math.maximum(variance,self.variance_offset) - self.variance_offset)/self.varianz_max)

class VarianceLocatonLoss(tf.keras.layers.Layer):
    
    def __init__(self, loc_converter):
        super().__init__()
        self.loc_converter = loc_converter
        self.variance_offset = (self.loc_converter.loc_delta/2)**2
        self.varianz_max = ((self.loc_converter.max_loc-self.loc_converter.min_loc)*0.5)**2.0 - self.variance_offset
        

    def call(self, discret_loc, loc):
        loc = tf.reshape(loc,[loc.shape[0],*[1 for i in range(discret_loc.ndim)]])
        variance = tf.reduce_sum(discret_loc * (self.loc_converter.loc_map - loc)**2, axis=range(1,len(discret_loc.shape) + 1))
        return tf.reduce_mean((tf.math.maximum(variance,self.variance_offset) - self.variance_offset))
  

class DiscretLocationLoss(tf.keras.layers.Layer):
    def __init__(self, loc_converter):
        super().__init__()
        self.loc_converter = loc_converter

    def call(self, discret_loc, loc):
        error = tf.reduce_mean(tf.math.squared_difference(self.loc_converter(discret_loc),loc))
        return error

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
    