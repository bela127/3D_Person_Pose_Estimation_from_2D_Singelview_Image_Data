import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

def main():
    locator = Locator2D()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
    loss_func = locator.loss
    loc_loss = DiscretLocationLoss(locator.dtcl)
    var_loss = VarianceLocatonLoss(locator.dtcl)
    var_reg = VarianceLocatonRegularization(locator.dtcl)

    variance_reg = tf.keras.metrics.Mean(name='variance_reg')
    variance_loss = tf.keras.metrics.Mean(name='variance_loss')
    location_loss = tf.keras.metrics.Mean(name='location_loss')

    epochs_history = []
    variance_loss_history = []
    variance_reg_history = []
    location_loss_history = []
    
    #@tf.function
    def train_step(observations, goals):
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = locator(observations, training=True)
            loss = loss_func(predictions, goals)
            #plt.imshow(predictions)
            #plt.show()

        gradients = tape.gradient(loss, locator.trainable_variables)
        optimizer.apply_gradients(zip(gradients, locator.trainable_variables))

        variance_reg(var_reg(predictions))
        variance_loss(var_loss(predictions, goals))
        location_loss(loc_loss(predictions, goals))
        return loss
        
    
    def train_loop(epochs):
        observations = np.array([15,5])
        goals = np.array([1450,1450], dtype=np.float32)

        for epoch in range(epochs):
            # Reset the metrics at the start of the next epoch
            variance_loss.reset_states()
            location_loss.reset_states()
            variance_reg.reset_states()

            loss = train_step(observations, goals)
            if loss < 0.0000001: break

            template = 'Epoch {}, variance_loss: {}, location_loss: {}, variance_reg: {}'
            print(template.format(  epoch+1,
                                    variance_loss.result(),
                                    location_loss.result(),
                                    variance_reg.result(),
                                    ))
            epochs_history.append(epoch+1)
            variance_loss_history.append(variance_loss.result())
            location_loss_history.append(location_loss.result())
            variance_reg_history.append(variance_reg.result())

    train_loop(500)
    representation = locator(2)
    plt.imshow(representation)
    plt.show()
    plt.plot(epochs_history, variance_loss_history)
    plt.axis([0, 500, 0, 0.5])
    plt.show()
    plt.plot(epochs_history, variance_reg_history)
    plt.axis([0, 500, 0, 0.02])
    plt.show()
    plt.plot(epochs_history, location_loss_history)
    plt.axis([0, 500, 0, 0.02])
    plt.show()




class Locator2D(tf.keras.Model):

    def __init__(self):
        super().__init__()
        bins = [10,10]
        mins = [0,0]
        maxs = [3000,3000]
        self.dl = DiscretLocation2D(bins)
        self.dtcl = DiscretToContinuousLocation(mins,maxs,bins)
        self.loss = RegularizedDiscretLocationLoss(self.dtcl)
        self.loss = VarianceLocatonLoss(self.dtcl)
        self.loss = VarianceDiscretLocationLoss(self.dtcl)

    def call(self, inputs):
        return self.dl(inputs)



    

class Convolution(tf.keras.layers.Layer):
    def __init__(self, filter_size=[1,1], filter_count=1):
        super().__init__()
        self.filter_size = filter_size
        self.filter_count = filter_count

    def build(self, input_shape):
        self.filter = self.add_weight(trainable=True, shape = [*self.filter_size,input_shape[-1],self.filter_count], initializer = tf.initializers.he_uniform())
        self.bias = self.add_weight(trainable=True, shape = [self.filter_count], initializer = tf.initializers.he_uniform())


    def call(self, inputs):
        convolved = tf.nn.convolution(inputs, self.filter,padding="SAME")
        #convolved = tf.nn.bias_add(convolved, self.bias)
        activated = convolved
        #activated = custom_relu(convolved)
        return activated

@tf.custom_gradient
def custom_relu(inputs):

    activated = tf.nn.relu(inputs)

    def grad(dy):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(inputs)
            act = tf.nn.leaky_relu(inputs)

        return dy*tape.gradient(act, inputs)
    return activated, grad


class DiscretLocation2D(tf.keras.layers.Layer):
    def __init__(self, bins=[10,10]):
        super().__init__()
        self.bins = np.array(bins)
        self.representation = self.add_weight(trainable=True, shape = self.bins, initializer = tf.initializers.he_uniform())
        self.conv = Convolution()

    def call(self, inputs):
        shape = self.representation.shape
        blow = tf.reshape(self.representation,[1,*self.bins,1])
        maxed = blow
        #maxed = self.conv(blow)
        flat = tf.reshape(maxed,[-1])
        flat = tf.nn.softmax(flat)
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
        error = tf.reduce_mean((tf.math.maximum(variance,self.variance_offset) - self.variance_offset)/self.varianz_max)
        return error

class VarianceLocatonLoss(tf.keras.layers.Layer):
    
    def __init__(self, loc_converter):
        super().__init__()
        self.loc_converter = loc_converter
        self.variance_offset = (self.loc_converter.loc_delta/2)**2        

    def call(self, discret_loc, loc):
        loc = tf.reshape(loc,[loc.shape[0],*[1 for i in range(len(discret_loc.shape))]])
        discret_loc = tf.reshape(discret_loc,[1,*discret_loc.shape])
        variance = tf.reduce_sum(discret_loc * (self.loc_converter.loc_map - loc)**2, axis=range(1,len(discret_loc.shape)))
        shifted_var = tf.reduce_mean((tf.math.maximum(variance,self.variance_offset) - self.variance_offset))
        return shifted_var # tf.reduce_mean(variance)

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
    
class VarianceDiscretLocationLoss(tf.keras.layers.Layer):
    def __init__(self, loc_converter):
        super().__init__()
        self.loc_converter = loc_converter
        self.dll = DiscretLocationLoss(loc_converter)
        self.vll = VarianceLocatonLoss(loc_converter)

    def call(self, discret_loc, loc):
        return self.dll(discret_loc, loc) + self.vll(discret_loc,loc)

if __name__ == '__main__':
    main()
    