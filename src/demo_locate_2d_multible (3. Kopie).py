import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np


def main():
    bins = tf.cast([40,40], tf.int32)
    mins = tf.cast([0,0], tf.float32)
    maxs = tf.cast([10,10], tf.float32)

    offset = 100

    locator = Locator(min_loc = mins+offset, max_loc = maxs+offset, bins = bins)
    clm = ContinuousLocationMap(min_loc = mins+offset, max_loc = maxs+offset, bins = bins)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1)
    loss_func = locator.loss

    variance_loss = tf.keras.metrics.Mean(name='variance_loss')
    location_loss = tf.keras.metrics.Mean(name='location_loss')

    #@tf.function
    def train_step(observations, goals):
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = locator(observations, training=True)

            loss, loc_loss, var_loss = loss_func(predictions, goals)

        gradients = tape.gradient(loss, locator.trainable_variables)
        optimizer.apply_gradients(zip(gradients, locator.trainable_variables))

        variance_loss(var_loss)
        location_loss(loc_loss)
        return loss
    
    
    def train_loop(epochs):
        observations = np.array([15,5])
        goals_raw = np.array([[[5,5],[7.5,7.5],[2,7.5],[5,1],[8.25,2]]], dtype=np.float32)+offset


        for epoch in range(epochs):
            # Reset the metrics at the start of the next epoch
            variance_loss.reset_states()
            location_loss.reset_states()
            goals = clm(goals_raw)
            

            loss = train_step(observations, goals)
            if loss < 0.0001: break

            template = 'Epoch {}, variance_loss: {}, location_loss: {}'
            print(template.format(  epoch+1,
                                    variance_loss.result(),
                                    location_loss.result()))

    train_loop(500)
    representation = locator.dl()
    for element in representation:
        plt.imshow(element[:,:,0])
        plt.show()
        plt.imshow(element[:,:,1])
        plt.show()
    location = locator(1)
    for element in location:
        for layer_index in range(4):
            image = tf.reshape(element[:,:,layer_index], element.shape[0:2])
            indexe = np.nonzero(image)
            for x,y in zip(*indexe):
                print(image[x,y])
            plt.imshow(image)
            plt.show()

class LocationMap(tf.keras.layers.Layer):
    def __init__(self, min_loc = [0,0], max_loc = [10,10], bins = [10,10]):
        super().__init__()
        self.min_loc = tf.constant(min_loc, dtype = tf.float32)
        self.max_loc = tf.constant(max_loc, dtype = tf.float32)
        self.bins = tf.cast(tf.cast(bins, dtype = tf.int32), dtype = tf.float32)
        self.loc_delta = (self.max_loc - self.min_loc) / self.bins
        loc_map = np.meshgrid(*[np.arange(_min_loc, _max_loc, _delta) for _min_loc, _max_loc, _delta in  zip(self.min_loc, self.max_loc, self.loc_delta)])
        loc_map = tf.constant(loc_map, dtype =tf.float32)
        loc_map = tf.reshape(loc_map,[*loc_map.shape,1])
        self.loc_map = tf.transpose(loc_map, perm=[3, 1, 2, 0])

    def __call__(self):
        return self.loc_map


class SlidingWindowPatches(tf.keras.layers.Layer):
    def __init__(self, size, stride=[1,1], padding='VALID'):
        super().__init__()
        self.size = size
        self.stride = stride
        self.padding = padding

    def call(self, inputs):
        inputs = tf.reshape(inputs, [*inputs.shape,1])
        patched = tf.extract_volume_patches(inputs,[1,*self.size,1,1],[1,*self.stride,1,1],padding=self.padding)
        return patched



class Locator(tf.keras.Model):

    def __init__(self, min_loc, max_loc, bins, window=[5,5]):
        super().__init__()
        self.dl = DiscretLocation(bins)
        self.dtcl = DiscretToContinuousLocation(stride=[1,1], min_loc = min_loc, max_loc = max_loc, window=window)
        self.loss = DiscretLocationMSE()

    def call(self, inputs):
        discret_location = self.dl()
        loc = self.dtcl(discret_location)
        return loc

class DiscretLocationMSE(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def __call__(self, predictions, goals):
        corr, pos = tf.split(predictions, 2, axis=-1)
        corr_goal, pos_goal = tf.split(goals, 2, axis=-1)
        corr_loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(corr, corr_goal))
        pos_loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(pos, pos_goal))
        loss = corr_loss + 10*pos_loss
        return 100*loss, 100*corr_loss, 100*pos_loss

class DiscretLocation(tf.keras.layers.Layer):
    def __init__(self, bins=[10,10]):
        super().__init__()
        self.bins = tf.constant(bins, dtype = tf.int32)
        self.probability_representation = self.add_weight(trainable=True, shape = [1,*self.bins.numpy(),1], initializer = tf.initializers.glorot_uniform())
        self.correctness_representation = self.add_weight(trainable=True, shape = [1,*self.bins.numpy(),1], initializer = tf.initializers.glorot_uniform())

    def __call__(self):
        return tf.concat([tf.abs(self.probability_representation),self.correctness_representation],axis=-1)

class DiscretToContinuousLocation(tf.keras.layers.Layer):

    def __init__(self, window=[5,5], stride=[1,1], min_loc=[0,0], max_loc=[]):
        super().__init__()
        self.min_loc = tf.constant(min_loc, dtype = tf.float32)
        self.max_loc = tf.constant(max_loc, dtype = tf.float32)
        self.bins = None
        self.loc_delta = None
        self.loc_patches = None
        self.window = tf.cast(window, dtype = tf.float32)
        self.stride = tf.cast(stride, dtype = tf.float32)
        self.window_center_index = tf.cast(tf.reduce_prod(self.window)/2.+0.5, dtype = tf.int32)-1
        self.weight1 = self.add_weight(trainable=True, shape = [1], initializer = tf.initializers.constant(0.5))
        self.bias1 = self.add_weight(trainable=True, shape = [1], initializer = tf.initializers.constant(10.))
        self.weight2 = self.add_weight(trainable=True, shape = [1], initializer = tf.initializers.constant(0.5))
        self.bias2 = self.add_weight(trainable=True, shape = [1], initializer = tf.initializers.constant(10.))


    def build(self, inputs_shape):
        self.bins = tf.constant(inputs_shape[1:3], dtype = tf.int32)
        if len(self.max_loc) == 0:
            self.max_loc = tf.constant(self.bins, dtype = tf.float32)
        self.loc_delta = (self.max_loc - self.min_loc) / tf.cast(self.bins, tf.float32)
        loc_map = LocationMap(self.min_loc,self.max_loc, self.bins)()
        self.loc_patches = SlidingWindowPatches(self.window, self.stride, padding='SAME')(loc_map)
        self.patches = SlidingWindowPatches(self.window, self.stride, padding='SAME')
        self.variance_offset = (self.loc_delta/2)**2
        self.varianz_max = ((self.max_loc-self.min_loc)*0.5)**2.0 - self.variance_offset

    def call(self, discret_loc):
        probability_representation,correctness_representation = tf.split(discret_loc,2,axis=-1)

        occurrence_patches = self.patches(probability_representation)
        occurrence_sum = tf.reduce_sum(occurrence_patches, axis=[-1])
        occurrence_sum = tf.reshape(occurrence_sum, [*occurrence_sum.shape,1])
        probability_patches = occurrence_patches / occurrence_sum

        loc = tf.reduce_sum(probability_patches * self.loc_patches, axis=[-1])
        loc_blow = tf.reshape(loc, [*loc.shape,1])

        variance = tf.reduce_sum(probability_patches * (self.loc_patches - loc_blow)**2, axis=[-1])

        correctness_patches = self.patches(correctness_representation)
        center_probability = correctness_patches[:,:,:,:,self.window_center_index]
        pos_marker1 = discret_sigmoid((center_probability*self.weight1-self.bias1))
        pos_marker2 = discret_sigmoid((center_probability*self.weight2-self.bias2))
        
        correctness = 1-1*(tf.math.maximum(variance,self.variance_offset) - self.variance_offset)/self.varianz_max
        correctness = correctness * pos_marker2

        loc = loc * pos_marker1

        continuous_location = tf.concat([correctness, loc],axis=-1)
        return continuous_location

@tf.custom_gradient
def discret_sigmoid(inputs):

    activated = tf.minimum(tf.maximum(inputs/8.+0.5, 0.), 1.)

    def grad(dy):
        with tf.GradientTape() as tape:
            tape.watch(inputs)
            act = tf.nn.sigmoid(inputs)

        return dy*tape.gradient(act, inputs)
    return activated, grad

class ContinuousLocationMap(tf.keras.layers.Layer):
    
    def __init__(self, min_loc=[0,0], max_loc=[10,10], bins=[10,10]):
        super().__init__()
        self.bins = tf.cast(tf.constant(bins, tf.int32),tf.float32)
        self.min_loc = tf.constant(min_loc, dtype = tf.float32)
        self.max_loc = tf.constant(max_loc, dtype = tf.float32)
        self.loc_delta = (self.max_loc - self.min_loc) / self.bins
        self.loc_repr_base = tf.fill([1,*self.bins,2],0.0)
        self.loc_repr = self.add_weight(trainable=False, shape = [1,*tf.cast(self.bins, tf.int32),2], initializer = tf.initializers.zeros())
        self.corr_repr_base = tf.fill([1,*self.bins,2],0.0)
        self.corr_repr = self.add_weight(trainable=False, shape = [1,*tf.cast(self.bins, tf.int32),2], initializer = tf.initializers.zeros())


    def __call__(self, batch):
        loc_map = tf.concat([self.map_from_list(loc_list) for loc_list in batch], axis=0)
        return loc_map
        
    
    def map_from_list(self,loc_list):
        self.loc_repr.assign(self.loc_repr_base)
        self.corr_repr.assign(self.corr_repr_base)
        for loc in loc_list:
            index = tf.cast((loc-self.min_loc) / self.loc_delta+0.5, tf.int32)
            self.loc_repr[0,index[0],index[1], 0].assign(loc[0])
            self.loc_repr[0,index[0],index[1], 1].assign(loc[1])
            self.corr_repr[0,index[0],index[1], 0].assign(1.)
            self.corr_repr[0,index[0],index[1], 1].assign(1.)
        return tf.concat([self.corr_repr,self.loc_repr], axis=-1)


if __name__ == '__main__':
    main()
    