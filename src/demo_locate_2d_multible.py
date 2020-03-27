import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np


def main():
    bins = tf.cast([15,15], tf.int32)
    mins = tf.cast([0,0], tf.float32)
    maxs = tf.cast([15,15], tf.float32)

    locator = Locator(min_loc = mins, max_loc = maxs, bins = bins)

    loc_delta = (maxs - mins) / tf.cast(bins, tf.float32)
    clm = ContinuousLocationMap(min_loc = mins, max_loc = maxs, bins = bins)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
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
        goals_raw = np.array([[[5,5]]], dtype=tf.float32)


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

    train_loop(250)
    representation = locator.dl()
    for element in representation:
        image = tf.reshape(element, element.shape[0:2])
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

class SlidingWindow(tf.keras.layers.Layer):
    def __init__(self, size, stride=[1,1], padding='VALID'):
        super().__init__()
        self.size = size
        self.stride = stride
        self.value_count = tf.reduce_prod(size)
        self.padding = padding

    def call(self, inputs):
        channels = inputs.shape[-1]
        inputs = tf.reshape(inputs, [*inputs.shape,1])
        patched = tf.extract_volume_patches(inputs,[1,*self.size,1,1],[1,*self.stride,1,1],padding=self.padding)
        patched = tf.reshape(patched,[-1,1,1,self.value_count])
        patched = tf.transpose(patched, perm=[3,1,2,0])
        patched = tf.batch_to_space(patched,self.size,[[0,0],[0,0]])
        patched = tf.reshape(patched,[1,*self.size,-1,channels])
        patched = tf.transpose(patched, perm=[0,3,1,2,4])
        slides = tf.reshape(patched,[-1,*self.size,channels])
        return slides

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

    def __init__(self, min_loc, max_loc, bins):
        super().__init__()
        self.dl = DiscretLocation(bins)
        self.dtcl = DiscretToContinuousLocation(stride=[4,4], min_loc = min_loc, max_loc = max_loc, window=[5,5])
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
        loss = corr_loss + pos_loss
        return loss, corr_loss, pos_loss

class DiscretLocationMock(tf.keras.layers.Layer):
    def __init__(self, bins=[10,10]):
        super().__init__()
        self.bins = tf.constant(bins, dtype = tf.int32)
        self.representation = self.add_weight(trainable=True, shape = [1,*self.bins.numpy(),1], initializer = tf.initializers.zeros())
        self.representation[0,5,5,0].assign(1.0)

    def __call__(self):
        return self.representation

class DiscretLocation(tf.keras.layers.Layer):
    def __init__(self, bins=[10,10]):
        super().__init__()
        self.bins = tf.constant(bins, dtype = tf.int32)
        self.representation = self.add_weight(trainable=True, shape = [1,*self.bins.numpy(),1], initializer = tf.initializers.he_uniform())

    def __call__(self):
        return self.representation**2

class DiscretToContinuousLocation(tf.keras.layers.Layer):

    def __init__(self, window=[5,5], stride=[1,1], min_loc=[0,0], max_loc=[]):
        super().__init__()
        self.min_loc = tf.constant(min_loc, dtype = tf.float32)
        self.max_loc = tf.constant(max_loc, dtype = tf.float32)
        self.bins = None
        self.loc_delta = None
        self.loc_patches = None
        self.window = tf.constant(window, dtype = tf.int32)
        self.stride = tf.constant(stride, dtype = tf.int32)
        

    def build(self, inputs_shape):
        self.bins = tf.constant(inputs_shape[1:3], dtype = tf.int32)
        if len(self.max_loc) == 0:
            self.max_loc = tf.constant(bins, dtype = tf.float32)
        self.loc_delta = (self.max_loc - self.min_loc) / tf.cast(self.bins, tf.float32)
        loc_map = LocationMap(self.min_loc,self.max_loc, self.bins)()
        self.loc_patches = SlidingWindowPatches(self.window, self.stride, padding='VALID')(loc_map)
        self.occurrence_patches = SlidingWindowPatches(self.window, self.stride, padding='VALID')
        self.variance_offset = (self.loc_delta/2)**2
        self.varianz_max = ((self.max_loc-self.min_loc)*0.5)**2.0 - self.variance_offset

    def call(self, discret_loc):
        occurrence_patches = self.occurrence_patches(discret_loc)
        occurrence_sum = tf.reduce_sum(occurrence_patches, axis=[-1])
        occurrence_sum = tf.reshape(occurrence_sum, [*occurrence_sum.shape,1])
        probability_patches = occurrence_patches / occurrence_sum
        loc = tf.reduce_sum(probability_patches * self.loc_patches, axis=[-1])
        loc_blow = tf.reshape(loc, [*loc.shape,1])
        variance = tf.reduce_sum(probability_patches * (self.loc_patches - loc_blow)**2, axis=[-1])
        
        correctness = 1-(tf.math.maximum(variance,self.variance_offset) - self.variance_offset)/self.varianz_max
        
        continuous_location = tf.concat([correctness, loc],axis=-1)
        return continuous_location

class ContinuousLocationMapYY(tf.keras.layers.Layer):
    
    def __init__(self, min_loc=[0,0], max_loc=[10,10], bins=[10,10]):
        super().__init__()
        self.bins = tf.constant(bins, tf.int32)
        self.min_loc = tf.constant(min_loc, dtype = tf.float32)
        self.max_loc = tf.constant(max_loc, dtype = tf.float32)
        self.loc_delta = (self.max_loc - self.min_loc) / tf.cast(bins, tf.float32)
        self.loc_repr_base = LocationMap(self.min_loc,self.max_loc,self.bins)()
        self.loc_repr = self.add_weight(trainable=False, shape = [1,*self.bins,2], initializer = tf.initializers.zeros())
        self.corr_repr_base = tf.fill([1,*self.bins,2],0.63)
        self.corr_repr = self.add_weight(trainable=False, shape = [1,*self.bins,2], initializer = tf.initializers.zeros())

    def __call__(self, batch):
        return tf.concat([self.map_from_list(loc_list) for loc_list in batch], axis=0)
        
    
    def map_from_list(self,loc_list):
        self.loc_repr.assign(self.loc_repr_base)
        self.corr_repr.assign(self.corr_repr_base)
        for loc in loc_list:
            index = tf.cast(loc / self.loc_delta, tf.int32)
            self.loc_repr[0,index[0],index[1], 0].assign(loc[0])
            self.loc_repr[0,index[0],index[1], 1].assign(loc[1])
            self.corr_repr[0,index[0],index[1], 0].assign(1.)
            self.corr_repr[0,index[0],index[1], 1].assign(1.)
        return tf.concat([self.corr_repr,self.loc_repr], axis=-1)

#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

class ContinuousLocationMap(tf.keras.layers.Layer):
    
    def __init__(self, min_loc=[0,0], max_loc=[10,10], bins=[10,10],stride = [4,4], window=[5,5]):
        super().__init__()
        self.bins = tf.cast(tf.constant(bins, tf.int32),tf.float32)
        self.min_loc = tf.constant(min_loc, dtype = tf.float32)
        self.max_loc = tf.constant(max_loc, dtype = tf.float32)
        self.window = tf.cast(tf.constant(window, tf.int32),tf.float32)
        self.stride = tf.cast(tf.constant(stride, tf.int32),tf.float32)
        window_side = tf.cast(tf.cast(self.window/2,tf.int32),tf.float32)
        self.loc_delta = (self.max_loc - self.min_loc) / tf.cast(bins, tf.float32)
        bins_window = (self.bins-2*window_side)
        min_window = self.min_loc+self.loc_delta*window_side
        max_window = self.min_loc + self.loc_delta * bins_window
        bins_stride = tf.cast(tf.cast((bins_window+1)/self.stride,tf.int32),tf.float32)
        self.loc_repr_base = LocationMap(min_window,max_window,bins_stride)()
        self.loc_repr = self.add_weight(trainable=False, shape = [1,*tf.cast(bins_stride, tf.int32),2], initializer = tf.initializers.zeros())
        self.corr_repr_base = tf.fill([1,*bins_stride,2],0.634)
        self.corr_repr = self.add_weight(trainable=False, shape = [1,*tf.cast(bins_stride, tf.int32),2], initializer = tf.initializers.zeros())


    def __call__(self, batch):
        return tf.concat([self.map_from_list(loc_list) for loc_list in batch], axis=0)
        
    
    def map_from_list(self,loc_list):
        self.loc_repr.assign(self.loc_repr_base)
        self.corr_repr.assign(self.corr_repr_base)
        for loc in loc_list:
            index = tf.cast(loc / self.loc_delta / self.stride, tf.int32)
            for x in range(-1,1):
                window_pos_x = index[0]+x
                if  window_pos_x > self.loc_repr.shape[1]-1: continue
                for y in range(-1,1):
                    window_pos_y = index[1]+y
                    if  window_pos_y > self.loc_repr.shape[2]-1: continue
                    
                    self.loc_repr[0,window_pos_x,window_pos_y, 0].assign(loc[0])
                    self.loc_repr[0,window_pos_x,window_pos_y, 1].assign(loc[1])
                    self.corr_repr[0,window_pos_x,window_pos_y, 0].assign(1.)
                    self.corr_repr[0,window_pos_x,window_pos_y, 1].assign(1.)
        return tf.concat([self.corr_repr,self.loc_repr], axis=-1)

#yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy
class ContinuousLocationMapXX(tf.keras.layers.Layer):

    def __init__(self, min_loc=[0,0], max_loc=[10,10], bins=[10,10], window=[1,1]):
        super().__init__()
        self.bins = tf.constant(bins, tf.int32)
        self.min_loc = tf.constant(min_loc, dtype = tf.float32)
        self.max_loc = tf.constant(max_loc, dtype = tf.float32)
        self.loc_delta = (self.max_loc - self.min_loc) / tf.cast(bins, tf.float32)
        self.loc_repr_base = LocationMap(self.min_loc,self.max_loc,self.bins)()
        self.loc_repr = self.add_weight(trainable=False, shape = [1,*self.bins,2], initializer = tf.initializers.zeros())
        self.corr_repr_base = tf.fill([1,*self.bins,1],0.633)
        self.corr_repr = self.add_weight(trainable=False, shape = [1,*self.bins,1], initializer = tf.initializers.zeros())
        self.window = window

    def __call__(self, batch):
        return tf.concat([self.map_from_list(loc_list) for loc_list in batch], axis=0)
        
    
    def map_from_list(self,loc_list):
        self.loc_repr.assign(self.loc_repr_base)
        self.corr_repr.assign(self.corr_repr_base)
        for loc in loc_list:
            index = tf.cast(loc / self.loc_delta, tf.int32)
            for x in range(int(-self.window[0]/2),int(self.window[0]/2)):
                for y in range(int(-self.window[0]/2),int(self.window[0]/2)):
                    self.loc_repr[0,index[0]+x,index[1]+y, 0].assign(loc[0])
                    self.loc_repr[0,index[0]+x,index[1]+y, 1].assign(loc[1])
                    self.corr_repr[0,index[0]+x,index[1]+y, 0].assign(0.0)
        return tf.concat([self.corr_repr,self.loc_repr], axis=-1)

if __name__ == '__main__':
    main()
    