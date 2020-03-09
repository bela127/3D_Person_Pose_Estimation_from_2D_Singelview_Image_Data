import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


import ShAReD_Net.model.layer.heatmap_1d as heatmap_1d
import ShAReD_Net.model.layer.heatmap_2d as heatmap_2d
import ShAReD_Net.training.loss.base as loss_base

def standart_callbacks():
    return {1: print_step, 10: print_loss}

def main():
    keypoints = 15
    x = y = z = 250
    singel_gt = tf.constant([[x+245*kp,y+204*kp,z+200*kp] for kp in range(keypoints)],dtype = tf.float32)
    singel_input = tf.constant([1.] * 4,dtype = tf.float32)
    dataset = tf.data.Dataset.from_tensors((singel_input, singel_gt))
    
    def get_train_model():
        return loss_base.LossTestTrainingsModel(keypoints = keypoints)

    #dist_strat = tf.distribute.MirroredStrategy()
    dist_strat = tf.distribute.MirroredStrategy(cross_device_ops = tf.distribute.HierarchicalCopyAllReduce())
    #dist_strat = tf.distribute.MirroredStrategy(cross_device_ops = tf.distribute.ReductionToOneDevice())

    #dist_strat = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    step_callbacks = standart_callbacks()
    step_callbacks[400] = show_plot
    
    train(400, get_train_model, dataset, dist_strat, batch_size = 16, step_callbacks = step_callbacks)

def print_step(train_model, loss, step):
    tf.print("Step:", step)
    
def print_loss(train_model, loss, step):
    tf.print("Loss:", loss)
    
def show_plot(train_model, loss, step):
    prob_maps = tf.unstack(train_model.representation, axis=-1)
    for prob_map_batch in prob_maps:
        prob_map_batch = heatmap_2d.feature_to_location_propability_map(prob_map_batch)
        loc_map_xy = train_model.loss.loc_map_xy([0.,0.])
        loc_xy = heatmap_2d.propability_map_to_location(tf.expand_dims(prob_map_batch,axis=-1),loc_map_xy)
        print(loc_xy)
        for prob_map in prob_map_batch:
            plt.imshow(prob_map)
            plt.show()

def train(steps, get_train_model, dataset, dist_strat, batch_size = 8, learning_rate = 0.01, step_callbacks = {1: print_step, 10: print_loss}):
    with dist_strat.scope():
        optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
        train_model = get_train_model()
        
        init_callback = step_callbacks.get(0,None)
        if init_callback:
            init_callback(train_model)
    
    dataset = dataset.repeat(-1).batch(batch_size)
    dist_dataset = dist_strat.experimental_distribute_dataset(dataset)
    
    def train_loop(dist_dataset):
        step = tf.Variable(0, trainable=False, dtype = tf.int32)

        @tf.function
        def train_step(inputs):
            
            @tf.function
            def singel_device_train_step(inputs):
                with tf.GradientTape() as tape:
                    loss = train_model(inputs)
                    loss_per_input = loss / batch_size
                
                trainable_vars = train_model.trainable_variables
                gradients = tape.gradient(loss_per_input, trainable_vars)
                capped_gradients, _ = tf.clip_by_global_norm(gradients, 10.)
                to_optimize = zip(capped_gradients, trainable_vars)
                optimizer.apply_gradients(to_optimize)
                return loss_per_input
                
            
            per_example_losses = dist_strat.experimental_run_v2(singel_device_train_step, args=(inputs,))
            return per_example_losses
            
        with dist_strat.scope():
            for inputs in dist_dataset:
                step.assign_add(1)
                if steps < 0 or step > steps:
                    break
                loss = train_step(inputs)
                if step_callbacks:
                    for callback_step, step_callback in step_callbacks.items():
                        if callback_step > 0 and step % callback_step == 0:
                            step_callback(train_model, loss, step)
        
    train_loop(dist_dataset)

if __name__ == "__main__":
    main()