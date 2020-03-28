import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import ShAReD_Net.training.train as singel_dev_train

import ShAReD_Net.model.layer.heatmap_1d as heatmap_1d
import ShAReD_Net.model.layer.heatmap_2d as heatmap_2d
import ShAReD_Net.training.loss.base as loss_base



def standart_callbacks():
    return {1: print_step, 5: print_loss}

def main():
    keypoints = 15
    x = y = z = 250
    singel_gt = tf.constant([[x+245*kp,y+204*kp,z+200*kp] for kp in range(keypoints)],dtype = tf.float32)
    singel_input = tf.constant([1.] * 4,dtype = tf.float32)
    dataset = tf.data.Dataset.from_tensors((singel_input, singel_gt))
    
    def get_train_model():
        return loss_base.LossTestTrainingsModel(keypoints = keypoints)

    #dist_strat = tf.distribute.MirroredStrategy()
    #dist_strat = tf.distribute.MirroredStrategy(cross_device_ops = tf.distribute.HierarchicalCopyAllReduce())
    #dist_strat = tf.distribute.MirroredStrategy(cross_device_ops = tf.distribute.ReductionToOneDevice())

    #dist_strat = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    dist_strat = tf.distribute.OneDeviceStrategy(device="/cpu:0")
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
        loc_map_xy = train_model.loss.pose_loss_xy.loc_map_xy([0.,0.])
        loc_xy = heatmap_2d.propability_map_to_location(tf.expand_dims(prob_map_batch,axis=-1),loc_map_xy)
        print(loc_xy)
        for prob_map in prob_map_batch:
            plt.imshow(prob_map)
            plt.show()

def train(steps, get_train_model, dataset, dist_strat, batch_size = 8, learning_rate = 0.01, step_callbacks = standart_callbacks()):
    if dist_strat is None:
        singel_dev_train.train(steps = steps, get_train_model = get_train_model, dataset = dataset, batch_size = batch_size, learning_rate = learning_rate, step_callbacks = step_callbacks)
        return
    
    batch_size = tf.cast(batch_size, dtype=tf.int64)
    with dist_strat.scope():
        optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate,epsilon=0.01)
        train_model = get_train_model()
    
    input_preprocessing_callback = step_callbacks.get("input_preprocessing",None)
    batching_callback = step_callbacks.get("batching",None)
    init_callback = step_callbacks.get("train_init",None)
    loss_pre_callback = step_callbacks.get("loss_pre",None)
    
    dataset = dataset.repeat(-1)
    
    if batching_callback:
        def dataset_fn(input_context):
            per_replica_batch_size = input_context.get_per_replica_batch_size(batch_size)
            d = batching_callback(dataset, per_replica_batch_size)
            return d.shard(input_context.num_input_pipelines, input_context.input_pipeline_id)
        dist_dataset = dist_strat.experimental_distribute_datasets_from_function(dataset_fn)
    else:
        dataset = dataset.batch(batch_size)
        dist_dataset = dist_strat.experimental_distribute_dataset(dataset)
    
    with dist_strat.scope():
        if init_callback:
            init_callback(train_model, dist_dataset)
    
    def train_loop(dist_dataset):
        step = tf.Variable(0, trainable=False, dtype = tf.int32)

        @tf.function
        def train_step(batch):
            
            def singel_device_train_step(batch):
                if input_preprocessing_callback:
                    inputs = input_preprocessing_callback(batch)
                loss = train_model(batch)
                
                if loss_pre_callback:
                    loss_per_input = loss_pre_callback(loss)
                loss_per_input = loss_per_input / tf.cast(batch_size, dtype=tf.float32)
            
                trainable_vars = train_model.trainable_variables
                                
                gradients = optimizer.get_gradients(loss_per_input, trainable_vars)
                
                tf.print("gradients",[(var.name,grad) for var,grad in zip(trainable_vars,gradients)])
                non_nan_gradients = [tf.where(tf.math.is_nan(grad), tf.zeros_like(grad), grad) for grad in gradients]
                capped_gradients, _ = tf.clip_by_global_norm(non_nan_gradients, 10.)

                to_optimize = zip(capped_gradients, trainable_vars)
                optimizer.apply_gradients(to_optimize)
                return loss
                
            
            per_example_losses = dist_strat.experimental_run_v2(singel_device_train_step, args=(batch,))
            return per_example_losses
            
        with dist_strat.scope():
            for batch in dist_dataset:
                step.assign_add(1)
                if steps < 0 or step > steps:
                    break
                loss = train_step(batch)
                if step_callbacks:
                    for callback_step, step_callback in step_callbacks.items():
                        if isinstance(callback_step,int) and callback_step > 0 and step % callback_step == 0:
                            step_callback(train_model, loss, step)
        
    train_loop(dist_dataset)

if __name__ == "__main__":
    main()