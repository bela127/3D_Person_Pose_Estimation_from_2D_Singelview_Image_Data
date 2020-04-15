import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from addict import Dict


import ShAReD_Net.training.train as singel_dev_train

import ShAReD_Net.model.layer.heatmap_1d as heatmap_1d
import ShAReD_Net.model.layer.heatmap_2d as heatmap_2d
import ShAReD_Net.training.loss.base as loss_base

from ShAReD_Net.configure import config

def standart_callbacks():
    callbacks = Dict()
    callbacks.every_steps[1] = print_step
    callbacks.every_steps[10] = print_loss
        #TODO split model in output and loss
    callbacks.at_step = {}

    callbacks.make_batches = None
    callbacks.train_init = None
    
    callbacks.create_ckpt = None
    callbacks.create_model = None
    callbacks.create_loss = None
    
    callbacks.input_pre = None
    callbacks.loss_pre = None
    callbacks.grad_pre = None
    return callbacks

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
    step_callbacks.step_step[400] = show_plot
    
    train(400, get_train_model, dataset, dist_strat, batch_size = 16, callbacks = step_callbacks)

def print_step(dev, step, batch, output, loss, extra_loss, ckpt, manager, train_model):
    tf.print("Step:", step)
    
def print_loss(dev, step, batch, output, loss, extra_loss, ckpt, manager, train_model):
    tf.print("Loss:", loss)
    
def show_plot(dev, step, batch, output, loss, extra_loss, ckpt, manager, train_model):
    prob_maps = tf.unstack(train_model.representation, axis=-1)
    for prob_map_batch in prob_maps:
        prob_map_batch = heatmap_2d.feature_to_location_propability_map(prob_map_batch)
        loc_map_xy = train_model.loss.pose_loss_xy.loc_map_xy([0.,0.])
        loc_xy = heatmap_2d.propability_map_to_location(tf.expand_dims(prob_map_batch,axis=-1),loc_map_xy)
        print(loc_xy)
        for prob_map in prob_map_batch:
            plt.imshow(prob_map)
            plt.show()

def train(steps, dataset, dist_strat, batch_size = 8, learning_rate = 0.01, callbacks = standart_callbacks()):
    writer = tf.summary.create_file_writer(config.tensorboard.path)
    
    if dist_strat is None:
        #singel_dev_train.train(steps = steps, get_train_model = get_train_model, dataset = dataset, batch_size = batch_size, learning_rate = learning_rate, step_callbacks = step_callbacks)
        print("changed interface so None is not supported use a starategie")
        return
    
    dataset = dataset.repeat(-1)
    
    if callbacks.make_batches:
        def dataset_fn(input_context):
            per_replica_batch_size = input_context.get_per_replica_batch_size(batch_size)
            d = callbacks.make_batches(dataset, per_replica_batch_size)
            return d.shard(input_context.num_input_pipelines, input_context.input_pipeline_id).take(steps)
        dist_dataset = dist_strat.experimental_distribute_datasets_from_function(dataset_fn)
    else:
        dataset = dataset.batch(batch_size).take(steps)
        dist_dataset = dist_strat.experimental_distribute_dataset(dataset)
    
    step = tf.Variable(0, trainable=False, dtype = tf.int32)
    
    with dist_strat.scope():
        
        optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate,epsilon=0.01)
        
        if callbacks.create_model:
            train_model = callbacks.create_model()
        else:
            print("No train_model, set create_model in the callbacks")
            return
        
        if callbacks.create_loss:
            train_loss = callbacks.create_loss()
        else:
            print("No train_loss, set create_loss in the callbacks")
            return
        
        if callbacks.train_init:
            callbacks.train_init(train_model, dist_dataset)
    
        if callbacks.create_ckpt:
            ckpt, manager = callbacks.create_ckpt(step, optimizer, train_model)
            ckpt.restore(manager.latest_checkpoint)
            if manager.latest_checkpoint:
                tf.print("Restored from {}".format(manager.latest_checkpoint))
            else:
                tf.print("Initializing from scratch.")
        else:
            ckpt = manager = None

    @tf.function(experimental_relax_shapes=True)
    def singel_device_train_step(batch):
        
        if callbacks.input_pre:
            inputs = callbacks.input_pre(batch)
        else:
            inputs = batch
            
        output = train_model(inputs)
        
        if callbacks.loss_pre:
            loss_input = callbacks.loss_pre(output, batch)
        else:
            loss_input = output
            
        loss = train_loss(loss_input)
        
        if callbacks.grad_pre:
            loss_per_batch, extra_loss = callbacks.grad_pre(loss, train_model.losses, batch)
        else:
            loss_per_batch = loss
            extra_loss = tf.reduce_sum(train_model.losses)
        
        loss_per_input = loss_per_batch / tf.cast(batch_size, dtype=loss_per_batch.dtype)
        
        agg_loss = loss_per_input + extra_loss
        
        trainable_vars = train_model.trainable_variables
        
        dev = tf.distribute.get_replica_context().devices
        print(f"tracing gradients on {dev}")
        gradients = optimizer.get_gradients(agg_loss, trainable_vars)
        
        #tf.print("gradients",[(var.name,grad) for var,grad in zip(trainable_vars,gradients)])
        non_nan_gradients = [tf.where(tf.math.is_nan(grad), tf.zeros_like(grad), grad) for grad in gradients]
        capped_gradients, _ = tf.clip_by_global_norm(non_nan_gradients, 10.)

        to_optimize = zip(capped_gradients, trainable_vars)
        optimizer.apply_gradients(to_optimize)
        
        for callback_step, step_callback in callbacks.every_steps.items():
            if callback_step > 0 and step % callback_step == 0:
                step_callback(dev, step, batch, output, loss, train_model.losses, ckpt, manager, train_model)
        for callback_step, step_callback in callbacks.at_step.items():
            if step == callback_step:
                step_callback(dev, step, batch, output, loss, train_model.losses, ckpt, manager, train_model)
    
    @tf.function(experimental_relax_shapes=True)
    def train_step(batch): 
        dist_strat.experimental_run_v2(singel_device_train_step, args=(batch,))
    
    @tf.function
    def train_loop():  
        with dist_strat.scope():
            for batch in dist_dataset:
                step.assign_add(1)
                tf.summary.experimental.set_step(tf.cast(step,tf.int64))
                train_step(batch)
                writer.flush()
        
    with writer.as_default():
        train_loop()

if __name__ == "__main__":
    main()