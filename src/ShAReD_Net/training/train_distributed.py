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
    callbacks.at_step = {}
    
    callbacks.every_eval_steps = {}
    
    callbacks.make_batches = None
    callbacks.train_init = None
    
    callbacks.create_train_dataset = None
    callbacks.create_test_dataset = None
    
    callbacks.create_ckpt = None
    callbacks.create_model = None
    
    callbacks.create_loss = None
    callbacks.create_eval_loss = None
    
    callbacks.create_opt = None
    
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

def print_step(dev, step, batch, output, loss, extra_loss, ckpt, manager, train_model, grads):
    tf.print("Step:", step)
    
def print_loss(dev, step, batch, output, loss, extra_loss, ckpt, manager, train_model, grads):
    tf.print("Loss:", loss)
    
def show_plot(dev, step, batch, output, loss, extra_loss, ckpt, manager, train_model, grads):
    prob_maps = tf.unstack(train_model.representation, axis=-1)
    for prob_map_batch in prob_maps:
        prob_map_batch = heatmap_2d.feature_to_location_propability_map(prob_map_batch)
        loc_map_xy = train_model.loss.pose_loss_xy.loc_map_xy([0.,0.])
        loc_xy = heatmap_2d.propability_map_to_location(tf.expand_dims(prob_map_batch,axis=-1),loc_map_xy)
        print(loc_xy)
        for prob_map in prob_map_batch:
            plt.imshow(prob_map)
            plt.show()
            
def train(steps, dist_strat, batch_size = 8, learning_rate = 0.01, callbacks = standart_callbacks()):
    writer_train = tf.summary.create_file_writer(config.tensorboard.path+"/train")
    writer_eval = tf.summary.create_file_writer(config.tensorboard.path+"/eval")
    
    if dist_strat is None:
        #singel_dev_train.train(steps = steps, get_train_model = get_train_model, dataset = dataset, batch_size = batch_size, learning_rate = learning_rate, step_callbacks = step_callbacks)
        print("changed interface so None is not supported use a starategie")
        return
    
    if callbacks.create_train_dataset:
        per_replica_batch_size = batch_size // dist_strat.num_replicas_in_sync
        dataset = callbacks.create_train_dataset(per_replica_batch_size)
    else:
        print("No train dataset, set create_train_dataset in the callbacks")
        return
    
    if callbacks.create_test_dataset:
        per_replica_batch_size = batch_size // dist_strat.num_replicas_in_sync
        test_dataset = callbacks.create_test_dataset(per_replica_batch_size)
    else:
        test_dataset = dataset
    
    dataset = dataset.repeat(-1)
    test_dataset = test_dataset.repeat(-1)
    
    def make_dist_dataset(dataset, steps = None):
        def dataset_fn(input_context):
            per_replica_batch_size = input_context.get_per_replica_batch_size(batch_size)
            d = callbacks.make_batches(dataset, per_replica_batch_size)
            if steps is None:
                return d.shard(input_context.num_input_pipelines, input_context.input_pipeline_id).take(1)
            else:
                return d.shard(input_context.num_input_pipelines, input_context.input_pipeline_id).take(steps)
        return dataset_fn
    
    if callbacks.make_batches:
        dist_dataset = dist_strat.experimental_distribute_datasets_from_function(make_dist_dataset(dataset, steps))
        dist_test_dataset = dist_strat.experimental_distribute_datasets_from_function(make_dist_dataset(test_dataset))
    else:
        dataset = dataset.batch(batch_size).take(steps)
        test_dataset = test_dataset.batch(batch_size).take(1)
        dist_dataset = dist_strat.experimental_distribute_dataset(dataset)
        dist_test_dataset = dist_strat.experimental_distribute_dataset(test_dataset)
    
    step = tf.Variable(0, trainable=False, dtype = tf.int32)
    
    with dist_strat.scope():
        
        if callbacks.create_opt:
            optimizer1 = callbacks.create_opt()
        else:
            optimizer1 = tf.keras.optimizers.Nadam(learning_rate = config.training.learning_rate, epsilon=0.001)
        
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
        
        if callbacks.create_eval_loss:
            eval_loss = callbacks.create_eval_loss()
        else:
            eval_loss = callbacks.create_loss()
        
        if callbacks.train_init:
            callbacks.train_init(train_model, train_loss, eval_loss, dist_dataset, dist_test_dataset)
    
        if callbacks.create_ckpt:
            ckpt, manager = callbacks.create_ckpt(step, optimizer1, train_model, train_loss)
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
        
        extra_loss_list = train_model.losses# + train_loss.losses
        if callbacks.grad_pre:
            loss_per_batch, extra_loss, trainable_vars = callbacks.grad_pre(loss, extra_loss_list, batch, optimizer1, train_model, train_loss)
        else:
            loss_per_batch = loss
            extra_loss = tf.reduce_sum(extra_loss_list)
            trainable_vars = train_model.low_level_extractor.trainable_variables + train_model.encoder.trainable_variables + train_model.pos_decoder.trainable_variables + train_model.pose_decoder.trainable_variables+ train_loss.trainable_variables
        
        loss_per_input1 = loss_per_batch / tf.cast(batch_size, dtype=loss_per_batch.dtype)
        
        agg_loss1 = loss_per_input1 + extra_loss

        dev = tf.distribute.get_replica_context().devices
        
        print(f"tracing gradients on {dev}")
        gradients1 = optimizer1.get_gradients(agg_loss1, trainable_vars)

        non_nan_gradients1 = [tf.where(tf.math.is_nan(grad), tf.zeros_like(grad), grad) for grad in gradients1]
        capped_gradients1, norm = tf.clip_by_global_norm(non_nan_gradients1, 10.)
        
        to_optimize1 = zip(capped_gradients1, trainable_vars)
        
        optimizer1.apply_gradients(to_optimize1)
        
        for callback_step, step_callback in callbacks.every_steps.items():
            if callback_step > 0 and step % callback_step == 0:
                step_callback(dev, step, batch, output, loss, extra_loss_list, ckpt, manager, train_model, non_nan_gradients1)
        for callback_step, step_callback in callbacks.at_step.items():
            if step == callback_step:
                step_callback(dev, step, batch, output, loss, extra_loss_list, ckpt, manager, train_model, non_nan_gradients1)
    
    @tf.function(experimental_relax_shapes=True)
    def train_step(batch): 
        dist_strat.experimental_run_v2(singel_device_train_step, args=(batch,))
    
    @tf.function(experimental_relax_shapes=True)
    def singel_device_eval_step(batch, step_callback):
        
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
        loss_eval = eval_loss(loss_input)
        
        dev = tf.distribute.get_replica_context().devices
        
        step_callback(dev, step, batch, output, loss, loss_eval)
        
        
    @tf.function
    def eval_step():
        for callback_step, step_callback in callbacks.every_eval_steps.items():
            if callback_step > 0 and step % callback_step == 0:
                #TODO FIX itterating over dataset with only one element, dist_dataset dont support singel element fetching
                for batch in dist_test_dataset:
                    dist_strat.experimental_run_v2(singel_device_eval_step, args=(batch, step_callback))
    
    @tf.function
    def train_loop():  
        with dist_strat.scope():
            for batch in dist_dataset:
                step.assign_add(1)
                with writer_train.as_default():
                    tf.summary.experimental.set_step(tf.cast(step,tf.int64))
                    train_step(batch)
                    writer_train.flush()
                
                with writer_eval.as_default():
                    tf.summary.experimental.set_step(tf.cast(step,tf.int64))
                    #eval_step()
                    writer_eval.flush()
        
    train_loop()

if __name__ == "__main__":
    main()