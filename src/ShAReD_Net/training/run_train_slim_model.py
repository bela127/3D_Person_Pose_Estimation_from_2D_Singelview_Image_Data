import time
import itertools

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def init_devs():
    tf.config.set_soft_device_placement(True)
    
    options = {
                "layout_optimizer": True,
                "constant_folding": True,
                "shape_optimization": True,
                "remapping": True,
                "arithmetic_optimization": True,
                "dependency_optimization": True,
                "loop_optimization": True,
                "function_optimization": True,
                "debug_stripper": False,
                "disable_model_pruning": False,
                "scoped_allocator_optimization": True,
                "pin_to_host_optimization": True,
                "implementation_selector": True,
                "disable_meta_optimizer": True
              }
    #tf.config.optimizer.set_experimental_options(options)

    
    devs = tf.config.get_visible_devices()
    print(devs)

    print(tf.config.threading.get_inter_op_parallelism_threads())
    print(tf.config.threading.get_intra_op_parallelism_threads())
    tf.config.threading.set_inter_op_parallelism_threads(12)
    tf.config.threading.set_intra_op_parallelism_threads(12)
    print(tf.config.threading.get_inter_op_parallelism_threads())
    print(tf.config.threading.get_intra_op_parallelism_threads())

    gpus = tf.config.experimental.list_physical_devices('GPU')
    gpus = gpus[:] 
    if gpus:
        try:
        # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.experimental.set_visible_devices(gpus, 'GPU')
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    logical_devs = tf.config.list_logical_devices()
    physical_devs = tf.config.experimental.list_physical_devices()

    print("physical_devs",physical_devs)
    print("logical_devs", logical_devs)
    
    print(tf.version.VERSION)
    input("start?")

init_devs()

import ShAReD_Net.training.train_distributed as train

import ShAReD_Net.training.slim as training_slim
import ShAReD_Net.model.slim as model_slim
import ShAReD_Net.data.transform.transform as transform

from ShAReD_Net.configure import config


def main():
        
    data_split = "train"
    
    with tf.device("/cpu:0"):
        train_ds = transform.create_dataset(data_split, cut_dist= 8).shuffle(500)

    dist_strat = tf.distribute.MirroredStrategy(cross_device_ops = tf.distribute.HierarchicalCopyAllReduce())

    steps = 50000
    
    step_callbacks = train.standart_callbacks()
    step_callbacks.every_steps[20] = print_loss
    step_callbacks.every_steps[200] = save_checkpoint
    step_callbacks.every_steps[10] = simple_summery
    step_callbacks.every_steps[50] = complex_summery
    
    step_callbacks.at_step[1] = count_params
    step_callbacks.at_step[2] = finalize
    
    step_callbacks.make_batches = batching
    step_callbacks.grad_pre = grad_pre
    step_callbacks.input_pre = input_pre
    step_callbacks.loss_pre = loss_pre
    
    step_callbacks.create_ckpt = create_checkpoint
    step_callbacks.create_loss = train_loss
    step_callbacks.create_model = train_model

    try:
        train.train(steps, train_ds, dist_strat, batch_size = config.training.batch_size, learning_rate=config.training.learning_rate, callbacks = step_callbacks)
    except Exception as e:
        print(e)
        input("throw?")
        raise e
    
    input("END?")
    
def input_pre(batch):
    img, (pos_hm_gt, loss_weights), (roi_indexes, roi_pose_gt) = batch
    return img, roi_indexes
    
def loss_pre(output, batch):
    pose_hm, pos_hm = output
    img, (pos_hm_gt, loss_weights), (roi_indexes, roi_pose_gt) = batch
    return pose_hm, pos_hm, roi_pose_gt, (pos_hm_gt, loss_weights)

def grad_pre(loss, extra_loss, batch):
    (detection_loss, estimator_loss_xy, estimator_loss_z) = loss
    
    extra_loss_sum = tf.reduce_sum(extra_loss)
    detection_loss_sum = tf.reduce_sum(detection_loss) * 1000
    estimator_loss_xy_sum = tf.reduce_sum(estimator_loss_xy)
    estimator_loss_z_sum = tf.reduce_sum(estimator_loss_z) * 100
    
    loss_per_batch = detection_loss_sum + estimator_loss_xy_sum + estimator_loss_z_sum
    
    return loss_per_batch, extra_loss_sum


def train_loss():
    loss = training_slim.SlimTrainingLoss()
    return loss
    
def train_model():
    low_level_extractor = model_slim.LowLevelExtractor(color_channel=13, texture_channel=16, texture_compositions=16, out_channel=32)

    encoder = model_slim.Encoder(dense_blocks_count=2, dense_filter_count=16)
    
    pos_decoder = model_slim.PosDecoder(dense_blocks_count=3, dense_filter_count=8)
    
    pose_decoder = model_slim.PoseDecoder(keypoints=config.model.output.keypoints, z_bins=config.model.z_bins, dense_blocks_count=2, dense_filter_count=16)
    
    model = training_slim.SlimTrainingModel(low_level_extractor, encoder, pos_decoder, pose_decoder)    
    
    return model

def create_checkpoint(step, optimizer, train_model):
    nets = {"low_level_extractor":train_model.low_level_extractor,
            "encoder":train_model.encoder,
            "pos_decoder":train_model.pos_decoder,
            "pose_decoder":train_model.pose_decoder}
    ckpt = tf.train.Checkpoint(step=step, optimizer=optimizer, **nets)
    manager = tf.train.CheckpointManager(ckpt, config.checkpoint.path, max_to_keep=50)
    return ckpt, manager
    
def save_checkpoint(dev, step, batch, output, loss, extra_loss, ckpt, manager, train_model):
    def save():
        save_path = manager.save()
        print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
    tf.py_function(save,[], [])

def finalize(dev, step, batch, output, loss, extra_loss, ckpt, manager, train_model):
    def run():
        print("Finalized")
        tf.Graph.finalize(tf.compat.v1.get_default_graph())
    tf.py_function(run,[], [])

def print_loss(dev, step, batch, output, loss, extra_loss, ckpt, manager, train_model):
    detection_loss, estimator_loss_xy, estimator_loss_z = loss
       
    extra_loss_sum = tf.reduce_sum(extra_loss)
    detection_loss_sum = tf.reduce_sum(detection_loss) * 1000
    estimator_loss_xy_sum = tf.reduce_sum(estimator_loss_xy)
    estimator_loss_z_sum = tf.reduce_sum(estimator_loss_z) * 100
    
    tf.print("On", dev)
    tf.print("detection_loss", detection_loss_sum)
    tf.print("estimator_loss_xy", estimator_loss_xy_sum)
    tf.print("estimator_loss_z", estimator_loss_z_sum)
    tf.print("extra_loss_sum", extra_loss_sum)

def simple_summery(dev, step, batch, output, loss, extra_loss, ckpt, manager, train_model):
    detection_loss, estimator_loss_xy, estimator_loss_z = loss
       
    extra_loss_sum = tf.reduce_sum(extra_loss)
    detection_loss_sum = tf.reduce_sum(detection_loss) * 1000
    estimator_loss_xy_sum = tf.reduce_sum(estimator_loss_xy)
    estimator_loss_z_sum = tf.reduce_sum(estimator_loss_z) * 100
    
    tf.summary.scalar(f"detection_loss {dev}", detection_loss_sum)
    tf.summary.scalar(f"estimator_loss_xy {dev}", estimator_loss_xy_sum)
    tf.summary.scalar(f"estimator_loss_z {dev}", estimator_loss_z_sum)
    tf.summary.scalar(f"extra_loss {dev}", extra_loss_sum)
    
    
def complex_summery(dev, step, batch, output, loss, extra_loss, ckpt, manager, train_model):
    img, (pos_hm_gt, loss_weights), (roi_indexes, roi_pose_gt) = batch
    pose_hm, pos_hm = output
    
    tf.summary.image("image", img, max_outputs=4)
    
    pos_hm_gt_near = pos_hm_gt[...,0,None]
    pos_hm_gt_fare = pos_hm_gt[...,1,None]
    tf.summary.image("pos_hm_gt_near", pos_hm_gt_near, max_outputs=4)
    tf.summary.image("pos_hm_gt_fare", pos_hm_gt_fare, max_outputs=4)
    
    pos_hm_near = pos_hm[...,0,None]
    pos_hm_fare = pos_hm[...,1,None]
    tf.summary.image("pos_hm_near", pos_hm_near, max_outputs=4)
    tf.summary.image("pos_hm_fare", pos_hm_fare, max_outputs=4)
    
    pose_hms = tf.unstack(pose_hm, axis=-1)
    for i, pose_hm in enumerate(pose_hms[:config.model.output.keypoints]):
        tf.summary.image(f"pose_hm for keypoint {i}", pose_hm[...,None], max_outputs=4)
    
    for i, z_slice in enumerate(pose_hms[config.model.output.keypoints:]):
        tf.summary.image(f"z_slice for bin {i}", z_slice[...,None], max_outputs=10)



def count_params(dev, step, batch, output, loss, extra_loss, ckpt, manager, train_model):
    print(train_model.count_params())

def batching(dataset, batch_size):
    batched_ds = dataset.batch(batch_size)

    def unragg(img, pos_stuff, pose_stuff):
        roi_indexes, roi_pose = pose_stuff

        def unragg_indexes(indexes, row_length):
            new_indexes = np.empty([indexes.shape[0], indexes.shape[1]+1], dtype=np.int32)
            new_indexes[:,1:] = indexes
            i = 0
            b = 0
            for length in row_length[1]:
                index = indexes[i:int(i+length)]
                new_indexes[i:int(i+length),0] = b
                i += length
                b += 1
                
            return new_indexes

        roi_indexes_flat = tf.numpy_function(unragg_indexes, [roi_indexes.flat_values, roi_indexes.nested_row_lengths()], Tout=roi_indexes.dtype)

        roi_pose_flat =  roi_pose.flat_values

        return img, pos_stuff, (roi_indexes_flat, roi_pose_flat)

    unragged_ds = batched_ds.map(unragg).prefetch(100)
    return unragged_ds

if __name__ == "__main__":
    main()