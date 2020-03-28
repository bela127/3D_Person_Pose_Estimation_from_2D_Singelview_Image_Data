import time
import itertools

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def init_devs():
    tf.keras.mixed_precision.experimental.set_policy('mixed_bfloat16')

    #tf.config.experimental.set_lms_enabled(True)
    
    devs = tf.config.get_visible_devices()
    print(devs)

    print(tf.config.threading.get_inter_op_parallelism_threads())
    print(tf.config.threading.get_intra_op_parallelism_threads())
    tf.config.threading.set_inter_op_parallelism_threads(12)
    tf.config.threading.set_intra_op_parallelism_threads(12)
    print(tf.config.threading.get_inter_op_parallelism_threads())
    print(tf.config.threading.get_intra_op_parallelism_threads())
    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    gpus = gpus[1:] 
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
    dtype = tf.float32
    tf.keras.backend.set_floatx(dtype.name)
    
    #tf.debugging.enable_check_numerics()
    #tf.config.experimental_run_functions_eagerly(True)
    
    #from tensorflow.python import debug as tf_debug
    
    #sess = tf_debug.LocalCLIDebugWrapperSession(tf.compat.v1.Session())
    #sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
    #sess.__enter__()
    
    input("start?")
    print()

init_devs()

import ShAReD_Net.training.model as train_model
import ShAReD_Net.training.train_distributed as train


def main():
    
    dataset = tf.data.Dataset.from_generator( 
     training_data(), 
     (tf.float32, (tf.float32, tf.float32) ,tf.float32,tf.float32), 
     (tf.TensorShape([None,None,3]), (tf.TensorShape([None]),tf.TensorShape([None,15,3])),tf.TensorShape([]),tf.TensorShape([]))) 

    
    def get_train_model():
        return train_model.TrainingModel(max_loc_xy = [2500,2500])

    #dist_strat = tf.distribute.MirroredStrategy()
    #dist_strat = tf.distribute.MirroredStrategy(cross_device_ops = tf.distribute.HierarchicalCopyAllReduce())
    #dist_strat = tf.distribute.MirroredStrategy(cross_device_ops = tf.distribute.ReductionToOneDevice())

    dist_strat = tf.distribute.OneDeviceStrategy(device="/device:CPU:0")
    #dist_strat = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    #dist_strat = tf.distribute.experimental.CentralStorageStrategy()
    #dist_strat = None
    
    step_callbacks = train.standart_callbacks()
    step_callbacks[2] = finish_training
    #step_callbacks["train_init"] = init_model
    step_callbacks["batching"] = batching
    step_callbacks["input_preprocessing"] = input_preprocessing 
    step_callbacks["loss_pre"] = loss_pre
    
    profile = False
    if profile:
        tf.profiler.experimental.start('/home/inferics/Docker/volumes/3D_Person_Pose_Estimation_from_2D_Singelview_Image_Data/logdir')
        #tf.profiler.experimental.start('./logdir')

    try:
        train.train(2, get_train_model, dataset, dist_strat, batch_size = 4, learning_rate=0.001, step_callbacks = step_callbacks)
    except Exception as e:
        print(e)
        input("throw?")
        if profile:
            tf.profiler.experimental.stop()
        raise e
    
    input("END?")
    if profile:
        tf.profiler.experimental.stop()

def finalize(train_model, loss, step):
    tf.Graph.finalize(tf.compat.v1.get_default_graph())


def loss_pre(loss):
    detection_loss, estimator_loss = loss
    loss_xy, loss_z = estimator_loss
    return tf.reduce_sum(loss_xy) + tf.reduce_sum(loss_z) #tf.reduce_sum(detection_loss) + 
    
time_start = time.time()
def input_preprocessing(inputs):
    (images, (batch_indexes, person_poses), focal_lengths, crop_factors) = inputs
    #tf.print("input",images.shape, (batch_indexes.shape, person_poses.shape), focal_lengths.shape, crop_factors.shape)
    global time_start
    time_end = time.time()
    print(time_end - time_start)
    time_start = time.time()
    return inputs
    
def batching(dataset, batch_size):
    data_itter = iter(dataset)
    
    def batch_gen():
        while True:
            time_start = time.time() #TODO
            images = []
            batch_indexes = []
            person_poses = []
            focal_lengths = []
            crop_factors = []
            for b in range(batch_size):
                _input = next(data_itter)
                (image, (batch_index, person_pose), focal_length, crop_factor) = _input
                images.append(image)
                batch_indexes.append(batch_index)
                person_poses.append(person_pose)
                focal_lengths.append(focal_length)
                crop_factors.append(crop_factor)
            images = tf.stack(images)
            batch_indexes = tf.concat(batch_indexes, axis=0)
            batch_indexes = batch_indexes - batch_indexes[0,...][None,...]
            person_poses = tf.concat(person_poses, axis=0)
            focal_lengths = focal_lengths[0]
            crop_factors = crop_factors[0]
            batch = (images, (batch_indexes, person_poses), focal_lengths, crop_factors)
            yield batch

    batched_dataset = tf.data.Dataset.from_generator( 
     batch_gen, 
     (tf.float32, (tf.float32, tf.float32) ,tf.float32,tf.float32), 
     (tf.TensorShape([batch_size,None,None,3]), (tf.TensorShape([None]),tf.TensorShape([None,15,3])),tf.TensorShape([]),tf.TensorShape([])))
    return batched_dataset.prefetch(25)
    
def training_data():
    image = tf.constant(100.,shape=[384,384,3],dtype=tf.float32)
    
    min_loc_xyz=tf.constant([0,0,50],dtype=tf.float32)
    loc_delta_xyz=tf.constant([150,150,150],dtype=tf.float32)
    
    pos_01=tf.constant([5,5,5],dtype=tf.float32)
    pos_02=tf.constant([2,2,8],dtype=tf.float32)
    pos_11=tf.constant([8,1,0],dtype=tf.float32)
    pos_12=tf.constant([3,7,1],dtype=tf.float32)
    pos_21=tf.constant([5,5,0],dtype=tf.float32)
    pos_31=tf.constant([5,5,0],dtype=tf.float32)
    pos_32=tf.constant([7,7,0],dtype=tf.float32)
    
    person_poses = [[min_loc_xyz+loc_delta_xyz*pos_01 for _ in range(15)],
                    [min_loc_xyz+loc_delta_xyz*pos_02 for _ in range(15)],
                    [min_loc_xyz+loc_delta_xyz*pos_11 for _ in range(15)],
                    [min_loc_xyz+loc_delta_xyz*pos_12 for _ in range(15)],
                    [min_loc_xyz+loc_delta_xyz*pos_21 for _ in range(15)],
                    [min_loc_xyz+loc_delta_xyz*pos_31 for _ in range(15)],
                    [min_loc_xyz+loc_delta_xyz*pos_32 for _ in range(15)]]
        
    batch_indexes = [0,0,1,1,2,3,3]
    
    focal_length = tf.cast(20, dtype=tf.float32)
    crop_factor = tf.cast(1, dtype=tf.float32)
        
    def training_items():
        for i in itertools.count(0):
            pose_batches = []
            batch_index = []
            for b, pose in zip(batch_indexes, person_poses):
                if b == i%4:
                    pose_batches.append(pose)
                    batch_index.append(i)
                else:
                    continue
            pose_batches_tensor = tf.cast(pose_batches, dtype=tf.float32)
            batch_index_tensor = tf.cast(batch_index, dtype=tf.float32)
            yield (image, (batch_index_tensor, pose_batches_tensor), focal_length, crop_factor)
    
    return training_items
    

def finish_training(train_model, loss, step):
    print(train_model.count_params())

def save_checkpoint(train_model, loss, step):
    pass

def init_model(train_model, dist_dataset):
    print("Init Model")
    try_run(train_model, dist_dataset)
    
def try_run(train_model, dist_dataset):
    inputs = next(iter(dist_dataset))
    inputs = input_preprocessing(inputs)
    dist_strat = tf.distribute.get_strategy()
    out = dist_strat.experimental_run_v2(train_model, args=(inputs,))

def validation_loop(train_model, loss, step):
    pass

if __name__ == "__main__":
    main()