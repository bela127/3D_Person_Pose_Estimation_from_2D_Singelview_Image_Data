import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import ShAReD_Net.model.layer.heatmap_1d as heatmap_1d
import ShAReD_Net.model.layer.heatmap_2d as heatmap_2d
import ShAReD_Net.training.train as train
import ShAReD_Net.training.loss.base as loss_base

def main():
    keypoints = 15
    x = y = z = 250
    singel_gt = tf.constant([[x+245*kp,y+204*kp,z+200*kp] for kp in range(keypoints)],dtype = tf.float32)
    singel_feature = tf.constant([1.] * 4,dtype = tf.float32)
    dataset = tf.data.Dataset.from_tensors((singel_feature, singel_gt))
    
    def get_train_model():
        return loss_base.LossTestTrainingsModel(keypoints = keypoints)

    #dist_strat = tf.distribute.MirroredStrategy()
    dist_strat = tf.distribute.MirroredStrategy(cross_device_ops = tf.distribute.HierarchicalCopyAllReduce())
    #dist_strat = tf.distribute.MirroredStrategy(cross_device_ops = tf.distribute.ReductionToOneDevice())

    #dist_strat = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    step_callbacks = train.standart_callbacks()
    step_callbacks[400] = finish_training
    step_callbacks[0] = init_model
    
    train.train(400, get_train_model, dataset, dist_strat, batch_size = 16, step_callbacks = step_callbacks)

def finish_training(train_model, loss, step):
    prob_maps = tf.unstack(train_model.representation, axis=-1)
    for prob_map_batch in prob_maps:
        prob_map_batch = heatmap_2d.feature_to_location_propability_map(prob_map_batch)
        loc_map_xy = train_model.loss.loc_map_xy([0.,0.])
        loc_xy = heatmap_2d.propability_map_to_location(tf.expand_dims(prob_map_batch,axis=-1),loc_map_xy)
        print(loc_xy)
        for prob_map in prob_map_batch:
            plt.imshow(prob_map)
            plt.show()

def save_checkpoint(train_model, loss, step):
    pass

def init_model(train_model):
    print("Init Model")
    try_run(train_model)
    
@tf.function
def try_run(train_model):
    keypoints = 15
    x = y = z = 250
    singel_gt = tf.constant([[[x+245*kp,y+204*kp,z+200*kp] for kp in range(keypoints)]]*4,dtype = tf.float32)
    singel_feature = tf.constant([1.] * 4,dtype = tf.float32)
    train_model([singel_feature,singel_gt])

def validation_loop(train_model, loss, step):
    pass

if __name__ == "__main__":
    main()