from addict import Dict

import ShAReD_Net.data.load.dataset_jta.loader as dataset_jta

default_config = Dict()

config = default_config

def set_config(config):
    config.update(config)

config.dataset = dataset_jta
config.dataset.IMG_PATH = "/media/inferics/DataSets/Public_Datasets/JTA-Dataset/images"
config.dataset.ANNO_PATH = "/media/inferics/DataSets/Public_Datasets/JTA-Dataset/new_image_annotations"

config.model.data.cut_steps = 20
config.model.data.cut_delta = 1.5
config.model.data.cut_min_dist = 5
config.model.data.upscaling = 64

config.model.roi_size = [35,35]
config.model.img_downsampling = 4
config.model.roi_downsampling = 16
config.model.z_bins = 20

config.model.output.keypoints = 15

config.checkpoint.path = "/home/inferics/Docker/volumes/3D_Person_Pose_Estimation_from_2D_Singelview_Image_Data/checkpoints"
config.tensorboard.path = "/home/inferics/Docker/volumes/3D_Person_Pose_Estimation_from_2D_Singelview_Image_Data/logdir"

config.training.learning_rate = 0.001
config.training.batch_size = 4
config.training.regularization_rate = 0.0001

config.training.weighting.detection = 50
config.training.weighting.xy_loc = 10
config.training.weighting.xy_var = 1
config.training.weighting.z_loc = 200
config.training.weighting.z_var = 1