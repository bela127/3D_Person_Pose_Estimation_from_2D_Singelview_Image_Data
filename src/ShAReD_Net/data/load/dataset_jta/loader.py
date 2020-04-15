import sys
import os

import tensorflow as tf
import numpy as np
import pathlib

## Dataset parameters
cam_intr_f = 1158
x0 = 960
y0 = 540

cam_transform = tf.constant([[cam_intr_f,0,x0],[0,cam_intr_f,y0],[0,0,1]], dtype=tf.float32)

inverse_cam_transforme = (tf.constant([[1,0,-x0],[0,1,-y0],[0,0,cam_intr_f]],dtype = tf.float32) / cam_intr_f)

inverse_cam_transforme_unscaled = tf.constant([[1,0,-x0],[0,1,-y0],[0,0,cam_intr_f]],dtype = tf.float32)

#Path to per image dataset location
IMG_PATH = "./JTA-Dataset/images"
ANNO_PATH = "./JTA-Dataset/new_image_annotations"

import ShAReD_Net.data.load.dataset_jta.joint as joint
import ShAReD_Net.data.load.dataset_jta.pose as pose
#TODO just a fix, can be removed if dataset is converted using the library methodes
sys.modules['joint'] = joint
sys.modules['pose'] = pose

def create_dataset(data_split):
    img_root = pathlib.Path(IMG_PATH)
    anno_root = tf.constant(ANNO_PATH)
    return create_img_poses_dataset(img_root, anno_root, data_split)

def create_img_poses_dataset(img_root, anno_root, data_split):
    file_ds = create_file_dataset(img_root, data_split)
    img_poses_ds = file_ds.map(img_path_to_img_and_poses(anno_root),
                         num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return img_poses_ds


#TODO only hardcoded train dataset
def create_file_dataset(img_root, data_split):
    file_ds = tf.data.Dataset.list_files(str(img_root/f'{data_split}/*/*'))
    return file_ds



def img_path_to_annotation_path(img_path, anno_root):
    # convert the path to a list of path components
    parts = tf.strings.split(img_path, os.path.sep)
    data_split = parts[-3]
    seq = parts[-2]
    img_name = parts[-1]

    anno_name = tf.strings.split(img_name, '.')[0] + '.npy'
    anno_path = tf.strings.join([anno_root, data_split,seq,anno_name], separator='/')
    # The second to last is the class-directory
    return anno_path

def pose_path_to_poses(pose_path):
    raw_poses = np.load(pose_path, allow_pickle=True)
    poses = np.empty([len(raw_poses),15,3], dtype=np.float32)
    for i, pose in enumerate(raw_poses):
        poses[i,0,0] = pose[1].x3d
        poses[i,0,1] = pose[1].y3d
        poses[i,0,2] = pose[1].z3d

        poses[i,1,0] = pose[2].x3d
        poses[i,1,1] = pose[2].y3d
        poses[i,1,2] = pose[2].z3d

        poses[i,2,0] = pose[15].x3d
        poses[i,2,1] = pose[15].y3d
        poses[i,2,2] = pose[15].z3d

        poses[i,3,0] = pose[4].x3d
        poses[i,3,1] = pose[4].y3d
        poses[i,3,2] = pose[4].z3d

        poses[i,4,0] = pose[8].x3d
        poses[i,4,1] = pose[8].y3d
        poses[i,4,2] = pose[8].z3d

        poses[i,5,0] = pose[5].x3d
        poses[i,5,1] = pose[5].y3d
        poses[i,5,2] = pose[5].z3d

        poses[i,6,0] = pose[9].x3d
        poses[i,6,1] = pose[9].y3d
        poses[i,6,2] = pose[9].z3d

        poses[i,7,0] = pose[6].x3d
        poses[i,7,1] = pose[6].y3d
        poses[i,7,2] = pose[6].z3d

        poses[i,8,0] = pose[10].x3d
        poses[i,8,1] = pose[10].y3d
        poses[i,8,2] = pose[10].z3d

        poses[i,9,0] = pose[16].x3d
        poses[i,9,1] = pose[16].y3d
        poses[i,9,2] = pose[16].z3d

        poses[i,10,0] = pose[19].x3d
        poses[i,10,1] = pose[19].y3d
        poses[i,10,2] = pose[19].z3d

        poses[i,11,0] = pose[17].x3d
        poses[i,11,1] = pose[17].y3d
        poses[i,11,2] = pose[17].z3d

        poses[i,12,0] = pose[20].x3d
        poses[i,12,1] = pose[20].y3d
        poses[i,12,2] = pose[20].z3d

        poses[i,13,0] = pose[18].x3d
        poses[i,13,1] = pose[18].y3d
        poses[i,13,2] = pose[18].z3d

        poses[i,14,0] = pose[21].x3d
        poses[i,14,1] = pose[21].y3d
        poses[i,14,2] = pose[21].z3d

    return poses

def decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_png(img, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img

def img_path_to_img_and_poses(anno_root):
    def to_img_and_poses(file_path):
        anno_path = img_path_to_annotation_path(file_path, anno_root)
        poses = tf.numpy_function(pose_path_to_poses, [anno_path], tf.float32)
        poses.set_shape([None,15,3])

        # load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        img = decode_img(img)
        return img, poses
    return to_img_and_poses
