import tensorflow as tf

from ShAReD_Net.configure import config

# Model dataset params
cut_dist = 8
cut_delta = config.cut_delta
upscaling = config.model.data.upscaling
original_pose_size = 80
roi_size = 20

img_downscaling = original_pose_size / roi_size
cam_transform = config.dataset.cam_transform
cam_intr_f = cam_transform[0,0]
inverse_cam_transforme = config.dataset.inverse_cam_transforme

##INVERSE roi to relative
def roi_poses_to_rel(img_downscaling):
    def scale_to_rel(poses):
        rel_poses = poses * img_downscaling
        return rel_poses
    return scale_to_rel

##INVERSE relativ to abs
def relativ_poss_and_poses_abs(poss, poses):
    abs_poses = poses + poss[:,None,:]
    return abs_poses

##INVERSE cut to img
def cut_poss_to_img(poss, dist, upscaling, cam_intr_f):
    poss_cut = poss * cam_intr_f / upscaling / dist

    return tf.concat([poss_cut[:,:-1],poss[:,None,-1]], axis=-1)

def cut_poses_to_img(poses, dist, upscaling, cam_intr_f):
    poses_cut = poses * cam_intr_f / upscaling / dist

    return tf.concat([poses_cut[:,:,:-1],poses[:,:,None,-1]], axis=-1)

#TEST HELPER CODE
def cut_poss_and_poses_to_img(cut_dist, upscaling, cam_intr_f):
    def to_img(img, poses, poss):
        poss_img = cut_poss_to_img(poss, cut_dist, upscaling, cam_intr_f)
        poses_img = cut_poses_to_img(poses, cut_dist, upscaling, cam_intr_f)
        tf.print("uncut poss, poses",poss_img, poses_img)
        return img, poses_img, poss_img
    return to_img

def create_uncut_dataset(imgcut_cutposes_cutposs_ds):
    uncut_ds = imgcut_cutposes_cutposs_ds.map(cut_poss_and_poses_to_img(cut_dist, upscaling, cam_intr_f))
    return uncut_ds

##INVERSE img to real

def poss_to_real(poss, inverse_cam_transforme):
    poss_homog = poss * poss[:,None,-1]
    poss_homog = tf.concat([poss_homog[:,:-1],poss[:,None,-1]], axis=-1)
    poss_real = tf.linalg.matvec(inverse_cam_transforme, poss_homog)
    return tf.concat([poss_real[:,:-1],poss[:,None,-1]], axis=-1)

def poses_to_real(poses, inverse_cam_transforme):
    poses_homog = poses * poses[:,:,None,-1]
    poses_homog = tf.concat([poses_homog[:,:,:-1],poses[:,:,None,-1]], axis=-1)
    poses_real = tf.linalg.matvec(inverse_cam_transforme, poses_homog)
    return tf.concat([poses_real[:,:,:-1],poses[:,:,None,-1]], axis=-1)

#TEST HELPER CODE

def img_poss_and_poses_to_real(inverse_cam_transforme):
    def to_real(img, poses, poss):
        poss_real = poss_to_real(poss, inverse_cam_transforme)
        poses_real = poses_to_real(poses, inverse_cam_transforme)
        tf.print("real poss, poses",poss_real, poses_real)
        return img, poses_real, poss_real
    return to_real

def create_real_dataset(imgcut_imgposes_imgposs_ds):
    real_ds = imgcut_imgposes_imgposs_ds.map(img_poss_and_poses_to_real(inverse_cam_transforme))
    return real_ds