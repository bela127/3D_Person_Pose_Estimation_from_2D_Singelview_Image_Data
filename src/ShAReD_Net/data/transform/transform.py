import tensorflow as tf
import numpy as np

from ShAReD_Net.configure import config
import ShAReD_Net.model.modules.slim as slim_modules

# Model dataset params
cut_steps = config.model.data.cut_steps
cut_min_dist = config.model.data.cut_min_dist
cut_delta = config.model.data.cut_delta
upscaling = config.model.data.upscaling
downsample_pose = config.model.img_downsampling
downsample_heatmap = config.model.roi_downsampling
roi_size = np.asarray(config.model.roi_size)


cam_transform = config.dataset.cam_transform
cam_intr_f = cam_transform[0,0]



## DATASET
def create_data_at_cutdist_dataset(data_split, cut_dist):
    img_poses_ds = config.dataset.create_dataset(data_split)
    img_poses_poss_ds = create_img_poses_poss_dataset(img_poses_ds)
    imgcut_poses_poss_ds = create_imgcut_poses_poss_dataset(img_poses_poss_ds, cut_dist)
    imgcut_imgposes_imgposs_ds = create_imgcut_imgposes_imgposs_dataset(imgcut_poses_poss_ds)
    imgcut_cutposes_cutposs_ds = create_imgcut_cutposes_cutposs_dataset(imgcut_imgposes_imgposs_ds, cut_dist)
    crop_ds = create_croped_dataset(imgcut_cutposes_cutposs_ds)
    filtered_cutposes_cutposs_ds = create_filtered_cutposes_cutposs_dataset(crop_ds)
    imgcut_cutposes_cutposs_heatmap_indices_ds = create_imgcut_cutposes_cutposs_heatmap_indices_dataset(filtered_cutposes_cutposs_ds, cut_dist)
    imgcut_cutposes_cutposs_heatmap_indices_weights_ds = create_imgcut_cutposes_cutposs_heatmap_indices_weights_dataset(imgcut_cutposes_cutposs_heatmap_indices_ds)
    imgcut_relposes_roiindices_heatmap_indices_weights_ds = create_imgcut_relposes_roiindices_heatmap_indices_weights_dataset(imgcut_cutposes_cutposs_heatmap_indices_weights_ds, cut_dist) 
    imgcut_relposes_roiindices_heatmap_indices_weights_poseindexes_ds = create_imgcut_relposes_roiindices_heatmap_indices_weights_poseindexes_dataset(imgcut_relposes_roiindices_heatmap_indices_weights_ds)
    
    batchable_ds = create_batchable_dataset(imgcut_relposes_roiindices_heatmap_indices_weights_poseindexes_ds)
    return batchable_ds

#----------------------------------------------------------

def img_poses_add_poss(image, poses):
    poss = poses_to_poss(poses)
    return image, poses, poss

def poses_to_poss(poses):
    poss = tf.reduce_mean(poses, axis=-2)
    return poss

## DATASET
def create_img_poses_poss_dataset(img_poses_ds):
    img_poses_poss_ds = img_poses_ds.map(img_poses_add_poss)
    return img_poses_poss_ds

#----------------------------------------------------------

def img_poses_poss_convert_img_to_imgcut(cut_dist, cut_delta, upscaling, cam_intr_f):
    def frustum_image(image, poses, poss):
        scaled_img = scale_img_to_pos(image, cut_dist, upscaling, cam_intr_f)
        filtered_poss, filtered_poses = filter_poss_and_pose(poss, poses, cut_dist, cut_delta)
        return scaled_img, filtered_poses, filtered_poss
    return frustum_image

def scale_img_to_pos(image, dist, upscaling, cam_intr_f):
    image_size = tf.shape(image)[:-1]
    new_size = tf.cast(image_size, dtype=tf.float32) * upscaling * dist / cam_intr_f
    sized_img = tf.image.resize_with_pad(image, tf.cast(new_size[0], dtype=tf.int32), tf.cast(new_size[1], dtype=tf.int32), antialias=True)

    return sized_img

def filter_poss_and_pose(poss, poses, dist, dist_delta):
    indexes = tf.where(tf.abs(poss[:,-1]-dist) < dist_delta)
    filtered_poss = tf.gather_nd(poss, indexes)
    filtered_poses = tf.gather_nd(poses, indexes)
    
    for k in range(15):
        indexes = tf.where(tf.abs(filtered_poses[:,k,-1]-dist) < dist_delta)
        
        filtered_poss = tf.gather_nd(filtered_poss, indexes)
        filtered_poses = tf.gather_nd(filtered_poses, indexes)

    return filtered_poss, filtered_poses

## DATASET
def create_imgcut_poses_poss_dataset(img_poses_poss_ds, cut_dist):
    imgcut_poses_poss_ds = img_poses_poss_ds.map(img_poses_poss_convert_img_to_imgcut(cut_dist, cut_delta, upscaling, cam_intr_f))
    return imgcut_poses_poss_ds

#----------------------------------------------------------

#@tf.function
def imgcut_poses_poss_convert_poses_poss_to_imgposes_imgposs(cam_transform):
    def to_img(img, poses, poss):
        poss_img = poss_to_img(poss, cam_transform)
        poses_img = poses_to_img(poses, cam_transform)
        return img, poses_img, poss_img
    return to_img

def poss_to_img(poss, cam_transform):
    poss_homog = tf.linalg.matvec(cam_transform, poss)
    poss_img = poss_homog / poss_homog[:,None,-1]
    return tf.concat([poss_img[:,:-1],poss[:,None,-1]], axis=-1)

def poses_to_img(poses, cam_transform):
    poses_homog = tf.linalg.matvec(cam_transform, poses)
    poses_img = poses_homog / poses_homog[:,:,None,-1]
    return tf.concat([poses_img[:,:,:-1],poses[:,:,None,-1]], axis=-1)

## DATASET
def create_imgcut_imgposes_imgposs_dataset(imgcut_poses_poss_ds):
    imgcut_imgposes_imgposs_ds = imgcut_poses_poss_ds.map(imgcut_poses_poss_convert_poses_poss_to_imgposes_imgposs(cam_transform))
    return imgcut_imgposes_imgposs_ds

#----------------------------------------------------------

def imgcut_imgposes_imgposs_convert_imgposes_imgposs_to_cutposes_cutposs(cut_dist, upscaling, cam_intr_f):
    def to_cut(img, poses, poss):
        poss_cut = poss_to_cut(poss, cut_dist, upscaling, cam_intr_f)
        poses_cut = poses_to_cut(poses, cut_dist, upscaling, cam_intr_f)
        poss_cut_filtered, poses_cut_filtered = filter_poss_and_pose_img_size(poss_cut, poses_cut, tf.shape(img)[:-1])
        return img, poses_cut_filtered, poss_cut_filtered
    return to_cut

def poss_to_cut(poss, dist, upscaling, cam_intr_f):
    poss_cut = poss * upscaling * dist / cam_intr_f
    return tf.concat([poss_cut[:,:-1],poss[:,None,-1]], axis=-1)

def poses_to_cut(poses, dist, upscaling, cam_intr_f):
    poses_cut = poses * upscaling * dist / cam_intr_f
    return tf.concat([poses_cut[:,:,:-1],poses[:,:,None,-1]], axis=-1)

def filter_poss_and_pose_img_size(poss, poses, image_size):
    image_size = tf.cast(image_size, poss.dtype)

    indexes = tf.where(tf.math.logical_and(
        tf.math.logical_and(poss[:,0] >= 0, poss[:,0] < image_size[1] - 1),
        tf.math.logical_and(poss[:,1] >= 0, poss[:,1] < image_size[0] - 1)))
    filtered_poss = tf.gather_nd(poss, indexes)
    filtered_poses = tf.gather_nd(poses, indexes)

    # for each keypoint filter person not fully in image
    for k in range(15):
        indexes = tf.where(tf.math.logical_and(
            tf.math.logical_and(filtered_poses[:,k,0] >= 0, filtered_poses[:,k,0] < image_size[1] - 1),
            tf.math.logical_and(filtered_poses[:,k,1] >= 0, filtered_poses[:,k,1] < image_size[0] - 1)))
        
        filtered_poss = tf.gather_nd(filtered_poss, indexes)
        filtered_poses = tf.gather_nd(filtered_poses, indexes)

    return filtered_poss, filtered_poses


## DATASET
def create_imgcut_cutposes_cutposs_dataset(imgcut_imgposes_imgposs_ds, cut_dist):
    imgcut_cutposes_cutposs_ds = imgcut_imgposes_imgposs_ds.map(imgcut_imgposes_imgposs_convert_imgposes_imgposs_to_cutposes_cutposs(cut_dist, upscaling, cam_intr_f))
    return imgcut_cutposes_cutposs_ds

#----------------------------------------------------------


image_crop_size = np.asarray(roi_size) * downsample_pose * 3 + 1
roi_extractor = slim_modules.Roi_Extractor(roi_size=image_crop_size)

def random_crop(imgcut, poses_cut, poss_cut):
    imgcut_batched = imgcut[None,...]
    offset_size = tf.cast(min(image_crop_size), tf.float32) / 3
    offsets = tf.random.uniform(tf.shape(poss_cut), minval=-offset_size, maxval=offset_size, dtype=tf.dtypes.float32)

    cut_index = tf.cast(poss_cut + offsets + 0.5, tf.int32)[...,::-1]

    mask = tf.constant([[0, 1, 1]], tf.int32)

    cut_index = cut_index * mask

    pose_imges = roi_extractor([imgcut_batched, cut_index])

    poses = tf.repeat(poses_cut[None,...], tf.shape(cut_index)[0], axis=0)
    poss = tf.repeat(poss_cut[None,...], tf.shape(cut_index)[0], axis=0)

    center = tf.constant([[image_crop_size[1]//2, image_crop_size[0]//2, 0]], tf.float32)
    cut_pos = tf.cast(cut_index[...,::-1], tf.float32) - center
    cut_pos_poses = cut_pos[:,None,None,:]
    cut_pos_poss = cut_pos[:,None,:]

    new_poses = poses - cut_pos_poses
    new_poss = poss - cut_pos_poss


    dataset = tf.data.Dataset.from_tensor_slices((pose_imges, new_poses, new_poss))

    return dataset

def filter_poss_and_pose_crop_size(image, poses, poss):
    image_size = tf.cast(tf.shape(image)[0:2], poss.dtype)
    
    indexes = tf.where(tf.math.logical_and(
        tf.math.logical_and(poss[:,0] >= 0, poss[:,0] < image_size[1] - 1),
        tf.math.logical_and(poss[:,1] >= 0, poss[:,1] < image_size[0] - 1)))
    filtered_poss = tf.gather_nd(poss, indexes)
    filtered_poses = tf.gather_nd(poses, indexes)

    # for each keypoint filter person not fully in image
    for k in range(15):
        indexes = tf.where(tf.math.logical_and(
            tf.math.logical_and(filtered_poses[:,k,0] >= 0, filtered_poses[:,k,0] < image_size[1] - 1),
            tf.math.logical_and(filtered_poses[:,k,1] >= 0, filtered_poses[:,k,1] < image_size[0] - 1)))
        
        filtered_poss = tf.gather_nd(filtered_poss, indexes)
        filtered_poses = tf.gather_nd(filtered_poses, indexes)

    return image, filtered_poses, filtered_poss 

## DATASET
def create_croped_dataset(dataset):
    croped_ds = dataset.flat_map(random_crop)
    filtered_ds = croped_ds.map(filter_poss_and_pose_crop_size)
    return filtered_ds

#----------------------------------------------------------


def filter_cutposes_cutposs(img, cutposes ,cutposs):
    length = tf.shape(cutposs)[0]
    if length == 0:
        return False
    else:
        return True

## DATASET
def create_filtered_cutposes_cutposs_dataset(imgcut_cutposes_cutposs_ds):
    filtered_ds = imgcut_cutposes_cutposs_ds.filter(filter_cutposes_cutposs)
    return filtered_ds

#----------------------------------------------------------

# VERSION with TWO channels with probapility per direction
def imgcut_cutposes_cutposs_add_heatmap_indices(cut_dist, cut_delta, downsample_heatmap):
    def pos_heatmap(img, poses, poss):
        img_size = tf.shape(img)[:-1]
        hm_size = tf.cast(img_size // downsample_heatmap, dtype=tf.int64)
        
        map_indices = tf.cast(poss[:,:-1] / downsample_heatmap + 0.5, dtype = tf.int64)[:,::-1]
        map_indices = tf.maximum(tf.minimum(map_indices, hm_size - 1), 0)
        
        dist = poss[:,-1]
        values = (dist - cut_dist) / cut_delta
        pos_indices = tf.where(tf.math.logical_and(values >= 0,values < 1))
        neg_indices = tf.where(tf.math.logical_and(values <= 0,values > -1))
        pos_val = 1 - tf.gather_nd(values, pos_indices)
        neg_val = 1 + tf.gather_nd(values, neg_indices)
        pos_map_indices = tf.gather_nd(map_indices, pos_indices)
        neg_map_indices = tf.gather_nd(map_indices, neg_indices)
        
        pos_sparse_hm = tf.SparseTensor(pos_map_indices, pos_val, hm_size)
        pos_heatmap = tf.expand_dims(tf.sparse.to_dense(pos_sparse_hm, validate_indices=False),axis=-1)
        neg_sparse_hm = tf.SparseTensor(neg_map_indices, neg_val, hm_size)
        neg_heatmap = tf.expand_dims(tf.sparse.to_dense(neg_sparse_hm, validate_indices=False),axis=-1)

        heatmap = tf.concat([neg_heatmap,pos_heatmap],axis=-1)
        return img, poses, poss, heatmap, map_indices
    return pos_heatmap


## DATASET
def create_imgcut_cutposes_cutposs_heatmap_indices_dataset(imgcut_cutposes_cutposs_ds, cut_dist):
    imgcut_cutposes_cutposs_heatmap_indices_ds = imgcut_cutposes_cutposs_ds.map(imgcut_cutposes_cutposs_add_heatmap_indices(cut_dist, cut_delta, downsample_heatmap))
    return imgcut_cutposes_cutposs_heatmap_indices_ds

#----------------------------------------------------------

def imgcut_cutposes_cutposs_heatmap_indices_add_weights(img, poses, poss, heatmap, indices):
    weights = heatmap_to_weights(heatmap)
    return img, poses, poss, heatmap, indices, weights

def heatmap_to_weights(heatmap):
    bin_heatmap = tf.math.ceil(heatmap)

    heatmap_batch = tf.expand_dims(bin_heatmap,axis=0)
    heatmap_3d = tf.expand_dims(heatmap_batch,axis=-1)
    
    heatmap_3d_cast = tf.cast(heatmap_3d, dtype=tf.float32) #FIX Tf Maxpool does not support Float64
    dilated_heatmap_3d_cast = tf.nn.max_pool3d(heatmap_3d_cast, ksize=3, strides=1, padding="SAME", name='dilation')
    dilated_heatmap_3d = tf.cast(dilated_heatmap_3d_cast, dtype=heatmap.dtype)
    
    dilated_heatmap = dilated_heatmap_3d[0,...,0]

    heatmap_shape = tf.shape(heatmap)
    
    nr_persons = tf.reduce_sum(bin_heatmap)
    nr_non_zeros = tf.reduce_sum(dilated_heatmap)
    nr_positions = tf.cast(tf.reduce_prod(heatmap_shape),dtype=heatmap.dtype)
    scale_negative = nr_non_zeros / nr_positions
    scale_positive = 1 - nr_persons / nr_positions
    scale_negative = tf.reshape(scale_negative,[1,1,1])
    scale_positive = tf.reshape(scale_positive,[1,1,1])

    ones = tf.ones(heatmap_shape, dtype=heatmap.dtype)
    negative_weights = (ones - dilated_heatmap) * scale_negative
    positive_weights = bin_heatmap * scale_positive
    weights = negative_weights + positive_weights
    return weights

## DATASET
def create_imgcut_cutposes_cutposs_heatmap_indices_weights_dataset(imgcut_cutposes_cutposs_heatmap_indices_ds):
    imgcut_cutposes_cutposs_heatmap_indices_weights_ds = imgcut_cutposes_cutposs_heatmap_indices_ds.map(imgcut_cutposes_cutposs_heatmap_indices_add_weights)
    return imgcut_cutposes_cutposs_heatmap_indices_weights_ds

#----------------------------------------------------------

def imgcut_cutposes_cutposs_heatmap_indices_weights_convert_cutposes_cutpos_to_relposes_roiindices(cut_dist, downsample_pose, downsample_heatmap):
    def convert_cutposes_cutpos_to_relposes_roiindices(img, poses, poss, heatmap, indices, weights):
        rel_poses, roi_indices = poses_indices_to_relposes_roiindices(poses, indices, cut_dist, downsample_pose, downsample_heatmap)
        return img, rel_poses, roi_indices, heatmap, indices, weights 
    return convert_cutposes_cutpos_to_relposes_roiindices

def poses_indices_to_relposes_roiindices(poses, indices, cut_dist, downsample_pose, downsample_heatmap):
    abs_poss = tf.cast(indices * downsample_heatmap, dtype=tf.float32)[:,::-1]
    roi_indices = tf.cast(tf.cast(indices * downsample_heatmap, dtype=tf.float32) / downsample_pose + 0.5 , dtype=tf.int32)
    rel_poses_xy = poses[:,:,:-1] - abs_poss[:,None,:]
    rel_poses_z = poses[:,:,-1,None] - cut_dist
    rel_poses = tf.concat([rel_poses_xy, rel_poses_z], axis = -1)
    return rel_poses, roi_indices

## DATASET
def create_imgcut_relposes_roiindices_heatmap_indices_weights_dataset(imgcut_cutposes_cutposs_heatmap_indices_weights_ds, cut_dist):
    imgcut_relposes_roiindices_heatmap_indices_weights_ds = imgcut_cutposes_cutposs_heatmap_indices_weights_ds.map(imgcut_cutposes_cutposs_heatmap_indices_weights_convert_cutposes_cutpos_to_relposes_roiindices(cut_dist, downsample_pose, downsample_heatmap))
    return imgcut_relposes_roiindices_heatmap_indices_weights_ds

#----------------------------------------------------------

def imgcut_relposes_roiindices_heatmap_indices_weights_add_poseindexes(downsample_pose, roi_size):
    def add_poseindexes(img, rel_poses, roi_indices, heatmap, indices, weights):
        pose_indexes = relposes_to_poseindexes(rel_poses, downsample_pose, roi_size)
        return img, rel_poses, roi_indices, heatmap, indices, weights, pose_indexes
    return add_poseindexes

def relposes_to_poseindexes(rel_poses, downsample_pose, roi_size):
    pose_indexes = tf.cast(rel_poses[...,:-1] / downsample_pose + roi_size/2 + 0.5 , dtype=tf.int32)
    pose_indexes = tf.maximum(pose_indexes, 0)
    pose_indexes = tf.minimum(pose_indexes, roi_size)
    return pose_indexes[...,::-1]

## DATASET
def create_imgcut_relposes_roiindices_heatmap_indices_weights_poseindexes_dataset(imgcut_relposes_roiindices_heatmap_indices_weights_ds):
    imgcut_relposes_roiindices_heatmap_indices_weights_poseindexes_ds = imgcut_relposes_roiindices_heatmap_indices_weights_ds.map(imgcut_relposes_roiindices_heatmap_indices_weights_add_poseindexes(downsample_pose, roi_size))
    return imgcut_relposes_roiindices_heatmap_indices_weights_poseindexes_ds

#----------------------------------------------------------


def make_ragged_annotation(anno):
    expanded = tf.expand_dims(anno, 0)
    return tf.RaggedTensor.from_tensor(expanded)

def prepare_batching(img, rel_poses, roi_indices, heatmap, indices, weights, pose_indexes):
    roi_indexes = make_ragged_annotation(roi_indices)
    rel_pose = make_ragged_annotation(rel_poses)
    pose_indexes = make_ragged_annotation(pose_indexes)
    return img, (heatmap, weights), roi_indexes, (rel_pose, pose_indexes)


def squeeze_annotation(anno):
    squeezed_anno = tf.squeeze(anno, axis=1)
    return squeezed_anno

def squeeze_batch(img_b, poss_b, poses_b, pos_b, pose_b):
    poss_b = squeeze_annotation(poss_b)
    poses_b = squeeze_annotation(poses_b)
    return img_b, poses_b, poss_b, pos_b, pose_b

## DATASET
def create_batchable_dataset(dataset):
    batchable_ds = dataset.map(prepare_batching)
    return batchable_ds

#----------------------------------------------------------

## DATASET
def create_cutdist_dataset():
    cut_max_dist = cut_min_dist + cut_steps * cut_delta
    cutdist_ds = tf.data.Dataset.from_tensor_slices(tf.range(cut_min_dist, cut_max_dist, cut_delta, dtype=tf.float32))
    return cutdist_ds

#----------------------------------------------------------

def interleav_dataset(data_split):
    def interleav(cut_dist):
        return create_data_at_cutdist_dataset(data_split, cut_dist)
    return interleav
    

## DATASET
def create_dataset(data_split, batch_size):
    cutdist_ds = create_cutdist_dataset()
    interleaved_multiscale_ds = cutdist_ds.interleave(interleav_dataset(data_split), block_length=batch_size)
    return interleaved_multiscale_ds

#----------------------------------------------------------

