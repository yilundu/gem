import imageio
import collections
import skimage
import cv2
from glob import glob
import os
import numpy as np
import torch
import random


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def lin2video(video_flat, num_frames, sidelength):
    batch_size = video_flat.shape[0]
    return video_flat.reshape(batch_size, num_frames, sidelength, sidelength, -1)


def pick(list, item_idcs):
    if not list:
        return list
    return [list[i] for i in item_idcs]


def get_mgrid(sidelen, dim=2, subsample=1):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.'''
    # if isinstance(sidelen, int):
    #     sidelen = dim * (sidelen,)

    if dim == 1:
        # pixel_coords = np.stack(np.mgrid[:sidelen[0]], axis=-1)[None, ...].astype(np.float32).T
        # pixel_coords[..., 0] = pixel_coords[..., 0] / (sidelen[0] - 1)
        #from: https://colab.research.google.com/github/vsitzmann/siren/blob/master/explore_siren.ipynb#scrollTo=V0Py4OsOaqgI
        tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
        mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
        mgrid = mgrid.reshape(-1, dim)
        return mgrid
    elif dim == 2:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[0, :, :, 0] = pixel_coords[0, :, :, 0] / (sidelen[0] - 1)
        pixel_coords[0, :, :, 1] = pixel_coords[0, :, :, 1] / (sidelen[1] - 1)
    elif dim == 3:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1], :sidelen[2]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[..., 0] = pixel_coords[..., 0] / max(sidelen[0] - 1, 1)
        pixel_coords[..., 1] = pixel_coords[..., 1] / (sidelen[1] - 1)
        pixel_coords[..., 2] = pixel_coords[..., 2] / (sidelen[2] - 1)
    else:
        raise NotImplementedError('Not implemented for dim=%d' % dim)

    pixel_coords = pixel_coords[:, ::subsample, ::subsample]
    pixel_coords -= 0.5
    pixel_coords *= 2.
    pixel_coords = torch.Tensor(pixel_coords).reshape(-1, dim)
    return pixel_coords


def load_rgb(path, sidelength=None):
    img = imageio.imread(path)[:, :, :3]
    img = skimage.img_as_float32(img)

    img = square_crop_img(img)

    if sidelength is not None:
        img = cv2.resize(img, (sidelength, sidelength), interpolation=cv2.INTER_AREA)

    img -= 0.5
    img *= 2.
    img = img.transpose(2, 0, 1)
    return img


def permute_final_two_axes(tensor):
    permute_tuple = tuple(range(tensor.dim()-2)) + (-1, -2)
    return tensor.permute(permute_tuple)


def lin2img(tensor, image_resolution=None):
    batch_size, num_samples, channels = tensor.shape
    if image_resolution is None:
        width = np.sqrt(num_samples).astype(int)
        height = width
    else:
        height = image_resolution[0]
        width = image_resolution[1]

    return tensor.permute(0, 2, 1).view(batch_size, channels, height, width)


def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def glob_imgs(path):
    imgs = []
    for ext in ['*.png', '*.jpg', '*.JPEG', '*.JPG']:
        imgs.extend(glob(os.path.join(path, ext)))
    return imgs


def parse_intrinsics(filepath, trgt_sidelength=None, invert_y=False):
    # Get camera intrinsics
    with open(filepath, 'r') as file:
        f, cx, cy, _ = map(float, file.readline().split())
        grid_barycenter = torch.Tensor(list(map(float, file.readline().split())))
        scale = float(file.readline())
        height, width = map(float, file.readline().split())

        try:
            world2cam_poses = int(file.readline())
        except ValueError:
            world2cam_poses = None

    if world2cam_poses is None:
        world2cam_poses = False

    world2cam_poses = bool(world2cam_poses)

    if trgt_sidelength is not None:
        cx = cx/width * trgt_sidelength
        cy = cy/height * trgt_sidelength
        f = trgt_sidelength / height * f

    fx = f
    if invert_y:
        fy = -f
    else:
        fy = f

    # Build the intrinsic matrices
    full_intrinsic = np.array([[fx, 0., cx, 0.],
                               [0., fy, cy, 0],
                               [0., 0, 1, 0],
                               [0, 0, 0, 1]])

    return full_intrinsic, grid_barycenter, scale, world2cam_poses


def load_pose(filename):
    lines = open(filename).read().splitlines()
    if len(lines) == 1:
        pose = np.zeros((4, 4), dtype=np.float32)
        for i in range(16):
            pose[i // 4, i % 4] = lines[0].split(" ")[i]
        return pose.squeeze()
    else:
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines[:4])]
        return np.asarray(lines).astype(np.float32).squeeze()


def square_crop_img(img):
    min_dim = np.amin(img.shape[:2])
    center_coord = np.array(img.shape[:2]) // 2
    img = img[center_coord[0] - min_dim // 2:center_coord[0] + min_dim // 2,
          center_coord[1] - min_dim // 2:center_coord[1] + min_dim // 2]
    return img


def convert_int2color(imgs,color_mapping):
    # convert int to a color
    # color_mapping = {int: [r,g,b],...}

    colored_imgs = []

    imgs = imgs.cpu().numpy()

    for img in imgs:
        r = np.zeros_like(img).astype(np.uint8)
        g = np.zeros_like(img).astype(np.uint8)
        b = np.zeros_like(img).astype(np.uint8)

        for class_idx, [r_val, g_val, b_val] in color_mapping.items():

            r[img == class_idx] = r_val
            g[img == class_idx] = g_val
            b[img == class_idx] = b_val

        rgb = np.stack([r.squeeze(), g.squeeze(), b.squeeze()], axis=0)
        colored_imgs.append(rgb)

    return torch.from_numpy(np.array(colored_imgs)).cuda()



def dict_to_gpu(ob):
    if isinstance(ob, collections.Mapping):
        return {k: dict_to_gpu(v) for k, v in ob.items()}
    else:
        return ob.cuda()

def stitch_audio_subset(audio_samples, num_sample=10, elmts=None):
    # concatenate arrays of audio samples
    # audio samples = [B, 1, T] where B = batch size, T = len of sample
    # only sample k from batch of samples
    # if elmts is not None, use the pre-sampled samples in batch
    # returns concatenated signals and sampled set of indices

    if num_sample == -1: # use all
        elmts = list(range(len(audio_samples)))
    elif elmts is None:
        # sample indices
        batch_size = audio_samples.shape[0]
        if num_sample > batch_size: num_sample = batch_size
        elmts = random.sample(range(batch_size), num_sample)

    audio_subset = []
    for elmt in elmts:
        audio_subset.append(audio_samples[elmt])

    combo_signal = torch.cat(audio_subset, dim=-1)
    return combo_signal, elmts


def process_tf_record_data(data_path):
    # reads in TFRecord data
    # returns list of data features (dicts)
    import tensorflow as tf # adding this here to not break other envs

    # help from: https://www.tensorflow.org/tutorials/load_data/tfrecord
    import pdb
    pdb.set_trace()
    raw_dataset = [raw_record for raw_record in tf.data.TFRecordDataset(data_path)]#.take(1)]
    data_features = []
    for raw_record in raw_dataset:
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        data_features.append(example.features.feature)

    return data_features

def convert_byte2str(str_data):
    # extract string data from compressed rep
    return [x.decode('utf-8') for x in str_data.bytes_list.value][0]
