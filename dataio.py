import h5py
import config
import cv2
import imageio
from imageio import imread, imwrite
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from skimage.transform import resize
import random
import pickle as pck
from glob import glob
import torchaudio
from torchaudio.transforms import MelSpectrogram, Spectrogram
import pandas as pd
import utils
import pickle
import io
from copy import deepcopy
from torchvision.datasets.utils import download_file_from_google_drive
from tqdm.autonotebook import tqdm
from torchmeta.datasets.utils import get_asset
import pandas as pd
import os.path as osp

import torch.utils.data as data
import os
import os.path
import errno
import h5py
import json

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Resize, Compose, ToTensor, Normalize, CenterCrop,ToPILImage
import torchvision

from sklearn.model_selection import train_test_split


class MovingMNIST(data.Dataset):
    """`MovingMNIST <http://www.cs.toronto.edu/~nitish/unsupervised_video/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        split (int, optional): Train/test split size. Number defines how many samples
            belong to test set. 
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in an PIL
            image and returns a transformed version. E.g, ``transforms.RandomCrop``
    """
    urls = [
        'https://github.com/tychovdo/MovingMNIST/raw/master/mnist_test_seq.npy.gz'
    ]
    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'moving_mnist_train.pt'
    test_file = 'moving_mnist_test.pt'

    def __init__(self, root, sidelength, train=True, split=1000, download=False):
        self.root = os.path.expanduser(root)
        self.split = split
        self.train = train  # training set or test set

        self.transform = Compose([
            Resize((sidelength, sidelength)),
            CenterCrop((sidelength, sidelength)),
            ToTensor(),
            Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
        ])
        self.mgrid = utils.get_mgrid(sidelength, dim=2)

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if self.train:
            self.train_data = torch.load(
                os.path.join(self.root, self.processed_folder, self.training_file))
        else:
            self.test_data = torch.load(
                os.path.join(self.root, self.processed_folder, self.test_file))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (seq, target) where sampled sequences are splitted into a seq
                    and target part
        """
        if self.train:
            seq, target = self.train_data[index, :10], self.train_data[index, 10:]
        else:
            seq, target = self.test_data[index, :10], self.test_data[index, 10:]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        seq = Image.fromarray(seq.numpy(), mode='L')
        # target = Image.fromarray(target.numpy(), mode='L')

        # if self.transform is not None:
        seq = self.transform(seq)
        #
        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        return seq, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.training_file)) and \
            os.path.exists(os.path.join(self.root, self.processed_folder, self.test_file))

    def download(self):
        """Download the Moving MNIST data if it doesn't exist in processed_folder already."""
        from six.moves import urllib
        import gzip

        if self._check_exists():
            return

        # download files
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        for url in self.urls:
            print('Downloading ' + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.root, self.raw_folder, filename)
            with open(file_path, 'wb') as f:
                f.write(data.read())
            with open(file_path.replace('.gz', ''), 'wb') as out_f, \
                    gzip.GzipFile(file_path) as zip_f:
                out_f.write(zip_f.read())
            os.unlink(file_path)

        # process and save as torch files
        print('Processing...')

        training_set = torch.from_numpy(
            np.load(os.path.join(self.root, self.raw_folder, 'mnist_test_seq.npy')).swapaxes(0, 1)[:-self.split]
        )
        test_set = torch.from_numpy(
            np.load(os.path.join(self.root, self.raw_folder, 'mnist_test_seq.npy')).swapaxes(0, 1)[-self.split:]
        )

        with open(os.path.join(self.root, self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.root, self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Train/test: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


class PointCloud(Dataset):
    def __init__(self, pointcloud_path, on_surface_points, keep_aspect_ratio=True):
        super().__init__()

        print("Loading point cloud")
        point_cloud = np.genfromtxt(pointcloud_path)
        print("Finished loading point cloud")

        coords = point_cloud[:, :3]
        self.normals = point_cloud[:, 3:]

        # Reshape point cloud such that it lies in bounding box of (-1, 1) (distorts geometry, but makes for high
        # sample efficiency)
        coords -= np.mean(coords, axis=0, keepdims=True)
        if keep_aspect_ratio:
            coord_max = np.amax(coords)
            coord_min = np.amin(coords)
        else:
            coord_max = np.amax(coords, axis=0, keepdims=True)
            coord_min = np.amin(coords, axis=0, keepdims=True)

        self.coords = (coords - coord_min) / (coord_max - coord_min)
        self.coords -= 0.5
        self.coords *= 2.

        self.on_surface_points = on_surface_points

    def __len__(self):
        return self.coords.shape[0] // self.on_surface_points

    def __getitem__(self, idx):
        point_cloud_size = self.coords.shape[0]

        off_surface_samples = self.on_surface_points  # **2
        total_samples = self.on_surface_points + off_surface_samples

        # Random coords
        rand_idcs = np.random.choice(point_cloud_size, size=self.on_surface_points)

        on_surface_coords = self.coords[rand_idcs, :]
        on_surface_normals = self.normals[rand_idcs, :]

        off_surface_coords = np.random.uniform(-1, 1, size=(off_surface_samples, 3))
        off_surface_normals = np.ones((off_surface_samples, 3)) * -1

        sdf = np.zeros((total_samples, 1))  # on-surface = 0
        sdf[self.on_surface_points:, :] = -1  # off-surface = -1

        coords = np.concatenate((on_surface_coords, off_surface_coords), axis=0)
        normals = np.concatenate((on_surface_normals, off_surface_normals), axis=0)

        return {'coords': torch.from_numpy(coords).float()}, {'sdf': torch.from_numpy(sdf).float(),
                                                              'normals': torch.from_numpy(normals).float()}


class MultiMNIST(torch.utils.data.Dataset):
    filename = '{0}_data.hdf5'
    filename_labels = '{0}_labels.json'

    def __init__(self, split, selected_classes, sidelength,
                 folder, gdrive_id, zip_filename, zip_md5, image_folder):
        self.gdrive_id = gdrive_id
        self.image_folder = image_folder
        self.folder = folder
        self.zip_md5 = zip_md5
        self.zip_filename = zip_filename
        self.sidelength = sidelength
        self.mgrid = utils.get_mgrid(sidelength, dim=2)
        self.channels = 1

        self.transform = Compose([
            Resize((sidelength, sidelength)),
            ToTensor(),
            Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
        ])

        self.root = os.path.join('/om2/user/sitzmann/', self.folder)
        self.split_filename = os.path.join(self.root,
                                           self.filename.format(split))
        self.split_filename_labels = os.path.join(self.root,
                                                  self.filename_labels.format(split))
        self.download()

        self.data_file = h5py.File(self.split_filename, 'r')
        self.data = self.data_file['datasets']

        if selected_classes:
            self.selected_classes = selected_classes
        else:
            with open(self.split_filename_labels, 'r') as f:
                self.selected_classes = json.load(f)

        self.concat_data, self.labels = list(), list()
        for int_class, string_class in enumerate(self.selected_classes):
            sub_dataset = self.data[string_class]
            self.concat_data.append(sub_dataset)
            self.labels.extend([int_class]*len(sub_dataset))

        self.dataset = torch.utils.data.ConcatDataset(self.concat_data)

    def __getitem__(self, index):
        image = Image.open(io.BytesIO(self.dataset[index]))
        image = self.transform(image)
        return {"img": image, "label":self.labels[index]}

    def __len__(self):
        return len(self.labels)

    def _check_integrity(self):
        return (os.path.isfile(self.split_filename)
                and os.path.isfile(self.split_filename_labels))

    def download(self):
        import zipfile
        import shutil
        import glob

        if self._check_integrity():
            return

        zip_filename = os.path.join(self.root, self.zip_filename)
        if not os.path.isfile(zip_filename):
            download_file_from_google_drive(self.gdrive_id, self.root,
                                            self.zip_filename, md5=self.zip_md5)

        zip_foldername = os.path.join(self.root, self.image_folder)
        if not os.path.isdir(zip_foldername):
            with zipfile.ZipFile(zip_filename, 'r') as f:
                for member in tqdm(f.infolist(), desc='Extracting '):
                    try:
                        f.extract(member, self.root)
                    except zipfile.BadZipFile:
                        print('Error: Zip file is corrupted')

        for split in ['train', 'val', 'test']:
            filename = os.path.join(self.root, self.filename.format(split))
            if os.path.isfile(filename):
                continue

            labels = get_asset(self.folder, '{0}.json'.format(split))
            labels_filename = os.path.join(self.root,
                                           self.filename_labels.format(split))
            with open(labels_filename, 'w') as f:
                json.dump(labels, f)

            image_folder = os.path.join(zip_foldername, split)

            with h5py.File(filename, 'w') as f:
                group = f.create_group('datasets')
                dtype = h5py.special_dtype(vlen=np.uint8)
                for i, label in enumerate(tqdm(labels, desc=filename)):
                    images = glob.glob(os.path.join(image_folder, label,
                                                    '*.png'))
                    images.sort()
                    dataset = group.create_dataset(label, (len(images),),
                                                   dtype=dtype)
                    for i, image in enumerate(images):
                        with open(image, 'rb') as f:
                            array = bytearray(f.read())
                            dataset[i] = np.asarray(array, dtype=np.uint8)

        if os.path.isdir(zip_foldername):
            shutil.rmtree(zip_foldername)


class MNIST(torch.utils.data.Dataset):
    def __init__(self, split, sidelength, selected_digits=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]):
        super().__init__()

        assert split in ['train', 'test', 'val'], "Unknown split"

        self.mgrid = utils.get_mgrid([sidelength, sidelength], dim=2)
        self.sidelength = sidelength
        self.channels = 1

        transform = Compose([
            Resize((sidelength, sidelength)),
            ToTensor(),
            Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
        ])

        self.mnist = torchvision.datasets.MNIST('/om2/user/sitzmann/MNIST', train=True if split in ['train', 'val'] else False,
                                                download=True, transform=transform)

        # filter by selected numbers
        idx = [(x in selected_digits) for x in self.mnist.targets]
        self.mnist.targets = self.mnist.targets[idx]
        self.mnist.data = self.mnist.data[idx]

        # Take 10% of training dataset and create a validation dataset
        if split in ['train', 'val']:
            # Split into train and val splits
            torch.manual_seed(0)
            num_train = int(0.9 * len(self.mnist))
            num_val = len(self.mnist) - num_train
            train_dataset, val_dataset = torch.utils.data.random_split(self.mnist, [num_train, num_val])
            self.mnist = train_dataset if split == 'train' else val_dataset

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        return {"img": self.mnist[idx][0], "label": self.mnist[idx][1]}


class DoubleMNIST(MultiMNIST):
    def __init__(self, split, sidelength, selected_classes=None):
        super().__init__(split=split,
                         selected_classes=selected_classes,
                         sidelength=sidelength,
                         folder='doublemnist',
                         gdrive_id='1MqQCdLt9TVE3joAMw4FwJp_B8F-htrAo',
                         zip_filename='double_mnist_seed_123_image_size_64_64.zip',
                         zip_md5='6d8b185c0cde155eb39d0e3615ab4f23',
                         image_folder='double_mnist_seed_123_image_size_64_64')


class TripleMNIST(MultiMNIST):
    def __init__(self, split, sidelength, selected_classes=None):
        super().__init__(split=split,
                         selected_classes=selected_classes,
                         sidelength=sidelength,
                         folder='triplemnist',
                         gdrive_id='1xqyW289seXYaDSqD2jaBPMKVAAjPP9ee',
                         zip_filename='triple_mnist_seed_123_image_size_84_84.zip',
                         zip_md5='9508b047f9fbb834c02bc13ef44245da',
                         image_folder='triple_mnist_seed_123_image_size_84_84')


class GEO(torch.utils.data.Dataset):
    def __init__(self, sidelength, split='train', subset=None):
        self.name = 'geo'
        self.channels = 3
        self.mgrid = utils.get_mgrid(sidelength, dim=2)
        self.sidelength = sidelength

        if split == 'train':
            self.im_1_paths = sorted(glob('/om2/user/sitzmann/mod-shape3-demo/*im1*.png'))
            self.im_2_paths = sorted(glob('/om2/user/sitzmann/mod-shape3-demo/*im2*.png'))
        elif split == 'test':
            self.im_1_paths = sorted(glob('/om2/user/sitzmann/mod-shape3-demo-test/*im1*.png'))
            self.im_2_paths = sorted(glob('/om2/user/sitzmann/mod-shape3-demo-test/*im2*.png'))

    def __len__(self):
        return len(self.im_1_paths)

    def read_frame(self, path):
        frame = torch.from_numpy(imageio.imread(path)).float()
        frame /= 255.
        frame -= 0.5
        frame *= 2.
        frame = frame.permute(2, 0, 1)
        return frame

    def __getitem__(self, item):
        frame_1 = self.read_frame(self.im_1_paths[item])
        frame_2 = self.read_frame(self.im_2_paths[item])
        return {"frame_1":frame_1, 'frame_2':frame_2, "rgb":frame_1}


# class CelebAHQ(torch.utils.data.Dataset):
#     def __init__(self, sidelength=1024, split='train', subset=None):
#         self.name = 'celebahq'
#         self.channels = 3
#         self.mgrid = utils.get_mgrid(sidelength, dim=2)
#         self.sidelength = sidelength
#         # self.im_paths = sorted(glob('/om2/user/yilundu/datasets/data128x128/*.jpg'))
#         self.im_paths = sorted(glob('/om2/user/yilundu/datasets/data1024x1024/*.jpg'))
# 
#     def __len__(self):
#         return len(self.im_paths)
# 
#     def read_frame(self, path):
#         frame = imageio.imread(path)
#         frame = cv2.resize(frame, (self.sidelength, self.sidelength), interpolation=cv2.INTER_AREA)
#         frame = torch.from_numpy(frame).float()
#         frame /= 255.
#         frame -= 0.5
#         frame *= 2.
#         frame = frame.permute(2, 0, 1)
#         return frame
# 
#     def __getitem__(self, item):
#         rgb = self.read_frame(self.im_paths[item])
#         return {"rgb":rgb}


class CelebAHQGAN(torch.utils.data.Dataset):
    def __init__(self, sidelength=64, cache=None, cache_mask=None, split='train', subset=None):
        self.name = 'celebahq'
        # This is just a toy dataloader
        self.sidelength = sidelength
        subsample = 256 // self.sidelength
        self.mgrid = utils.get_mgrid((256, 256), dim=2, subsample=subsample)
        self.im_size = sidelength

    def __len__(self):
        return 27000

    def read_frame(self, path, item):
        frame = torch.zeros(64*64, 3)
        return frame

    def __getitem__(self, item):
        rgb = self.read_frame("", item)

        query_dict = {"idx": torch.Tensor([item]).long()}
        return {'context': query_dict, 'query': query_dict}, query_dict


class Cifar10(Dataset):
    def __init__(
            self,
            sidelength=1024,
            cache=None,
            cache_mask=None,
            split='train',
            subset=None):

        self.name = 'celebahq'
        self.channels = 3
        self.sidelength = sidelength

        cache = np.ctypeslib.as_array(cache.get_obj())
        cache = cache.reshape(50000, 3, 32, 32)
        cache = cache.astype("uint8")
        self.cache = torch.from_numpy(cache)

        self.im_size = 32

        cache_mask = np.ctypeslib.as_array(cache_mask.get_obj())
        cache_mask = cache_mask.reshape(50000)
        cache_mask = cache_mask.astype("uint8")
        self.cache_mask = torch.from_numpy(cache_mask)

        transform = transforms.ToTensor()

        self.data = CIFAR10(
            "data/cifar10",
            transform=transform,
            train=True,
            download=True)
        self.test_data = CIFAR10(
            "data/cifar10",
            transform=transform,
            train=False,
            download=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.cache_mask is not None:
            # cache, cache_mask = self.generate_array()
            if self.cache_mask[index] == 0:
                im, _ = self.data[index]
                frame = (im * 255).numpy()
                self.cache[index] = torch.from_numpy(frame.astype(np.uint8))
                self.cache_mask[index] = 1

            frame = np.array(self.cache[index])
        else:
            im, label = self.data[index]
            frame = (im * 255).numpy()

        frame = torch.from_numpy(frame).float()
        frame /= 255.
        frame -= 0.5
        frame *= 2.
        # np.random.seed((index + int(time.time() * 1e7)) % 2**32)

        return_dict = {'rgb': frame}

        return return_dict

class CelebAHQ(torch.utils.data.Dataset):
    def __init__(self, sidelength=1024, cache=None, cache_mask=None, split='train', subset=None):
        self.name = 'celebahq'
        self.channels = 3
        self.im_paths = sorted(glob('/data/vision/billf/scratch/yilundu/dataset/celebahq/data128x128/*.jpg'))
        # self.path = "/datasets01/celebAHQ/081318/imgHQ{:05}.npy"
        # self.mgrid = utils.get_mgrid(sidelength, dim=2)
        self.sidelength = sidelength
        # self.im_paths = sorted(glob('/om2/user/yilundu/datasets/data128x128/*.jpg'))

        cache = np.ctypeslib.as_array(cache.get_obj())
        cache = cache.reshape(30000, 128, 128, 3)
        cache = cache.astype("uint8")
        self.cache = torch.from_numpy(cache)
        self.im_size = sidelength

        cache_mask = np.ctypeslib.as_array(cache_mask.get_obj())
        cache_mask = cache_mask.reshape(30000)
        cache_mask = cache_mask.astype("uint8")
        self.cache_mask = torch.from_numpy(cache_mask)

        self.split = split

    def __len__(self):
        if self.split == "train":
            return 29000
        else:
            return 1000

    def read_frame(self, path, item):
        # if path in self.cache_im:
        #     im = self.cache_im[path]
        # else:
        #     self.cache_im[path] = im

        if self.cache_mask is not None:
            # cache, cache_mask = self.generate_array()
            if self.cache_mask[item] == 0:
                frame = imageio.imread(path)
                frame = cv2.resize(frame, (self.sidelength, self.sidelength), interpolation=cv2.INTER_AREA)
                self.cache[item, :self.sidelength, :self.sidelength] = torch.from_numpy(frame.astype(np.uint8))
                self.cache_mask[item] = 1

            frame = np.array(self.cache[item][:self.sidelength, :self.sidelength])
        else:
            frame = imageio.imread(path)
            # frame = frame.transpose((1, 2, 0))

        # scale = 128 // self.sidelength
        # frame = frame[::scale, ::scale, :]
        frame = torch.from_numpy(frame).float()
        frame /= 255.
        frame -= 0.5
        frame *= 2.
        frame = frame.permute(2, 0, 1)

        return frame

    def __getitem__(self, item):
        if self.split == "train":
            rgb = self.read_frame(self.im_paths[item], item)
        else:
            rgb = self.read_frame(self.im_paths[item+29000], item)

        return {"rgb":rgb}


class CelebA(torch.utils.data.Dataset):
    def __init__(self, sidelength, split='train'):
        transform = Compose([
            Resize((sidelength, sidelength)),
            CenterCrop((sidelength, sidelength)),
            ToTensor(),
            Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
        ])

        self.transform = transform
        self.path = "/datasets01/CelebA/CelebA/072017/img_align_celeba/"
        self.labels = pd.read_csv("/private/home/yilundu/list_attr_celeba.txt", sep="\s+", skiprows=1)
        self.name = 'celeba'

        self.channels = 3
        self.mgrid = utils.get_mgrid(sidelength, dim=2)
        self.sidelength = sidelength

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        info = self.labels.iloc[index]
        fname = info.name
        path = osp.join(self.path, fname)
        im = self.transform(Image.open(path))

        return {"rgb": im}


class NYUDepth(torch.utils.data.Dataset):
    def __init__(self, sidelength, split='train', depth_data_path="/om2/user/katiemc/data/nyu_depth_v2_labeled.mat"):

        # read mat file
        #f = sio.loadmat(depth_data_path,verify_compressed_data_integrity=False)
        f = h5py.File(depth_data_path)

        # save for normalization
        self.max_depth = np.max(f["depths"])
        self.min_depth = np.min(f["depths"])
        self.mean_depth = np.mean(f["depths"])

        # ad hoc train-val split --- to be updated w/ official split later
        train_size = 0.7
        img_train, img_val, depth_train, depth_val = train_test_split(np.array(f["images"]), np.array(f["depths"]), test_size = 1-train_size, random_state = 7)

        if split == "train":
            self.img_dataset = img_train
            self.depth_dataset = depth_train
        elif split == "val":
            self.img_dataset = img_val
            self.depth_dataset = depth_val

        self.channels = 3 + 1 #CHANGE??
        self.mgrid = utils.get_mgrid(sidelength, dim=2)
        self.sidelength = sidelength

    def __len__(self):
        return len(self.img_dataset)

    def __getitem__(self, item):

        transform = Compose([
            Resize((self.sidelength, self.sidelength)),
            CenterCrop((self.sidelength, self.sidelength)),
            ToTensor(),
            Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
        ])

        # help from: https://ddokkddokk.tistory.com/21
        img = self.img_dataset[item]
        # reshape
        img_ = np.empty([480, 640, 3])
        img_[:, :, 0] = img[0, :, :].T
        img_[:, :, 1] = img[1, :, :].T
        img_[:, :, 2] = img[2, :, :].T
        img = Image.fromarray(np.uint8(img_)) #Image.fromarray(img_)#torch.from_numpy(img_)
        img = transform(img)
        # read corresponding depth (aligned to the image, in-painted) of size [640 x 480], float64
        depth = self.depth_dataset[item]

        print("max depth: ", np.max(depth), " min depth: ", np.min(depth))
        depth = (depth - self.min_depth) / (self.max_depth - self.min_depth)
        depth = Image.fromarray(depth.T, mode="F")#depth_))#torch.from_numpy(img_)
        print("created pil image! ")
        transform = Compose([
            Resize((self.sidelength, self.sidelength)),
            CenterCrop((self.sidelength, self.sidelength)),
            ToTensor(),
            Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
        ])

       # depth = transform(depth)
        depth = transform(depth)
        print("TF max depth: ", depth.max(), " min depth: ", depth.min())

        #print("shapes: ", np.shape(img), " depth: ", np.shape(depth))
        return {"img": img, "depth": depth}#, label": self.img_dataset[item][1]}


class SingleClassImagenet(torch.utils.data.Dataset):
    def __init__(self, sidelength, split='train'):
        transform = Compose([
            Resize((sidelength, sidelength)),
            CenterCrop((sidelength, sidelength)),
            ToTensor(),
            Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
        ])

        self.img_dataset = torchvision.datasets.ImageFolder(os.path.join('/om2/user/sitzmann/single_class_imgnet', split),
                                                            transform=transform, target_transform=None)
        self.name = 'single_imgnet'
        self.channels = 3
        self.mgrid = utils.get_mgrid(sidelength, dim=2)
        self.sidelength = sidelength

    def __len__(self):
        return len(self.img_dataset)

    def __getitem__(self, item):
        return {"rgb": self.img_dataset[item][0]}


class Trombone(torch.utils.data.Dataset):
    # combine Spoken Digits dataset with MNIST
    def __init__(self, split='train', cache=None, cache_wav=None, cache_mask=None):
        if split == "train":
            data = np.load("trombone_train.npz")
            self.ims = data['rgbs']
            self.ims = self.ims.transpose((0, 3, 1, 2))
            self.wavs = data['spects']
        else:
            data = np.load("trombone_test.npz")
            self.ims = data['rgbs']
            self.ims = self.ims.transpose((0, 3, 1, 2))
            self.wavs = data['spects']

        data = np.load("trombone.npz")
        mean, std = data['mean'], data['std']
        self.mean, self.std = mean, std

    def __len__(self):
        return len(self.ims)

    def __getitem__(self, idx):
        frame = torch.Tensor(self.ims[idx])
        wav = self.wavs[idx]
        wav = (wav - self.mean[:, None]) / (self.std[:, None] * 3)
        wav = np.clip(wav, -1, 1)
        waveform = torch.Tensor(wav)

        output = {'rgb': frame, 'audio': waveform, 'label': 1}
        return output


class IMNet(torch.utils.data.Dataset):
    def __init__(self, split='train', sampling=None):
        self.data_path = os.path.join('/data/vision/billf/scratch/yilundu/dataset/imnet', 'IM-NET/IMSVR/data', 'all_vox256_img_' + split + '.hdf5')
        self.sampling = sampling
        self.init_model_bool = False
        # self.init_model()


    def __len__(self):
        return 35019

    def init_model(self):
        data_path = self.data_path
        self.data_dict = h5py.File(data_path, 'r')
        self.data_points_int = self.data_dict['points_64'][:]
        self.data_points = (self.data_points_int.astype(np.float32) + 1) / 128 - 1
        # import pdb
        # pdb.set_trace()
        # print(self.data_points)
        self.data_values = self.data_dict['values_64'][:]
        self.data_voxels = self.data_dict['voxels'][:]

        self.init_model_bool = True

    def __getitem__(self, idx):

        if not self.init_model_bool:
            self.init_model()

        points = torch.from_numpy(self.data_points[idx]).float()
        occs = torch.from_numpy(self.data_values[idx]).float()

        if self.sampling is not None:
            idcs = np.random.randint(0, len(points), size=self.sampling)
            points = points[idcs]
            occs = occs[idcs]

        ctxt_dict = query_dict = {'x':points, 'occupancy':occs, 'idx':torch.Tensor([idx]).long()}
        return {'context':ctxt_dict, 'query':query_dict}, query_dict


class ImageNet(torch.utils.data.Dataset):
    def __init__(self, sidelength, split='train'):
        transform = Compose([
            Resize((sidelength, sidelength)),
            CenterCrop((sidelength, sidelength)),
            ToTensor(),
            Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
        ])

        self.img_dataset = torchvision.datasets.ImageFolder(os.path.join('/om/data/public/imagenet/images_complete/ilsvrc', split),
                                                             transform=transform, target_transform=None)
        self.name = 'imagenet'
        self.channels = 3
        self.mgrid = utils.get_mgrid(sidelength, dim=2)
        self.sidelength = sidelength

    def __len__(self):
        return len(self.img_dataset)

    def __getitem__(self, item):
        return {"rgb": self.img_dataset[item][0]}


class Cityscapes(torch.utils.data.Dataset):
    def __init__(self, sidelength, split='train'):
        transform = Compose([
            Resize((sidelength, sidelength)),
            CenterCrop((sidelength, sidelength)),
            ToTensor(),
            Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
        ])

        self.target_transform = Compose([
            Resize((sidelength, sidelength)),
            CenterCrop((sidelength, sidelength)),
        ])

        self.img_dataset = torchvision.datasets.Cityscapes('/om2/user/katiemc/data/cityscapes/', split=split,
                                                           transform=transform, target_type=["semantic", 'instance'],
                                                           target_transform=None,
                                                           mode="coarse")
        self.name = 'cityscapes'
        self.channels = 3 + 1
        self.mgrid = utils.get_mgrid(sidelength, dim=2)
        self.sidelength = sidelength

        self.class_weights = torch.Tensor([1. if not seg_class.ignore_in_eval else 0. for seg_class in self.img_dataset.classes])

    def __len__(self):
        return len(self.img_dataset)

    def __getitem__(self, item):
        semantic_map = self.img_dataset[item][1][0]
        semantic_map = self.target_transform(semantic_map)
        semantic_map = torch.from_numpy(np.array(semantic_map.getdata()).reshape(1, semantic_map.size[0], semantic_map.size[1]))

        instance_map = self.img_dataset[item][1][1]
        instance_map = self.target_transform(instance_map)
        instance_map = torch.from_numpy(np.array(instance_map.getdata()).reshape(1, instance_map.size[0], instance_map.size[1]))
        return {"rgb": self.img_dataset[item][0], "semantic": semantic_map, "instance": instance_map}


class UCF_101(torch.utils.data.Dataset):
    def __init__(self):
        # self.video_dataset = torchvision.datasets.UCF101('/om2/user/sitzmann/UCF_101_subset', '/om2/user/sitzmann/ucfTrainTestlist',
        self.video_dataset = torchvision.datasets.UCF101('/om2/user/sitzmann/UCF-101', '/om2/user/sitzmann/ucfTrainTestlist',
                                                         frames_per_clip=30, step_between_clips=1, frame_rate=None, fold=1)

    def __len__(self):
        return len(self.video_dataset)

    def __getitem__(self, item):
        video = self.video_dataset[item][0]
        video = video.permute(-1, 0, 1, 2)[None, ...].float() # (ch, frames, height, width)
        video /= 127.5
        video -= 1.

        video = F.interpolate(input=video, size=(30, 64, 64), mode='trilinear').squeeze(0)
        video = video.permute(1, 2, 3, 0) # (frames, height, width, channels)
        return video, video


class VideoGeneralizationWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, context_sparsity, query_sparsity, sparsity_range=(2048, 2048), subsample_test=False):
        self.dataset = dataset

        vid, _ = self.dataset[0]
        self.frames, self.sidelength, self.channels = vid.shape[0], vid.shape[1], vid.shape[-1]
        self.mgrid = utils.get_mgrid((self.frames, self.sidelength, self.sidelength), dim=3)

        self.context_sparsity = context_sparsity
        self.query_sparsity = query_sparsity
        self.subsample_test = subsample_test
        self.sparsity_range = sparsity_range

    def __len__(self):
        return len(self.dataset)

    def sparsify(self, video_flat, mgrid, sparsity):
        if sparsity == 'full':
            return video_flat, mgrid, torch.ones(video_flat.shape[0], 1)
        elif sparsity == 'half':
            video_flat_half = video_flat[:self.sidelength ** 2 // 2, :]
            mgrid_half = mgrid[:self.sidelength ** 2 // 2, :]
            return video_flat_half, mgrid_half, torch.ones(video_flat_half.shape[0], 1)
        elif sparsity == 'sampled':
            # Sample the upper limit at random. This is what we'll pad each batch to.
            if self.sparsity_range[0] == self.sparsity_range[1]:
                subsamples = self.sparsity_range[0]
            else:
                subsamples = np.random.randint(self.sparsity_range[0],
                                               self.sparsity_range[1])

            rand_idcs = np.random.choice(video_flat.shape[0],
                                         size=self.sparsity_range[1],
                                         replace=False)
            video_flat_sparse = video_flat[rand_idcs, :]
            coords_sub = mgrid[rand_idcs, :]

            # From the upper limit, sample again.
            rand_idcs_2 = np.random.choice(video_flat_sparse.shape[0], size=subsamples, replace=False)
            mask = torch.zeros(video_flat_sparse.shape[0], 1)
            mask[rand_idcs_2, 0] = 1.
            return video_flat_sparse, coords_sub, mask

    def __getitem__(self, idx):
        video, label = self.dataset[idx]# (frames, height, width, channels)
        video_flat = video.view(-1, 3)

        ctxt_pixels, ctxt_coords, ctxt_mask = self.sparsify(video_flat, self.mgrid, self.context_sparsity)
        query_pixels, query_coords, query_mask = self.sparsify(video_flat, self.mgrid, self.query_sparsity)

        ctxt_dict = {'x':ctxt_coords, 'y':ctxt_pixels, 'mask':ctxt_mask}
        query_dict = {'x':query_coords, 'y':query_pixels, 'mask':query_mask}

        gt_dict = deepcopy(query_dict)
        gt_dict.update({'video_flat':video_flat, 'label':label, 'dense_coords':self.mgrid})

        return {'context':ctxt_dict, 'query':query_dict}, gt_dict


class GeneralizationWrapper(torch.utils.data.Dataset):
    '''Assumes that 2D modalities from dataset are ordered (ch, height, width)'''
    def __init__(self, dataset, context_sparsity, query_sparsity, sparsity_range=(10, 200),
                 inner_loop_supervision_key=None, persistent_mask=False, idx_offset=0, padding=False,
                 cache=None):
        self.dataset = dataset
        self.im_size = self.dataset.im_size

        img = self.dataset[0]["rgb"]
        self.sidelength = img.shape[-1]

        subsample = 256 // self.sidelength
        self.mgrid = utils.get_mgrid((256, 256), dim=len(img.shape)-1, subsample=subsample)

        self.per_key_channels = {key:value.shape[0] for key, value in self.dataset[0].items()}
        self.padding = padding
        if 'semantic' in list(self.dataset[0].keys()):
            self.total_channels = len(self.dataset.img_dataset.classes)
            self.total_channels += int(np.sum([value for key, value in self.per_key_channels.items() if key != 'semantic']))
        else:
            self.total_channels = int(np.sum(list(self.per_key_channels.values())))

        self.per_key_channels['x'] = 2

        self.persistent_mask = persistent_mask
        self.mask_dir = os.path.join('/tmp/'+ f'_masks_{padding}_{sparsity_range[0]}_{sparsity_range[1]}_{self.sidelength}')

        self.context_sparsity = context_sparsity
        self.query_sparsity = query_sparsity
        self.sparsity_range = sparsity_range
        self.inner_loop_supervision_key = inner_loop_supervision_key
        self.idx_offset = idx_offset

        self.cache = cache

    def __len__(self):
        return len(self.dataset)

    def flatten_dict_entries(self, dict):
        return {key: value.permute(1, 2, 0).reshape(-1, self.per_key_channels[key]) for key, value in dict.items()}

    def sparsify(self, sample_dict, mgrid, sparsity, idx):
        if sparsity == 'full':
            result_dict = sample_dict
            result_dict = self.flatten_dict_entries(result_dict)
            result_dict['x'] = mgrid
            result_dict['mask'] = torch.ones_like(result_dict['x'][...,:1])
            return result_dict
        elif sparsity == 'half':
            result_dict = {key: value[:, :, :self.sidelength // 2] for key, value in sample_dict.items()}
            mgrid = mgrid.view(self.sidelength, self.sidelength, 2).permute(2, 0, 1)
            result_dict['x'] = mgrid[:, :, :self.sidelength // 2]
            mask = torch.ones_like(mgrid[:1, ...])
            mask = mask[:, :, :self.sidelength // 2].contiguous()
            result_dict = self.flatten_dict_entries(result_dict)
            result_dict['mask'] = mask.view(-1, 1)
            return result_dict
        elif sparsity == 'context':
            mask = np.ones((self.sidelength, self.sidelength)).astype(np.bool)
            mask[32:96, 32:96] = 0
            result_dict = {key: value[:, mask].transpose(1, 0).contiguous() for key, value in sample_dict.items()}
            nelem = result_dict['rgb'].shape[0]
            rix = np.random.permutation(nelem)[:1024]

            result_dict = {key: value[rix] for key, value in result_dict.items()}
            # mgrid = mgrid.view(self.sidelength, self.sidelength, 2).permute(2, 0, 1)

            result_dict['x'] = mgrid[mask.flatten()][rix]
            result_dict['mask'] = mask[mask==1][rix, None]
            return result_dict
        elif sparsity == 'sampled':
            if self.sparsity_range[0] == self.sparsity_range[1]:
                subsamples = self.sparsity_range[0]
            else:
                subsamples = np.random.randint(self.sparsity_range[0], self.sparsity_range[1])

            if not self.padding:
                # Sample upper_limit pixel idcs at random.
                lower_rand_idcs = np.random.choice(self.sidelength ** 2, size=self.sparsity_range[1], replace=False)
                upper_rand_idcs = np.random.choice(self.sparsity_range[1], size=subsamples, replace=False)

                mask_filepath = os.path.join(self.mask_dir, "%09d"%idx)
                if self.persistent_mask:
                    if not os.path.exists(mask_filepath):
                        with open(mask_filepath, 'wb') as mask_file:
                            pck.dump((lower_rand_idcs, upper_rand_idcs), mask_file)
                    else:
                        with open(mask_filepath, 'rb') as mask_file:
                            lower_rand_idcs, upper_rand_idcs = pck.load(mask_file)

                flat_dict = self.flatten_dict_entries(sample_dict)
                result_dict = {key: value[lower_rand_idcs] for key, value in flat_dict.items()}

                result_dict['mask'] = torch.zeros(self.sparsity_range[1], 1)
                result_dict['mask'][upper_rand_idcs, 0] = 1.
                result_dict['x'] = mgrid.view(-1, 2)[lower_rand_idcs, :]
                return result_dict
            else:
                rand_idcs = np.random.choice(self.sidelength**2, size=subsamples, replace=False)
                mask_filepath = os.path.join(self.mask_dir, "%09d"%idx)
                if self.persistent_mask:
                    if not os.path.exists(mask_filepath):
                        with open(mask_filepath, 'wb') as mask_file:
                            pck.dump(rand_idcs, mask_file)
                    else:
                        with open(mask_filepath, 'rb') as mask_file:
                            rand_idcs = pck.load(mask_file)

                result_dict = self.flatten_dict_entries(sample_dict)
                result_dict['mask'] = torch.zeros(self.sidelength**2, 1)
                result_dict['mask'][rand_idcs, 0] = 1.
                result_dict['x'] = mgrid.view(-1, 2)
                return result_dict

    def __getitem__(self, idx):
        if self.cache is not None:
            if idx not in self.cache:
                self.cache[idx] = self.dataset[idx]
            # else:
            #     print('used cache')

            sample_dict = self.cache[idx]
        else:
            sample_dict = self.dataset[idx]


        idx_other = random.randint(0, len(self.dataset) - 1)
        if self.cache is not None:
            if idx not in self.cache:
                self.cache[idx_other] = self.dataset[idx_other]
            # else:
            #     print('used cache')

            sample_dict_other = self.cache[idx_other]
        else:
            sample_dict_other = self.dataset[idx_other]

        mgrid = self.mgrid
        dist_mse = (sample_dict_other['rgb'].reshape(-1) - sample_dict['rgb'].reshape(-1)).pow(2).mean()
        ctxt_dict = self.sparsify(sample_dict, mgrid, self.context_sparsity, idx)
        query_dict = self.sparsify(sample_dict, mgrid, self.query_sparsity, idx)

        if self.inner_loop_supervision_key is not None:
            ctxt_dict['y'] = ctxt_dict[self.inner_loop_supervision_key]
            query_dict['y'] = query_dict[self.inner_loop_supervision_key]

        ctxt_dict['idx'] = torch.Tensor([idx]).long() + self.idx_offset
        query_dict['idx'] = torch.Tensor([idx]).long() + self.idx_offset

        ctxt_dict['idx_other'] = torch.Tensor([idx_other]).long() + self.idx_offset
        query_dict['idx_other'] = torch.Tensor([idx_other]).long() + self.idx_offset
        query_dict['mse'] = dist_mse
        ctxt_dict['mse'] = dist_mse

        query_dict = ctxt_dict

        return {'context': ctxt_dict, 'query': query_dict}, query_dict


class ImplicitGANDataset():
    def __init__(self, real_dataset, fake_dataset):
        self.real_dataset = real_dataset
        self.fake_dataset = fake_dataset
        self.im_size = self.real_dataset.im_size

    def __len__(self):
        return len(self.fake_dataset)

    def __getitem__(self, idx):
        real = self.real_dataset[idx]
        fake = self.fake_dataset[idx]
        return fake, real


class DatasetSampler(torch.utils.data.sampler.Sampler):
    # modified from: https://stackoverflow.com/questions/57913825/how-to-select-specific-labels-in-pytorch-mnist-dataset
    def __init__(self, mask, data_source):
        # mask is a tensor specifying which classes to include
        self.mask = mask
        self.data_source = data_source

    def __iter__(self):
        return iter([i.item() for i in torch.nonzero(self.mask)])

    def __len__(self):
        return len(self.data_source)

class SpokenDigits(torch.utils.data.Dataset):
    def __init__(self, split='train', resample_rate=None, data_path="/om/user/katiemc/sc09/"): #,audio_len = 16000, ):

        # read in all .wav files for data split
        self.data_path = data_path + split
        self.wav_files = sorted(glob(os.path.join(self.data_path, '*'))) # added sorted() to ensure always pull same order

        # extract labels from file (each audio file begins with digit)
        self.labels = [fname.split("/")[-1].split("_")[0] for fname in self.wav_files]

        self.resample_rate = resample_rate
        # self.audio_len = audio_len
        self.channels=1

    def __len__(self):
        return len(self.wav_files)

    def __getitem__(self, idx):
        # audio processing help from: https://pytorch.org/tutorials/beginner/audio_preprocessing_tutorial.html
        # to install, run: conda install -c pytorch torchaudio

        # convert audio to tensor
        waveform, sample_rate = torchaudio.load(self.wav_files[idx])

        # # zero-pad to desired length
        # pad = torch.zeros([1,self.audio_len - waveform.shape[1]])
        # waveform = torch.cat([waveform, pad], dim=1)

        # transform by resample rate
        if self.resample_rate is not None:
            waveform = torchaudio.transforms.Resample(sample_rate, self.resample_rate)(waveform[0, :].view(1, -1))
            self.rate = self.resample_rate
        else: self.rate = sample_rate

        waveform = torch.Tensor(waveform).view(-1, 1)

        return {"audio": waveform, "label": self.labels[idx]}

class NSynth(torch.utils.data.Dataset):
    def __init__(self, split='train', resample_rate=None, data_path="/private/home/yilundu/sandbox/function-space-gan/data/nsynth/records/"):
        data_points = json.load(open("nsynth_{}.json".format(split), 'r'))
        self.data_points = data_points

        self.wav_files = [self.data_points[i][1]['path'] for i in range(len(self.data_points))]
        self.pitch_codes = [self.data_points[i][1]['pitch'] for i in range(len(self.data_points))]
        self.instrument_labels = [self.data_points[i][1]['instrument_source'] for i in range(len(self.data_points))]

        # combine meta-data into labels dict per point
        self.labels = [{'instrument_labels': inst_label, 'pitch_code': pitch_code} for
                       inst_label, pitch_code, in
                       zip(self.instrument_labels, self.pitch_codes)]

        self.resample_rate = resample_rate
        self.channels = 1
        self.rate = 16000

    def __len__(self):
        return len(self.data_points)

    def __getitem__(self, idx):
        # audio processing help from: https://pytorch.org/tutorials/beginner/audio_preprocessing_tutorial.html
        import torchaudio  # NOTE: adding here so it doesn't break other envs
        # to install, run: conda install -c pytorch torchaudio

        waveform, sample_rate = torchaudio.load(self.wav_files[idx])
        waveform = waveform[:, :16000]

        waveform = torch.Tensor(waveform).view(-1, 1)

        return {"audio": waveform, "label": self.labels[idx]}


class AudioGeneralizationWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, sampling=None, do_pad=True):#context_sparsity, query_sparsity, sparsity_range=(16000,)):

        self.dataset = dataset
        self.num_timesteps = 16000
        self.mgrid = utils.get_mgrid(int(self.num_timesteps), dim=1)

        self.sampling = sampling
        self.do_pad = do_pad

        self.rate = dataset.rate
        self.updated_rate_sampling = False # currently hacky to avoid changing rate multiple times (change!)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        sample_dict = self.dataset[idx]
        idx_other = random.randint(0, len(self.dataset) - 1)
        sample_dict_other = self.dataset[idx_other]

        dist_mse = (sample_dict_other['audio'].reshape(-1) - sample_dict['audio'].reshape(-1)).pow(2).mean()

        # get flattened audio signal
        wav = sample_dict["audio"]
        mgrid = self.mgrid

        if self.do_pad:
            # zero-pad to desired length
            num_zeros_pad = self.num_timesteps - wav.shape[0]
            pad = torch.zeros([num_zeros_pad,1])
            wav = torch.cat([wav, pad], dim=0)
            mask = torch.ones(self.num_timesteps, 1)
            mask[self.num_timesteps - num_zeros_pad:, 0] = 0

        if self.sampling is not None:
            idcs = sorted(np.random.choice(len(mgrid), size=self.sampling, replace=False))
            mgrid = mgrid[idcs,:]
            wav = wav[idcs,:]
            mask = mask[idcs,:]
            # correct rate: drop rate according to subsampling of signal
            if not self.updated_rate_sampling:
                self.rate = int(self.rate * (self.sampling/self.num_timesteps))
                self.updated_rate_sampling = True # to avoid updating multiple times

        ctxt_dict = query_dict = {'x':mgrid, 'wav':wav, 'mask':mask,'rate': self.rate, 'idx':torch.Tensor([idx]).long()}
        query_dict['idx_other'] = torch.Tensor([idx_other]).long()
        query_dict['mse'] = dist_mse

        return {'context':ctxt_dict, 'query':query_dict}, query_dict

class SubURMP(torch.utils.data.Dataset):
    # subset of URMP dataset - see: https://www.cs.rochester.edu/~cxu22/d/vagan/README.md
    def __init__(self, sidelength, split='train', resample_rate=None, data_path="/om/user/katiemc/Sub-URMP/"):

        # read in all .wav and .png files for data split
        self.img_files = sorted(glob(os.path.join(data_path + "img/" + split, '**/*.jpg'))) # added sorted() to ensure always pull same order
        self.wav_files = sorted(glob(os.path.join(data_path + "chunk/" + split, '**/*.wav')))

        # files are labeled "instrument_songNumber_timestep" (same across modalities)
        self.labels = [fname.split("/")[-1].split(".")[0] for fname in self.wav_files]

        # audio + img transforms
        self.resample_rate = resample_rate
        self.transform = Compose([
            Resize((sidelength, sidelength)),
            CenterCrop((sidelength, sidelength)),
            ToTensor(),
            Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
        ])

        self.channels = 4 # r,g,b (img) + amplitude (audio)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # audio processing help from: https://pytorch.org/tutorials/beginner/audio_preprocessing_tutorial.html
        import torchaudio # NOTE: adding here so it doesn't break other envs
        # to install, run: conda install -c pytorch torchaudio

        img = self.transform(Image.open(self.img_files[idx]))

        # convert audio to tensor
        waveform, sample_rate = torchaudio.load(self.wav_files[idx])
        waveform = waveform[0] # each file has duplicate audio signals

        # transform by resample rate
        if self.resample_rate is not None:
            waveform = torchaudio.transforms.Resample(sample_rate, self.resample_rate)(waveform[0, :].view(1, -1))
            self.rate = self.resample_rate
        else: self.rate = sample_rate

        waveform = torch.Tensor(waveform).view(-1, 1)

        return {"rgb": img, "audio": waveform, "label": self.labels[idx]}

class SpokenMNIST(torch.utils.data.Dataset):
    # combine Spoken Digits dataset with MNIST
    def __init__(self, sidelength, split='train'):

        self.digit_label_conversion = {digit_str: digit_int for digit_int, digit_str in zip([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                                                                     ["Zero","One","Two","Three","Four","Five","Six","Seven","Eight","Nine"])}

        self.mnist_dataset = MNIST(split, sidelength)
        self.spoken_digits_dataset = SpokenDigits("valid" if split == "val" else split) # handle diff naming in file system

        # pair same written digit class w/ spoken digit class

        # create class to sample map for audio
        self.audio_class_map = {}
        for digit_data in self.spoken_digits_dataset:
            digit = self.digit_label_conversion[digit_data['label']] # convert to int to match mnist dataset
            if digit in self.audio_class_map: self.audio_class_map[digit].append(digit_data['audio'])
            else: self.audio_class_map[digit] = [digit_data['audio']]

        self.data_pairs = [] # list of dicts - img, audio, label

        np.random.seed(7) # set seed to ensure always choose same pair for reruns
        for idx, digit_data in enumerate(self.mnist_dataset):
            digit_label = digit_data["label"]
            # if digit_label in self.audio_class_map: # ensure digits included in both datasets
            audio_sample_idx = np.random.choice(range(len(self.audio_class_map[digit_label])))
            self.data_pairs.append(({"audio": self.audio_class_map[digit_label][audio_sample_idx], "rgb": digit_data["img"], "label": digit_label}))

        self.channels = 4 # r,g,b (img) + amplitude (audio)
        self.rate = self.spoken_digits_dataset.rate

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        return self.data_pairs[idx]


class VoxCeleb(torch.utils.data.Dataset):
    # combine Spoken Digits dataset with MNIST
    def __init__(self, split='train'):
        self.image_path = "/data/vision/billf/scratch/yilundu/dataset/voxceleba/unzippedIntervalFaces/data/"
        self.audio_path = "/data/vision/billf/scratch/yilundu/dataset/voxceleba/wav"
        csv_path = "/data/vision/billf/scratch/yilundu/dataset/voxceleba/vox1_meta.csv"
        df = pd.read_csv(csv_path, sep='\t')
        vox_id = df['VoxCeleb1 ID']
        face_id = df['VGGFace1 ID']


        # examples = []

        # for i in range(len(vox_id)):
        #     vox_id_i = vox_id[i]
        #     face_id_i = face_id[i]

        #     audio_path = osp.join(self.audio_path, vox_id_i)
        #     image_path = osp.join(self.image_path, face_id_i, "1.6")

        #     folders = sorted(os.listdir(image_path))

        #     for folder in folders:
        #         audio_path_i = osp.join(audio_path, folder, '00001.wav')
        #         image_path_i = osp.join(image_path, folder, '1', '01.jpg')

        #         examples.append((audio_path_i, image_path_i))

        examples = pickle.load(open("examples.p", "rb"))
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):

        try:
            audio_path, im_path =  self.examples[idx]
            waveform, sample_rate = torchaudio.load(audio_path)
            waveform = waveform[:, 16000:32000]
            gt_spectrogram = torchaudio.transforms.Spectrogram()(waveform) / 100.
            # decode_waveform = torchaudio.transforms.GriffinLim().forward(gt_spectrogram)
            # import pdb
            # pdb.set_trace()
            # print(decode_waveform)
            # print(gt_spectrogram)

            waveform = waveform.view(-1, 1)

            frame = imageio.imread(im_path)
            frame = cv2.resize(frame, (64, 64), interpolation=cv2.INTER_AREA)
            frame = torch.from_numpy(frame).float()
            frame /= 255.
            frame -= 0.5
            frame *= 2.
            frame = frame.permute(2, 0, 1)

            output = {'rgb': frame, 'audio': gt_spectrogram, 'label': 1}
            # print("success")

            return output
        except Exception as e:
            # print(e)
            return self.__getitem__(random.randint(0, len(self.examples) - 1))


class Instrument(torch.utils.data.Dataset):
    # combine Spoken Digits dataset with MNIST
    def __init__(self, split='train', cache=None, cache_wav=None, cache_mask=None):

        if split == "train":
            ims = sorted(glob("/data/vision/billf/scratch/yilundu/dataset/suburmp/Sub-URMP/img/train/cello/*.jpg"))
            wavs = sorted(glob("/data/vision/billf/scratch/yilundu/dataset/suburmp/Sub-URMP/chunk/train/cello/*.wav"))
        else:
            ims = sorted(glob("/data/vision/billf/scratch/yilundu/dataset/suburmp/Sub-URMP/img/validation/cello/*.jpg"))
            wavs = sorted(glob("/data/vision/billf/scratch/yilundu/dataset/suburmp/Sub-URMP/chunk/validation/cello/*.wav"))
        self.ims = ims
        self.wavs = wavs

        if cache is not None:
            cache = np.ctypeslib.as_array(cache.get_obj())
            cache = cache.reshape(9800, 128, 128, 3)
            cache = cache.astype("uint8")
            self.cache = torch.from_numpy(cache)
        else:
            self.cache = cache

        if cache_wav is not None:
            cache_wav = np.ctypeslib.as_array(cache_wav.get_obj())
            cache_wav = cache_wav.reshape(9800, 200, 41)
            cache_wav = cache_wav.astype("float32")
            self.cache_wav = torch.from_numpy(cache_wav)
        else:
            self.cache_wav = cache_wav

        data = np.load("instrument.npz")
        self.mean = torch.Tensor(data['mean'])[:-1]
        self.std = torch.Tensor(data['std'])[:-1]

        if cache_mask is not None:
            cache_mask = np.ctypeslib.as_array(cache_mask.get_obj())
            cache_mask = cache_mask.reshape(9800)
            cache_mask = cache_mask.astype("uint8")

            self.cache_mask = torch.from_numpy(cache_mask)
        else:
            self.cache_mask = cache_mask

    def __len__(self):
        return len(self.ims)

    def __getitem__(self, idx):
        audio_path, im_path =  self.wavs[idx], self.ims[idx]

        if self.cache_mask is not None:
            if self.cache_mask[idx] == 0:
                frame = imageio.imread(im_path)
                frame = frame[:, 100:1180, :]
                frame = cv2.resize(frame, (128, 128), interpolation=cv2.INTER_AREA)
                waveform, sample_rate = torchaudio.load(audio_path)
                waveform = torchaudio.transforms.Resample(sample_rate, 16000)(waveform).permute(1, 0).contiguous()

                waveform = waveform[None, :, 0]
                waveform = Spectrogram()(waveform).squeeze()
                waveform = waveform[:-1, :].contiguous()
                waveform = torch.log(waveform)

                waveform = (waveform - self.mean[:, None]) / (self.std[:, None] * 3)
                waveform = torch.clamp(waveform, -1, 1)

                self.cache[idx] = torch.from_numpy(frame.astype(np.uint8))
                self.cache_wav[idx] = waveform
                self.cache_mask[idx] = 1

            frame = np.array(self.cache[idx])
            waveform = self.cache_wav[idx].clone()

        else:
            frame = imageio.imread(im_path)
            frame = imageio.imread(im_path)
            frame = frame[:, 100:1180, :]
            frame = cv2.resize(frame, (128, 128), interpolation=cv2.INTER_AREA)
            waveform, sample_rate = torchaudio.load(audio_path)
            waveform = torchaudio.transforms.Resample(sample_rate, 16000)(waveform).permute(1, 0).contiguous()

            waveform = waveform[None, :, 0]
            waveform = Spectrogram()(waveform).squeeze()
            waveform = waveform[:-1, :].contiguous()
            waveform = torch.log(waveform)

            waveform = (waveform - self.mean[:, None]) / (self.std[:, None] * 3)
            waveform = torch.clamp(waveform, -1, 1)


        # print(waveform.max(), waveform.min())
        # waveform = waveform / (3 * 0.7650)
        # waveform = torch.clamp(waveform, -1., 1.)
        frame = torch.from_numpy(frame).float()
        frame /= 255.
        frame -= 0.5
        frame *= 2.
        frame = frame.permute(2, 0, 1)

        output = {'rgb': frame, 'audio': waveform, 'label': 1}
        # print("success")

        return output


class AVGeneralizationWrapper(torch.utils.data.Dataset):
    # combined wrapper for audio and visual data
    # assumes dataset will return "audio" and "rgb" signals
    def __init__(self, dataset, sparsity, audio_sampling=None, do_pad=True, sparsity_range=(10,200), audio_sparsity_range=(16000,)):

        self.dataset = dataset

        # set to the maximum time duration of an audio signal for padding
        self.num_timesteps = 16000
        self.audio_mgrid = utils.get_mgrid(int(self.num_timesteps), dim=1)

        self.sidelength = 128
        self.img_mgrid = utils.get_mgrid((128, 128), dim=2, subsample=1)

        # Make audio prediction a spectogram
        self.audio_mgrid = utils.get_mgrid((200, 41), dim=2, subsample=1)
        # self.mgrid = utils.get_mgrid(8000, dim=1)

        self.audio_sampling = audio_sampling
        self.do_pad = do_pad

        self.rate = 16000
        self.updated_rate_sampling = False # currently hacky to avoid changing rate multiple times (change!)

        self.sparsity = sparsity
        self.sparsity_range = sparsity_range
        self.audio_sparsity_range= audio_sparsity_range

    def __len__(self):
        return len(self.dataset)

    def sparsify(self, img, mgrid, sparsity, audio=False):
        result_dict = {"rgb": img.permute(1, 2, 0).reshape(-1, img.shape[0])}
        if sparsity == 'full':
            result_dict['x'] = mgrid
            result_dict['mask'] = torch.ones_like(result_dict['x'][...,:1])
            return result_dict
        elif sparsity == 'sampled':
            if self.sparsity_range[0] == self.sparsity_range[1]:
                subsamples = self.sparsity_range[0]
            else:
                subsamples = np.random.randint(self.sparsity_range[0], self.sparsity_range[1])

            # Sample upper_limit pixel idcs at random.
            lower_rand_idcs = np.random.choice(mgrid.shape[0], size=self.sparsity_range[1], replace=False)
            upper_rand_idcs = np.random.choice(self.sparsity_range[1], size=subsamples, replace=False)

            flat_dict = result_dict
            result_dict = {key: value[lower_rand_idcs] for key, value in flat_dict.items()}

            result_dict['mask'] = torch.zeros(self.sparsity_range[1], 1)
            result_dict['mask'][upper_rand_idcs, 0] = 1.
            result_dict['x'] = mgrid.view(-1, 2)[lower_rand_idcs, :]
            return result_dict

    def __getitem__(self, idx):

        sample_dict = self.dataset[idx]
        idx_other = random.randint(0, len(self.dataset) - 1)
        sample_dict_other = self.dataset[idx_other]

        # process img
        img = sample_dict["rgb"]
        mgrid = self.img_mgrid
        ctxt_dict = self.sparsify(img, mgrid, self.sparsity)

        mgrid = self.audio_mgrid

        dist_mse = (sample_dict_other['rgb'].reshape(-1) - sample_dict['rgb'].reshape(-1)).pow(2).mean() + (sample_dict_other['audio'].reshape(-1) - sample_dict['audio'].reshape(-1)).pow(2).mean()

        # process audio
        # get flattened audio signal
        wav = sample_dict["audio"]
        mgrid = self.audio_mgrid
        wav = wav[None, :, :]
        audio_ctxt_dict = self.sparsify(wav, mgrid, self.sparsity, audio=True)

        # if self.do_pad:
        #     # zero-pad to desired length
        #     num_zeros_pad = self.num_timesteps - wav.shape[0]
        #     pad = torch.zeros([num_zeros_pad,1])
        #     wav = torch.cat([wav, pad], dim=0)
        #     mask = torch.ones(self.num_timesteps, 1)
        #     mask[self.num_timesteps - num_zeros_pad:, 0] = 0

        # wav = sample_dict["audio"]

        # if self.audio_sampling is not None:
        #     idcs = sorted(np.random.choice(len(mgrid), size=self.audio_sampling, replace=False))
        #     mgrid = mgrid[idcs,:]
        #     wav = wav[idcs,:]
            # mask = mask[idcs,:]

        # concatenate x + mask, idx is the same, independent wav, rate, + rgb
        ctxt_dict.update({'wav':audio_ctxt_dict['rgb'],'rate': self.rate,'idx':torch.Tensor([idx]).long()})
        ctxt_dict['idx_other'] = torch.Tensor([idx_other]).long()

        # account for differences in channels and input (for coords + mask)
        # include an attribute to decompose back to img + audio signals
        ctxt_dict['num_pixels'] = ctxt_dict['x'].shape[0]
        # zero_t_coord = torch.zeros(ctxt_dict['num_pixels'],1)
        # img_coords = np.concatenate([ctxt_dict['x'], zero_t_coord], axis=1)

        # ctxt_dict["num_timesteps"] = mgrid.shape[0]
        # zero_xy_coords = torch.ones(ctxt_dict["num_timesteps"],2)

        # audio_coords = np.concatenate([zero_xy_coords, mgrid], axis=1) * 100 # account for diff in frequency

        ctxt_dict['audio_coord'] = audio_ctxt_dict['x']
        ctxt_dict['audio_mask'] = audio_ctxt_dict['mask']
        # ctxt_dict['x'] = np.vstack([img_coords, audio_coords])
        ctxt_dict['mask'] = ctxt_dict['mask']
        ctxt_dict["label"] = sample_dict["label"]
        ctxt_dict['mse'] = dist_mse

        query_dict = ctxt_dict
        # label_dict = {}

        # label_dict['mse'] = ctxt_dict['mse']
        # label_dict['wav'] = ctxt_dict['wav']
        # label_dict['idx'] = ctxt_dict['idx']
        # label_dict['idx_other'] = ctxt_dict['idx_other']
        # label_dict['rate'] = ctxt_dict['rate']
        # label_dict['rgb'] = ctxt_dict['rgb']
        # label_dict['x'] = ctxt_dict['x']
        # label_dict['mask'] = ctxt_dict['mask']
        # label_dict['num_pixels'] = ctxt_dict['num_pixels']

        # del ctxt_dict['mse']
        # del ctxt_dict['wav']
        # del ctxt_dict['rgb']
        # del ctxt_dict['mask']
        # del ctxt_dict['num_pixels']

        return {'context':ctxt_dict, 'query':query_dict}, query_dict

if __name__ == "__main__":
    # dataset = VoxCeleb()
    # dataset[2]

    dataset = Instrument()
    spects = []
    for i in range(1000):
        spect = dataset[i]['audio']
        spects.append(spect)

    spects = torch.stack(spects, dim=0)
    import pdb
    pdb.set_trace()
    print(spects)
