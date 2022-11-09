# Enable import from parent package
import sys
import os
import collections
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
import numpy as np
import random

import dataio, meta_modules, summaries, loss_functions, modules, training
import torch.distributed as dist

from multiprocessing import Manager
import multiprocessing

import high_level_models
import torch
import torch.multiprocessing as mp

import ctypes
from torch.utils.data import DataLoader
import configargparse
import numpy as np
import config
from torch_ema import ExponentialMovingAverage
from imageio import imwrite

from torchvision.utils import make_grid
import mcubes
import math


def sample_points_triangle(vertices, triangles, num_of_points):
    epsilon = 1e-6
    triangle_area_list = np.zeros([len(triangles)],np.float32)
    triangle_normal_list = np.zeros([len(triangles),3],np.float32)
    for i in range(len(triangles)):
        #area = |u x v|/2 = |u||v|sin(uv)/2
        a,b,c = vertices[triangles[i,1]]-vertices[triangles[i,0]]
        x,y,z = vertices[triangles[i,2]]-vertices[triangles[i,0]]
        ti = b*z-c*y
        tj = c*x-a*z
        tk = a*y-b*x
        area2 = math.sqrt(ti*ti+tj*tj+tk*tk)
        if area2<epsilon:
            triangle_area_list[i] = 0
            triangle_normal_list[i,0] = 0
            triangle_normal_list[i,1] = 0
            triangle_normal_list[i,2] = 0
        else:
            triangle_area_list[i] = area2
            triangle_normal_list[i,0] = ti/area2
            triangle_normal_list[i,1] = tj/area2
            triangle_normal_list[i,2] = tk/area2
    
    triangle_area_sum = np.sum(triangle_area_list)
    sample_prob_list = (num_of_points/triangle_area_sum)*triangle_area_list

    triangle_index_list = np.arange(len(triangles))

    point_normal_list = np.zeros([num_of_points,6],np.float32)
    count = 0
    watchdog = 0

    while(count<num_of_points):
        np.random.shuffle(triangle_index_list)
        watchdog += 1
        if watchdog>100:
            print("infinite loop here!")
            return point_normal_list
        for i in range(len(triangle_index_list)):
            if count>=num_of_points: break
            dxb = triangle_index_list[i]
            prob = sample_prob_list[dxb]
            prob_i = int(prob)
            prob_f = prob-prob_i
            if np.random.random()<prob_f:
                prob_i += 1
            normal_direction = triangle_normal_list[dxb]
            u = vertices[triangles[dxb,1]]-vertices[triangles[dxb,0]]
            v = vertices[triangles[dxb,2]]-vertices[triangles[dxb,0]]
            base = vertices[triangles[dxb,0]]
            for j in range(prob_i):
                #sample a point here:
                u_x = np.random.random()
                v_y = np.random.random()
                if u_x+v_y>=1:
                    u_x = 1-u_x
                    v_y = 1-v_y
                ppp = u*u_x+v*v_y+base

                point_normal_list[count,:3] = ppp
                point_normal_list[count,3:] = normal_direction
                count += 1
                if count>=num_of_points: break

    return point_normal_list


def write_ply_triangle(name, vertices, triangles):
        fout = open(name, 'w')
        fout.write("ply\n")
        fout.write("format ascii 1.0\n")
        fout.write("element vertex "+str(len(vertices))+"\n")
        fout.write("property float x\n")
        fout.write("property float y\n")
        fout.write("property float z\n")
        fout.write("element face "+str(len(triangles))+"\n")
        fout.write("property list uchar int vertex_index\n")
        fout.write("end_header\n")
        for ii in range(len(vertices)):
                fout.write(str(vertices[ii,0])+" "+str(vertices[ii,1])+" "+str(vertices[ii,2])+"\n")
        for ii in range(len(triangles)):
                fout.write("3 "+str(triangles[ii,0])+" "+str(triangles[ii,1])+" "+str(triangles[ii,2])+"\n")
        fout.close()


p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--logging_root', type=str, default=os.path.join(config.log_root, 'latent_gan'), help='root for logging')
p.add_argument('--experiment_name', type=str, required=True,
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')

# General training options
p.add_argument('--gpus', type=int, default=1)
p.add_argument('--batch_size', type=int, default=128)
p.add_argument('--lr', type=float, default=1e-4, help='learning rate. default=5e-5')
p.add_argument('--num_epochs', type=int, default=40001,
               help='Number of epochs to train for.')
p.add_argument('--train_sparsity_range', type=int, nargs='+', default=[64**2, 64**2],
               help='Two integers: lowest number of sparse pixels sampled followed by highest number of sparse'
                    'pixels sampled when training the conditional neural process')

p.add_argument('--epochs_til_ckpt', type=int, default=100,
               help='Time interval in seconds until checkpoint is saved.')
p.add_argument('--steps_til_summary', type=int, default=1000,
               help='Time interval in seconds until tensorboard summary is saved.')
p.add_argument('--iters_til_ckpt', type=int, default=10000,
               help='Training steps until save checkpoint')
p.add_argument('--dataset', type=str, default='celeba',
               help='Time interval in seconds until tensorboard summary is saved.')
p.add_argument('--type', type=str, default='none',
               help='type of loss on latents')
p.add_argument('--pretrain', action='store_true')
p.add_argument('--sparsity', type=str, default='sampled',
               help='type of sparsity to test the manifold')

p.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.')
p.add_argument('--test_latent_path', default=None, help='Checkpoint to trained latents')
opt = p.parse_args()

def dict_to_gpu(ob):
    if isinstance(ob, collections.Mapping):
        return {k: dict_to_gpu(v) for k, v in ob.items()}
    elif type(ob) == float:
        return ob
    else:
        return ob.cuda()


def sync_model(model):
    size = float(dist.get_world_size())

    for param in model.parameters():
        dist.broadcast(param.data, 0)


class OccupancyDecoder():
    def __init__(self, model):
        self.model = model
        self.model.cuda()
        self.model.eval()

    def __call__(self, samples, latent=None):
        model_input = {'context':{'idx':torch.Tensor([0]).long().cuda()[None,...],
                                  'x':samples[None,...].cuda()}}
        model_out = self.model.forward_with_latent(latent, model_input)[..., 0]
        # model_out = self.model.forward(model_input)['model_out'][..., 0]
        return model_out


def multigpu_train(gpu, opt, shared_dict, shared_mask):
    num_samples = 29000

    if opt.gpus > 1:
        dist.init_process_group(backend='nccl', init_method='tcp://localhost:6007', world_size=opt.gpus, rank=gpu)

    sidelength = 64
    train_dataset = dataio.IMNet(split='train', sampling=None)
    num_items = len(train_dataset)
    idxs = list(range(len(train_dataset)))
    random.shuffle(idxs)
    # train_dataset = dataio.IMNet(split='test', sampling=None)

    torch.cuda.set_device(gpu)


    model = high_level_models.SirenImplicitGAN(num_items=num_items, hidden_layers=3, sigmoid_output=True, type=opt.type,
                                               in_features=3, out_features=1, amortized=False, latent_dim=1024, manifold_dim=10).cuda()

    state_dict = torch.load(opt.checkpoint_path, map_location="cpu")['model_dict']
    model.load_state_dict(state_dict)

    ix = 0
    close_idxs = []


    N = 64

    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / (N - 1)


    num_samples = N ** 3
    decoder = OccupancyDecoder(model)

    max_batch = 20000

    for counter, ix in enumerate(range(9000)):
        with torch.no_grad():
            ix = idxs[ix]
            full_ctx, full_label = train_dataset[ix]

            full_ctx['context']['idx'] = full_ctx['context']['idx'][None, :]

            full_ctx = dict_to_gpu(full_ctx)
            latent = model.gen_prior_sample(full_ctx)

            head = 0

            samples = torch.zeros(N ** 3, 4)
            samples.requires_grad = False

            overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
            # transform first 3 columns
            # to be the x, y, z index
            samples[:, 2] = overall_index % N
            samples[:, 1] = (overall_index.long() / N) % N
            samples[:, 0] = ((overall_index.long() / N) / N) % N

            # transform first 3 columns
            # to be the x, y, z coordinate
            samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
            samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
            samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

            while head < num_samples:
                print(head)
                sample_subset = samples[head : min(head + max_batch, num_samples), 0:3].cuda()

                samples[head : min(head + max_batch, num_samples), 3] = (
                    decoder(sample_subset, latent=latent)
                    .squeeze()#.squeeze(1)
                    .detach()
                    .cpu()
                )
                head += max_batch

            sdf_values = samples[:, 3]
            sample = sdf_values.reshape(N, N, N).detach().cpu().numpy()

            vertices, triangle = mcubes.marching_cubes(sample, 0.5)
            vertices = (vertices.astype(np.float32) - 0.5) / 64 - 0.5

            # write_ply_triangle("gen_imnet_sample/{}_pred.ply".format(counter), vertices, triangle)

            points = sample_points_triangle(vertices, triangle, 2048)
            np.save("shape-samples/point_{}.npy".format(counter), points[..., :3])
            counter = counter + 1


    # close_idxs = np.array(close_idxs)
    # np.save("shape_close.npy", close_idxs)

if __name__ == "__main__":
    opt = p.parse_args()

    shared_array_base = multiprocessing.Array(ctypes.c_ubyte, 30000*128*128*3,
            lock=True)
    shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
    shared_array = shared_array.reshape(30000, 128, 128, 3)
    shared_array[:, :, :, :] = 0.0
    shared_array = shared_array.astype("uint8")

    shared_array_ind_base = multiprocessing.Array(ctypes.c_ubyte, 30000,
            lock=True)
    shared_array_ind = np.ctypeslib.as_array(shared_array_ind_base.get_obj())
    shared_array_ind = shared_array_ind.reshape(30000)
    shared_array_ind[:] = 0
    shared_array = shared_array.astype("uint8")

    if opt.gpus > 1:
        mp.spawn(multigpu_train, nprocs=opt.gpus, args=(opt, shared_array_base, shared_array_ind_base))
    else:
        multigpu_train(0, opt, shared_array_base, shared_array_ind_base)
