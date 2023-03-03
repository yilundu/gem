# Enable import from parent package
import sys
import os
import collections
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
import numpy as np

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
import os

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
p.add_argument('--outdir', type=str, default='output_1',
               help='directory of output')
p.add_argument('--pretrain', action='store_true')
p.add_argument('--sparsity', type=str, default='sampled',
               help='type of sparsity to test the manifold')

p.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.')
p.add_argument('--test_latent_path', default=None, help='Checkpoint to trained latents')
opt = p.parse_args()

def dict_to_gpu(ob):
    if isinstance(ob, collections.Mapping):
        return {k: dict_to_gpu(v) for k, v in ob.items()}
    else:
        return ob.cuda()


def sync_model(model):
    size = float(dist.get_world_size())

    for param in model.parameters():
        dist.broadcast(param.data, 0)


def multigpu_train(gpu, opt, shared_dict, shared_mask):
    num_samples = 29000

    if opt.gpus > 1:
        dist.init_process_group(backend='nccl', init_method='tcp://localhost:6007', world_size=opt.gpus, rank=gpu)

    sidelength = 64
    train_img_dataset = dataio.CelebAHQ(sidelength, cache=shared_dict, cache_mask=shared_mask, split='test')
    train_generalization_dataset = dataio.GeneralizationWrapper(train_img_dataset, context_sparsity='half',
                                                                query_sparsity='half',
                                                                persistent_mask=False, inner_loop_supervision_key='rgb')

    train_generalization_dataset = torch.utils.data.Subset(train_generalization_dataset, np.arange(num_samples))

    val_dataset = dataio.GeneralizationWrapper(train_img_dataset, context_sparsity='full', query_sparsity='full',
                                               sparsity_range=(0, 50), inner_loop_supervision_key='rgb')
    val_dataset = torch.utils.data.Subset(val_dataset, np.arange(num_samples))

    torch.cuda.set_device(gpu)
    model = high_level_models.SirenImplicitGAN(num_items=num_samples, test_num_items=1000, hidden_layers=3,
                                               amortized=False, latent_dim=1024, noise=True, pos_encode=True, tanh_output=True, type=opt.type, manifold_dim=10).cuda()

    state_dict = torch.load(opt.checkpoint_path, map_location="cpu")['model_dict']
    model.load_state_dict(state_dict)

    # Define the loss
    root_path = os.path.join(opt.logging_root, opt.experiment_name)
    val_loss_fn = loss_functions.image_mse

    ix = 0

    mses = []
    psnrs = []

    if not os.path.exists(opt.outdir):
        os.makedirs(opt.outdir)

    with torch.no_grad():
        for ix in range(300):
            half_ctx, half_label = train_generalization_dataset[ix]
            full_ctx, full_label = val_dataset[ix]

            full_ctx['context']['idx'] = full_ctx['context']['idx'][None, :]

            full_ctx = dict_to_gpu(full_ctx)
            output = model(full_ctx)
            rgb = output['rgb'].view(64, 64, 3).detach().cpu().numpy()
            rgb_label = full_label['rgb'].view(64, 64, 3).detach().cpu().numpy()

            rgb_label_copy = rgb_label.copy()
            rgb_label_copy[16:48, 16:48] = 0.5

            imwrite("{}/pred_{:03d}.png".format(opt.outdir, ix), rgb)
            imwrite("{}/gt_{:03d}.png".format(opt.outdir, ix), rgb_label)
            imwrite("{}/complete_{:03d}.png".format(opt.outdir, ix), rgb_label_copy)



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
