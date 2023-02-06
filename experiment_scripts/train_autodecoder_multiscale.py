# Enable import from parent package
import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

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

p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--logging_root', type=str, default='log_root', help='root for logging')
p.add_argument('--experiment_name', type=str, required=True,
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')

# General training options
p.add_argument('--gpus', type=int, default=1)
p.add_argument('--batch_size', type=int, default=512)
p.add_argument('--lr', type=float, default=1e-4, help='learning rate. default=5e-5')
p.add_argument('--num_epochs', type=int, default=40001,
               help='Number of epochs to train for.')

p.add_argument('--epochs_til_ckpt', type=int, default=100,
               help='Time interval in seconds until checkpoint is saved.')
p.add_argument('--steps_til_summary', type=int, default=1000,
               help='Time interval in seconds until tensorboard summary is saved.')
p.add_argument('--iters_til_ckpt', type=int, default=10000,
               help='Training steps until save checkpoint')
p.add_argument('--dataset', type=str, default='celeba',
               help='Time interval in seconds until tensorboard summary is saved.')
p.add_argument('--type', type=str, default='linear_lle',
               help='type of loss on latents')
p.add_argument('--sparsity', type=str, default='sampled',
               help='type of sparsity to test the manifold')
p.add_argument('--pretrain', action='store_true')

p.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.')
opt = p.parse_args()



def sync_model(model):
    size = float(dist.get_world_size())

    for param in model.parameters():
        dist.broadcast(param.data, 0)


def multigpu_train(gpu, opt, shared_dict, shared_mask):
    num_samples = 29000

    if opt.gpus > 1:
        dist.init_process_group(backend='nccl', init_method='tcp://localhost:6007', world_size=opt.gpus, rank=gpu)

    def create_dataloader_callback(sidelength, batch_size):
        train_img_dataset = dataio.CelebAHQ(sidelength, cache=shared_dict, cache_mask=shared_mask)
        sparsity_range = min(256, sidelength**2)
        train_generalization_dataset = dataio.GeneralizationWrapper(train_img_dataset, context_sparsity=opt.sparsity,
                                                                    query_sparsity=opt.sparsity,
                                                                    sparsity_range=(sparsity_range, sparsity_range),
                                                                    persistent_mask=False, inner_loop_supervision_key='rgb')
        train_generalization_dataset = torch.utils.data.Subset(train_generalization_dataset, np.arange(num_samples))
        train_loader = DataLoader(train_generalization_dataset, shuffle=True, batch_size=batch_size,
                                  pin_memory=True, num_workers=6)

        val_dataset = dataio.GeneralizationWrapper(train_img_dataset, context_sparsity='full', query_sparsity='full',
                                                   sparsity_range=(0, 50), inner_loop_supervision_key='rgb')
        val_dataset = torch.utils.data.Subset(val_dataset, np.arange(num_samples))
        val_loader = DataLoader(val_dataset, shuffle=True, batch_size=1, pin_memory=True, num_workers=1)

        return train_loader, val_loader

    torch.cuda.set_device(gpu)

    # manifold_dim controls the number of nearest neighbors used for computing the linear_lle loss
    # This is unused with linear_lle is set to be false
    model = high_level_models.SirenImplicitGAN(num_items=num_samples, hidden_layers=3,
                                               latent_dim=1024, tanh_output=True, type=opt.type, manifold_dim=10).cuda()

    if opt.checkpoint_path is not None:
        model.load_state_dict(torch.load(opt.checkpoint_path, map_location="cpu")['model_dict'])

    if opt.gpus > 1:
        sync_model(model)

    ema_model = ExponentialMovingAverage(model.parameters(), decay=0.9999)

    # Define the loss
    summary_fn = summaries.film
    root_path = os.path.join(opt.logging_root, opt.experiment_name)
    val_loss_fn = loss_functions.image_mse

    if opt.type == "linear":
        loss_fn = loss_functions.image_mse_linear
    elif opt.type == "linear_lle":
        loss_fn = loss_functions.image_mse_linear
    elif opt.type == "spectral_loss":
        loss_fn = loss_functions.image_mse_spectral
    elif opt.type == "nn":
        loss_fn = loss_functions.image_mse_manifold
    elif opt.type == "none":
        loss_fn = loss_functions.image_mse

    training.multiscale_training(model=model, ema_model=ema_model, dataloader_callback=create_dataloader_callback, dataloader_iters=(10000000,),
                                 dataloader_params=((64, 128),),
                                 lr=opt.lr, steps_til_summary=opt.steps_til_summary, epochs_til_checkpoint=opt.epochs_til_ckpt,
                                 model_dir=root_path, loss_fn=loss_fn, iters_til_checkpoint=opt.iters_til_ckpt, summary_fn=summary_fn,
                                 clip_grad=False, val_loss_fn=val_loss_fn, overwrite=True, gpus=opt.gpus, rank=gpu)

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
