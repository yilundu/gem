# Enable import from parent package
import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

import dataio, meta_modules, summaries, loss_functions, modules, training, config
import torch.distributed as dist
import multiprocessing
import ctypes

import high_level_models
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
import configargparse
import numpy as np

from torch import nn
import torch.nn.functional as F
from torch_ema import ExponentialMovingAverage
from config import log_root

p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--logging_root', type=str, default='log_root', help='root for logging')
p.add_argument('--experiment_name', type=str, required=True,
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')

# General training options
p.add_argument('--gpus', type=int, default=1)
p.add_argument('--batch_size', type=int, default=64)
p.add_argument('--lr', type=float, default=1e-4, help='learning rate. default=5e-5')
p.add_argument('--num_epochs', type=int, default=40001,
               help='Number of epochs to train for.')
p.add_argument('--train_sparsity_range', type=int, nargs='+', default=[64**2, 64**2],
               help='Two integers: lowest number of sparse pixels sampled followed by highest number of sparse'
                    'pixels sampled when training the conditional neural process')

p.add_argument('--epochs_til_ckpt', type=int, default=100,
               help='Time interval in seconds until checkpoint is saved.')
p.add_argument('--steps_til_summary', type=int, default=100,
               help='Time interval in seconds until tensorboard summary is saved.')
p.add_argument('--iters_til_ckpt', type=int, default=10000,
               help='Training steps until save checkpoint')
p.add_argument('--dataset', type=str, default='celeba',
               help='Time interval in seconds until tensorboard summary is saved.')
p.add_argument('--pretrain', action='store_true')
p.add_argument('--split', type=str, default='train')
p.add_argument('--type', type=str, default='linear_lle',
               help='type of loss on latents')
p.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.')
p.add_argument('--index_path', default=None, help='Path to inidices to use for subsampling autodecoder')

opt = p.parse_args()

def sync_model(model):
    size = float(dist.get_world_size())

    for param in model.parameters():
        dist.broadcast(param.data, 0)


def multigpu_train(gpu, opt, shared_dict, shared_dict_wav, shared_mask):
    if opt.gpus > 1:
        print("creating more gpus: ", gpu)
        dist.init_process_group(backend='nccl', init_method='tcp://localhost:1495', world_size=opt.gpus, rank=gpu)

    root_path = os.path.join(opt.logging_root, opt.experiment_name)

    use_subset = False

    lr = opt.lr
    sampling = None #8000
    sidelen = 32
    img_sampling = "full"

    train_dataset = dataio.Instrument(split=opt.split, cache=shared_dict, cache_wav=shared_dict_wav, cache_mask=shared_mask)
    train_generalization_dataset = dataio.AVGeneralizationWrapper(train_dataset, "sampled", sparsity_range=[1024, 1024], audio_sampling=1024, do_pad=True)
    val_generalization_dataset = dataio.AVGeneralizationWrapper(train_dataset, "full", sparsity_range=[32**2, 32**2], audio_sampling=sampling, do_pad=True)
    if use_subset:
        if opt.index_path is not None: # use previously sampled examples
            indices = np.load(opt.index_path)
        else:
            num_sample = 1000
            indices = np.random.choice(np.arange(num_sample), num_sample, replace=False)
            with open(root_path + '_rand_indices.npy', 'wb') as f:
                np.save(f, indices)
        train_generalization_dataset = torch.utils.data.Subset(train_generalization_dataset, indices)
        val_generalization_dataset = torch.utils.data.Subset(val_generalization_dataset, indices)

    train_loader = DataLoader(train_generalization_dataset, shuffle=True, batch_size=opt.batch_size, pin_memory=True,
                              num_workers=8)
    val_loader = DataLoader(val_generalization_dataset, shuffle=True, batch_size=1, pin_memory=True, num_workers=2)

    torch.cuda.set_device(gpu)

    print("data set size: ", len(train_generalization_dataset))


    model = high_level_models.SirenImplicitGAN(num_items=len(train_generalization_dataset), hidden_layers=3, pos_encode=True, tanh_output=True, type=opt.type,
                                               in_features=2, out_features=3, amortized=False, latent_dim=1024, manifold_dim=10, audiovisual=True).cuda()


    if opt.checkpoint_path is not None:
        model.load_state_dict(torch.load(opt.checkpoint_path)['model_dict'])

    if opt.gpus > 1:
        sync_model(model)

    ema_model = ExponentialMovingAverage(model.parameters(), decay=0.9999)

    # Define the loss

    summary_fn = summaries.audio_visual
    val_loss_fn = loss_functions.audio_visual_mse

    if opt.type == "linear":
        loss_fn = loss_functions.audio_visual_mse_linear
    elif opt.type == "linear_lle":
        loss_fn = loss_functions.audio_visual_mse_linear
    elif opt.type == "none":
        loss_fn = loss_functions.audio_visual_mse


    training.train(model=model, ema_model=ema_model, train_dataloader=train_loader, val_dataloader=val_loader, epochs=opt.num_epochs,
                   lr=lr, steps_til_summary=opt.steps_til_summary, epochs_til_checkpoint=opt.epochs_til_ckpt,
                   model_dir=root_path, loss_fn=loss_fn, iters_til_checkpoint=opt.iters_til_ckpt, summary_fn=summary_fn,
                   clip_grad=False, val_loss_fn=val_loss_fn, overwrite=True, gpus=opt.gpus, rank=gpu)


if __name__ == "__main__":
    opt = p.parse_args()

    shared_array_base = multiprocessing.Array(ctypes.c_ubyte, 9800*128*128*3,
            lock=True)
    shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
    shared_array = shared_array.reshape(9800, 128, 128, 3)
    shared_array[:, :, :, :] = 0.0
    shared_array = shared_array.astype("uint8")

    shared_array_ind_base = multiprocessing.Array(ctypes.c_ubyte, 9800,
            lock=True)
    shared_array_ind = np.ctypeslib.as_array(shared_array_ind_base.get_obj())
    shared_array_ind = shared_array_ind.reshape(9800)
    shared_array_ind[:] = 0
    shared_array = shared_array.astype("uint8")


    shared_array_base_wav = multiprocessing.Array(ctypes.c_float, 9800*200*41,
            lock=True)
    shared_array = np.ctypeslib.as_array(shared_array_base_wav.get_obj())
    shared_array = shared_array.reshape(9800, 200, 41)
    shared_array[:, :] = 0.0
    shared_array = shared_array.astype("float32")

    if opt.gpus > 1:
        mp.spawn(multigpu_train, nprocs=opt.gpus, args=(opt, shared_array_base, shared_array_base_wav, shared_array_ind_base))
    else:
        multigpu_train(0, opt, shared_array_base, shared_array_base_wav, shared_array_ind_base)
