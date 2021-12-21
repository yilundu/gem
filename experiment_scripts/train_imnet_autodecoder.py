# Enable import from parent package
import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

import dataio, meta_modules, summaries, loss_functions, modules, training, config
import torch.distributed as dist

import high_level_models
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
import configargparse
import numpy as np
from torch_ema import ExponentialMovingAverage

p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--logging_root', type=str, default=os.path.join(config.log_root, 'latent_gan'), help='root for logging')
p.add_argument('--experiment_name', type=str, required=True,
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')

# General training options
p.add_argument('--gpus', type=int, default=1)
p.add_argument('--batch_size', type=int, default=16)
p.add_argument('--lr', type=float, default=5e-5, help='learning rate. default=5e-5')
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
p.add_argument('--type', type=str, default='linear_lle',
               help='type of loss on latents')

p.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.')
opt = p.parse_args()


def sync_model(model):
    size = float(dist.get_world_size())

    for param in model.parameters():
        dist.broadcast(param.data, 0)


def multigpu_train(gpu, opt):
    if opt.gpus > 1:
        dist.init_process_group(backend='nccl', init_method='tcp://localhost:1492', world_size=opt.gpus, rank=gpu)

    train_generalization_dataset = dataio.IMNet(split='train', sampling=4096)
    train_loader = DataLoader(train_generalization_dataset, shuffle=True, batch_size=opt.batch_size, pin_memory=True,
                              num_workers=2)

    torch.cuda.set_device(gpu)
    model = high_level_models.SirenImplicitGAN(num_items=len(train_generalization_dataset), hidden_layers=3, hidden_features=256,
                                               in_features=3, out_features=1, amortized=False, latent_dim=1024, sigmoid_output=True, manifold_dim=10, type=opt.type, pos_encode=True).cuda()

    if opt.checkpoint_path is not None:
        model.load_state_dict(torch.load(opt.checkpoint_path, map_location="cpu")['model_dict'])

    if opt.gpus > 1:
        sync_model(model)

    ema_model = ExponentialMovingAverage(model.parameters(), decay=0.9999)

    # Define the loss
    root_path = os.path.join(opt.logging_root, opt.experiment_name)
    summary_fn = summaries.imnet
    val_loss_fn = loss_functions.occupancy

    if opt.type == "linear":
        loss_fn = loss_functions.occupancy_linear
    elif opt.type == "linear_lle":
        loss_fn = loss_functions.occupancy_linear
    elif opt.type == "none":
        loss_fn = loss_functions.occupancy

    training.train(model=model, ema_model=ema_model, train_dataloader=train_loader, val_dataloader=None, epochs=opt.num_epochs,
                   lr=opt.lr, steps_til_summary=opt.steps_til_summary, epochs_til_checkpoint=opt.epochs_til_ckpt,
                   model_dir=root_path, loss_fn=loss_fn, iters_til_checkpoint=opt.iters_til_ckpt, summary_fn=summary_fn,
                   clip_grad=False, val_loss_fn=val_loss_fn, overwrite=True, gpus=opt.gpus, rank=gpu)


if __name__ == "__main__":
    opt = p.parse_args()
    if opt.gpus > 1:
        mp.spawn(multigpu_train, nprocs=opt.gpus, args=(opt,))
    else:
        multigpu_train(0, opt)
