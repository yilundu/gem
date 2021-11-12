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

from torch import nn
import torch.nn.functional as F
from torch_ema import ExponentialMovingAverage

p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--logging_root', type=str, default='log_root', help='root for logging')
p.add_argument('--experiment_name', type=str, required=True,
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')

# General training options
p.add_argument('--gpus', type=int, default=1)
p.add_argument('--batch_size', type=int, default=96)
p.add_argument('--lr', type=float, default=5e-5, help='learning rate. default=5e-5')
p.add_argument('--num_epochs', type=int, default=40001,
               help='Number of epochs to train for.')
p.add_argument('--sampling', type=int, default=[32000],
               help='Number of timesteps to downsample each audio waveform to')
p.add_argument('--epochs_til_ckpt', type=int, default=100,
               help='Time interval in seconds until checkpoint is saved.')
p.add_argument('--steps_til_summary', type=int, default=100,
               help='Time interval in seconds until tensorboard summary is saved.')
p.add_argument('--iters_til_ckpt', type=int, default=10000,
               help='Training steps until save checkpoint')
p.add_argument('--dataset', type=str, default='nsynth',
               help='Dataset to use for training and if included, validation.')
p.add_argument('--pretrain', action='store_true')
p.add_argument('--resample_rate',default=16000,
               help="Rate to resample audio signals. None uses default sample rate.")
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
        print("creating more gpus: ", gpu)
        dist.init_process_group(backend='nccl', init_method='tcp://localhost:1492', world_size=opt.gpus, rank=gpu)

    use_subset = False

    lr = opt.lr
    # sampling = 32000#opt.sampling
    sampling = 2048#opt.sampling
    # resample_rate = opt.resample_rate
    resample_rate = 16000

    if opt.dataset == "nsynth":
        train_audio_dataset = dataio.NSynth(split='train', resample_rate=resample_rate)
        val_audio_dataset = dataio.NSynth(split='train', resample_rate=resample_rate) # keep same split for indices
    elif opt.dataset == "spoken_digits":
        train_audio_dataset = dataio.NSynth(split='train', resample_rate=resample_rate)
        val_audio_dataset = dataio.NSynth(split='train', resample_rate=resample_rate)
    else:
        print("Please use an already-implemented audio dataset (i.e., nsynth, spoken_digits.")

    train_generalization_dataset = dataio.AudioGeneralizationWrapper(train_audio_dataset, sampling=4096, do_pad=True)
    if use_subset:
        num_samples = int(len(train_audio_dataset) * 0.25)
        train_generalization_dataset = torch.utils.data.Subset(train_generalization_dataset, np.arange(num_samples))

    train_loader = DataLoader(train_generalization_dataset, shuffle=True, batch_size=opt.batch_size, pin_memory=True,
                              num_workers=4)

    val_generalization_dataset = dataio.AudioGeneralizationWrapper(val_audio_dataset, sampling=None,do_pad=True)
    val_loader = DataLoader(val_generalization_dataset, shuffle=True, batch_size=4, pin_memory=True,
                              num_workers=4)

    print("data set size: ", len(train_generalization_dataset))
    print("val data set size: ", len(val_generalization_dataset))

    torch.cuda.set_device(gpu)
    model = high_level_models.SirenImplicitGAN(num_items=len(train_generalization_dataset), hidden_layers=3, hidden_features=512,
                                               share_first_layer=False,
                                               in_features=1, out_features=1, amortized=False, latent_dim=1024,first_omega_0=30, manifold_dim=10, type=opt.type).cuda()

    if opt.checkpoint_path is not None:
        model.load_state_dict(torch.load(opt.checkpoint_path, map_location="cpu")['model_dict'])

    if opt.gpus > 1:
        sync_model(model)

    # Define the loss
    root_path = os.path.join(opt.logging_root, opt.experiment_name)
    ema_model = ExponentialMovingAverage(model.parameters(), decay=0.9999)

    summary_fn = summaries.audio
    val_loss_fn = loss_functions.audio_mse

    if opt.type == "linear":
        loss_fn = loss_functions.audio_mse_linear
    elif opt.type == "linear_lle":
        loss_fn = loss_functions.audio_mse_linear
    elif opt.type == "none":
        loss_fn = loss_functions.audio_mse

    training.train(model=model, ema_model=ema_model, train_dataloader=train_loader, val_dataloader=val_loader, epochs=opt.num_epochs,
                   lr=lr, steps_til_summary=opt.steps_til_summary, epochs_til_checkpoint=opt.epochs_til_ckpt,
                   model_dir=root_path, loss_fn=loss_fn, iters_til_checkpoint=opt.iters_til_ckpt, summary_fn=summary_fn,
                   clip_grad=False, val_loss_fn=val_loss_fn, overwrite=True, gpus=opt.gpus, rank=gpu)


if __name__ == "__main__":
    opt = p.parse_args()
    if opt.gpus > 1:
        mp.spawn(multigpu_train, nprocs=opt.gpus, args=(opt,))
    else:
        multigpu_train(0, opt)
