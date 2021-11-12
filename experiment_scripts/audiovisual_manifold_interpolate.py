# Enable import from parent package
import sys
import os
import collections
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
import numpy as np
import torchaudio
import matplotlib.pyplot as plt
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
from imageio import imwrite, imread, get_writer
import os.path as osp
import torchaudio
import random
from skimage.transform import resize as imresize


p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--logging_root', type=str, default='log_root', help='root for logging')
p.add_argument('--experiment_name', type=str, required=True,
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')
p.add_argument('--outdir', type=str, required=True,
               help='directory to be outputed')

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
    else:
        if type(ob) == int:
            return ob
        else:
            return ob.cuda()


def sync_model(model):
    size = float(dist.get_world_size())

    for param in model.parameters():
        dist.broadcast(param.data, 0)


def plot_audio_signal(data, lw=1.0, stem=False):
    plt.clf()
    fig, ax = plt.subplots()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    if stem:
        ax.stem(np.arange(len(data)), data, 'black', basefmt=" ", markerfmt="k,")
    else:
        ax.plot(data, 'black', lw=lw)
    ax.set_ylim((-0.8, 0.8))
    fig.savefig('/tmp/audio.png', bbox='tight', bbox_inches='tight', pad_inches=0.)
    im = imread('/tmp/audio.png')
    h, w = im.shape[0], im.shape[1]
    sf = 128 / h

    hn, wn = 128, int(sf * w)
    im = imresize(im, (hn, wn))[:, :, :3]

    return im


def multigpu_train(gpu, opt, shared_dict, shared_dict_wav, shared_mask):
    if opt.gpus > 1:
        dist.init_process_group(backend='nccl', init_method='tcp://localhost:6007', world_size=opt.gpus, rank=gpu)

    sidelength = 64
    # train_dataset = dataio.Instrument(split='train', cache=shared_dict, cache_wav=shared_dict_wav, cache_mask=shared_mask)
    num_items = 9800

    train_dataset = dataio.Instrument(split='test', cache=shared_dict, cache_wav=shared_dict_wav, cache_mask=shared_mask)
    sampling = None


    # train_generalization_dataset = dataio.AVGeneralizationWrapper(train_dataset, "sampled", sparsity_range=[1024, 1024], audio_sampling=1024, do_pad=True)
    val_generalization_dataset = dataio.AVGeneralizationWrapper(train_dataset, "full", sparsity_range=[32**2, 32**2], audio_sampling=sampling, do_pad=True)

    model = high_level_models.SirenImplicitGAN(num_items=num_items, hidden_layers=3, pos_encode=True, tanh_output=True, type=opt.type,
                                               in_features=2, out_features=3, amortized=False, latent_dim=1024, manifold_dim=10, audiovisual=True).cuda()

    torch.cuda.set_device(gpu)

    state_dict = torch.load(opt.checkpoint_path, map_location="cpu")['model_dict']
    model.load_state_dict(state_dict)

    # Define the loss
    root_path = os.path.join(opt.logging_root, opt.experiment_name)
    val_loss_fn = loss_functions.image_mse

    ix = 0

    mses = []
    psnrs = []

    data = np.load("instrument.npz")

    mean = data['mean'][:-1]
    std = data['std'][:-1]

    if not osp.exists(opt.outdir):
        os.makedirs(opt.outdir)

    counter = 0

    latents = state_dict['latents.weight']
    ix = 0

    writer = get_writer("animation.mp4")

    while True:
        with torch.no_grad():
            dist = torch.norm(latents - latents[ix:ix+1], p=2, dim=-1)
            idx = random.randint(1, 100)
            idx_other = torch.sort(dist)[1][idx].item()

            latent = model.latents.weight[ix]
            latent_other = model.latents.weight[idx_other]

            interval = torch.linspace(0, 1.0, 10).cuda()
            diff = latent_other - latent

            latent = latent[None, :] + interval[:, None] * diff[None, :]

            full_ctx, full_label = val_generalization_dataset[0]

            full_ctx = dict_to_gpu(full_ctx)

            model_output = model.forward_with_latent(latent, full_ctx)

            pred_rgb, pred_wav = model_output

            pred_rgb = pred_rgb.detach().cpu().numpy()
            pred_wav = pred_wav.detach().cpu().numpy()

            pred_rgb = pred_rgb.reshape((10, 128, 128, 3))

            pred_wav = pred_wav.reshape((10, 200, 41)) * 3 * std[None, :, None] + mean[None, :, None]
            pred_wav = np.exp(pred_wav)

            pred_wav = torch.Tensor(pred_wav)

            pred_wav = torch.cat([pred_wav, torch.zeros_like(pred_wav[:, -1:, :])], dim=1)
            pred_wav = torchaudio.transforms.GriffinLim()(pred_wav)[None, :] * 10

            for i in range(10):
                im_i = (pred_rgb[i] + 1) / 2.
                wav_i = pred_wav[0, i]

                if i == 0:
                    torchaudio.save("audio_{}.wav".format(counter), torch.Tensor(wav_i[None, :]), 16000)

                wav_i = plot_audio_signal(wav_i)


                panel_im = np.concatenate([im_i, wav_i], axis=1)
                writer.append_data(panel_im)
                counter = counter + 1

            ix = idx_other
            print(counter)

        if counter > 100:
            writer.close()
            assert False
            break




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
