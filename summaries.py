# do not use gui
import matplotlib as mpl
mpl.use('Agg')

import utils
from torchaudio.transforms import InverseMelScale

import pickle
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from torchvision.utils import make_grid, save_image
import skimage.measure
import skimage.metrics
import shutil
import imageio
import random



class CompSegmentationSummarizer():
    def __init__(self, color_map):
        self.color_map = color_map

    def __call__(self, model, model_input, gt, model_output, writer, total_steps, prefix='train_'):
        gt_sem = utils.lin2img(gt['semantic'])

        sidelen = gt_sem.size()[-1]
        meta_batch_size = gt_sem.shape[0]

        pred_sem_logits = utils.lin2img(model_output['semantic'])
        _, pred_sem = torch.max(pred_sem_logits, dim=1)
        pred_sem = torch.reshape(pred_sem, torch.Size([meta_batch_size, 1, sidelen, sidelen]))

        output_vs_gt = torch.cat(
            (utils.convert_int2color(gt_sem, self.color_map), utils.convert_int2color(pred_sem, self.color_map)), dim=-1)
        writer.add_image(prefix + 'gt_vs_pred_semantic', make_grid(output_vs_gt, scale_each=False, normalize=False),
                         global_step=total_steps)

        gt_img = utils.lin2img(gt['rgb'])
        pred_img = utils.lin2img(model_output['rgb'])
        output_vs_gt = torch.cat((gt_img, pred_img), dim=-1)
        writer.add_image(prefix + 'gt_vs_pred_img', make_grid(output_vs_gt, scale_each=False, normalize=True),
                         global_step=total_steps)

        saliency_masks = model_output['saliency_masks'][0, :, 0][:, None, ...]
        writer.add_image(prefix + 'saliency_masks', make_grid(saliency_masks, scale_each=True, normalize=True), global_step=total_steps)

        pred_instance_mask_logits = model_output['instance_masks'].squeeze()
        _, pred_instance = torch.max(pred_instance_mask_logits, dim=1)
        pred_instance = torch.reshape(pred_instance, torch.Size([meta_batch_size, 1, sidelen, sidelen]))
        writer.add_image(prefix + 'pred_instance', make_grid(pred_instance, scale_each=False, normalize=False),
                         global_step=total_steps)

        model_log = model_output.get('log', list())
        for type, name, value in model_log:
            if type == 'scalar':
                writer.add_scalar(prefix + name, value, total_steps)
            elif type == 'model_output':
                img = utils.lin2img(value)
                writer.add_image(prefix + name, make_grid(img, scale_each=False, normalize=True),
                                 global_step=total_steps)

        min_max_summary(prefix + 'img_model_out_min_max', pred_img, writer, total_steps)
        min_max_summary(prefix + 'img_gt_min_max', gt_img, writer, total_steps)

        min_max_summary(prefix + 'semantic_model_out_min_max', pred_sem_logits, writer, total_steps)
        min_max_summary(prefix + 'semantic_gt_min_max', gt_sem, writer, total_steps)

        min_max_summary(prefix + 'coords_min_max', gt['x'], writer, total_steps)
        min_max_summary(prefix + 'lm_coords_min_max', model_output['lm_coords'], writer, total_steps)


class SegmentationSummarizer():
    def __init__(self, color_map):
        self.color_map = color_map

    def __call__(self, model, model_input, gt, model_output, writer, total_steps, prefix='train_'):
        gt_sem = utils.lin2img(gt['semantic'])

        sidelen = gt_sem.size()[-1]
        meta_batch_size = gt_sem.shape[0]

        pred_sem_logits = utils.lin2img(model_output['semantic'])
        _, pred_sem = torch.max(pred_sem_logits, dim=1)
        pred_sem = torch.reshape(pred_sem, torch.Size([meta_batch_size, 1, sidelen, sidelen]))

        output_vs_gt = torch.cat(
            (utils.convert_int2color(gt_sem, self.color_map), utils.convert_int2color(pred_sem, self.color_map)), dim=-1)
        writer.add_image(prefix + 'gt_vs_pred_semantic', make_grid(output_vs_gt, scale_each=False, normalize=False),
                         global_step=total_steps)

        gt_img = utils.lin2img(gt['rgb'])
        pred_img = utils.lin2img(model_output['rgb'])
        output_vs_gt = torch.cat((gt_img, pred_img), dim=-1)
        writer.add_image(prefix + 'gt_vs_pred_img', make_grid(output_vs_gt, scale_each=False, normalize=True),
                         global_step=total_steps)

        # fig, ax = plt.subplots()
        # coords = model_input['context']['x']
        # x, y = coords[0, ..., 0].detach().cpu().numpy(), coords[0, ..., 1].detach().cpu().numpy()
        # color = model_input['context']['y'][0].detach().cpu().numpy()
        # color = (color - np.amin(color)) / (np.amax(color) - np.amin(color))
        #
        # ax.scatter(y, x, c=color)
        # writer.add_figure(prefix + 'conditioning_points', fig, global_step=total_steps)

        model_log = model_output.get('log', list())
        for type, name, value in model_log:
            if type == 'scalar':
                writer.add_scalar(prefix + name, value, total_steps)
            elif type == 'model_output':
                img = utils.lin2img(value)
                writer.add_image(prefix + name, make_grid(img, scale_each=False, normalize=True),
                                 global_step=total_steps)

        min_max_summary(prefix + 'img_model_out_min_max', pred_img, writer, total_steps)
        min_max_summary(prefix + 'img_gt_min_max', gt_img, writer, total_steps)

        min_max_summary(prefix + 'semantic_model_out_min_max', pred_sem_logits, writer, total_steps)
        min_max_summary(prefix + 'semantic_gt_min_max', gt_sem, writer, total_steps)

        min_max_summary(prefix + 'coords_min_max', gt['x'], writer, total_steps)


class EgoSRNSummarizer():
    def __init__(self, num_context, num_trgt, sidelength):
        self.num_context = num_context
        self.num_trgt = num_trgt
        self.sidelength = sidelength

    def __call__(self, model, model_input, gt, model_output, writer, total_steps, prefix='train_'):
        context_rgb = model_input['context']['imgs'][0]
        trgt_gt_rgb = model_input['query']['imgs'][0]

        trgt_pred_rgb = torch.split(model_output['model_out'][0], int(self.sidelength**2), dim=0)
        trgt_pred_rgb = torch.stack(trgt_pred_rgb, dim=0)
        trgt_pred_rgb = utils.lin2img(trgt_pred_rgb)

        output_vs_gt = torch.cat((trgt_gt_rgb, trgt_pred_rgb), dim=-1)
        writer.add_image(prefix + 'gt_vs_pred', make_grid(output_vs_gt, scale_each=False, normalize=True),
                         global_step=total_steps)
        writer.add_image(prefix + 'context', make_grid(context_rgb, scale_each=False, normalize=True),
                         global_step=total_steps)

        model_depth = model_output['query_ego_out']['depth']
        writer.add_image(prefix + 'query_depth', make_grid(model_depth[0], scale_each=False, normalize=True),
                         global_step=total_steps)

        model_log = model_output.get('log', list())
        for type, name, value in model_log:
            if type == 'scalar':
                writer.add_scalar(prefix + name, value, total_steps)
            elif type == 'model_output':
                img = utils.lin2img(value)
                writer.add_image(prefix + name, make_grid(img, scale_each=False, normalize=True),
                                 global_step=total_steps)

        min_max_summary(prefix + 'query_depth', model_depth, writer, total_steps)
        min_max_summary(prefix + 'model_out_min_max', model_output['model_out'], writer, total_steps)
        min_max_summary(prefix + 'gt_min_max', gt['y'], writer, total_steps)


class RaySRNSummarizer():
    def __init__(self, num_context, num_trgt, sidelength):
        self.num_context = num_context
        self.num_trgt = num_trgt
        self.sidelength = sidelength

    def __call__(self, model, model_input, gt, model_output, writer, total_steps, prefix='train_'):
        context_rgb = torch.split(model_input['context']['y'][0], int(self.sidelength**2), dim=0)
        context_rgb = torch.stack(context_rgb, dim=0)
        context_rgb = utils.lin2img(context_rgb)

        trgt_gt_rgb = torch.stack(torch.split(model_input['query']['y'][0], int(self.sidelength**2), dim=0), dim=0)
        trgt_gt_rgb = utils.lin2img(trgt_gt_rgb)

        trgt_pred_rgb = torch.stack(torch.split(model_output['model_out'][0], int(self.sidelength**2), dim=0), dim=0)
        trgt_pred_rgb = utils.lin2img(trgt_pred_rgb)

        output_vs_gt = torch.cat((trgt_gt_rgb, trgt_pred_rgb), dim=-1)
        writer.add_image(prefix + 'gt_vs_pred', make_grid(output_vs_gt, scale_each=False, normalize=True),
                         global_step=total_steps)

        writer.add_image(prefix + 'context', make_grid(context_rgb, scale_each=False, normalize=True),
                         global_step=total_steps)

        model_log = getattr(model, 'log', {})
        for key in model_log.keys():
            writer.add_scalar(prefix + key, model_log[key], total_steps)

        min_max_summary(prefix + 'model_out_min_max', model_output['model_out'], writer, total_steps)
        min_max_summary(prefix + 'gt_min_max', gt['y'], writer, total_steps)

def summarize_latents(writer, model, model_output, total_steps, prefix="train_", n_iter_step=10000, add_statistics=True):

    # patch needed only if env has tensorflow
    # help from: https://github.com/pytorch/pytorch/issues/47139
    try:
        import tensorflow as tf
        import tensorboard as tb
        tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
    except ImportError as e:
        pass

    # add latent summary
    latents = model.latents.weight[:10000]
    if total_steps % n_iter_step == 0: writer.add_embedding(mat=latents, global_step=total_steps) # only save embeddings every k steps
    # if add_statistics:
    #     latents = latents.detach().cpu().numpy()
    #     representation = model_output["representation"].detach().cpu().numpy()
    #     for name, value in [("latents", latents), ("repr",representation)]:
    #         writer.add_scalar(prefix + name + "_stdev", np.std(value), total_steps)
    #         writer.add_scalar(prefix + name + "_mean", np.mean(value), total_steps)
    #         writer.add_histogram(prefix + name, value, total_steps)

def images(model, model_input, gt, model_output, writer, total_steps, prefix='train_'):
    # Full forward pass with full mgrid
    # Plot context

    # print("prefix: ", prefix)
    if prefix != "train_":
        pred_img = utils.lin2img(model_output['rgb'])
        gt_img = utils.lin2img(gt['rgb'])

        output_vs_gt = torch.cat((gt_img, pred_img), dim=-1)
        writer.add_image(prefix + 'gt_vs_pred', make_grid(output_vs_gt, scale_each=False, normalize=True),
                         global_step=total_steps)

    fig, ax = plt.subplots()
    coords = model_input['context']['x']
    x, y = coords[0, ..., 0].detach().cpu().numpy(), coords[0, ..., 1].detach().cpu().numpy()
    color = model_input['context']['y'][0].detach().cpu().numpy()
    color = (color - np.amin(color)) / (np.amax(color) - np.amin(color))
    mask = model_input['context']['mask'][0].detach().cpu().numpy() == 1
    mask = mask.squeeze()
    ax.scatter(y[mask], x[mask], c=color[mask])
    writer.add_figure(prefix + 'conditioning_points', fig, global_step=total_steps)

    coords = model_input['context']['x']


    model_log = model_output.get('log', list())
    for type, name, value in model_log:
        if type == 'scalar':
            writer.add_scalar(prefix + name, value, total_steps)
        elif type == 'model_output':
            img = utils.lin2img(value)
            writer.add_image(prefix + name, make_grid(img, scale_each=False, normalize=True),
                             global_step=total_steps)

    min_max_summary(prefix + 'model_out_min_max', model_output['model_out'], writer, total_steps)
    min_max_summary(prefix + 'gt_min_max', gt['rgb'], writer, total_steps)
    min_max_summary(prefix + 'coords_min_max', gt['x'], writer, total_steps)

def images_original(model, model_input, gt, model_output, writer, total_steps, prefix='train_'):
    # Full forward pass with full mgrid
    # Plot context
    gt_img = utils.lin2img(gt['rgb'])
    pred_img = utils.lin2img(model_output['rgb'])

    output_vs_gt = torch.cat((gt_img, pred_img), dim=-1)
    writer.add_image(prefix + 'gt_vs_pred', make_grid(output_vs_gt, scale_each=False, normalize=True),
                     global_step=total_steps)

    fig, ax = plt.subplots()
    coords = model_input['context']['x']
    x, y = coords[0, ..., 0].detach().cpu().numpy(), coords[0, ..., 1].detach().cpu().numpy()
    color = model_input['context']['y'][0].detach().cpu().numpy()
    color = (color - np.amin(color)) / (np.amax(color) - np.amin(color))
    mask = model_input['context']['mask'][0].detach().cpu().numpy() == 1
    mask = mask.squeeze()
    print("mask: ", mask.shape, " y: ", y.shape, " x: ", x.shape, " color: ",color.shape)
    ax.scatter(y[mask], x[mask], c=color.squeeze()[mask])
    writer.add_figure(prefix + 'conditioning_points', fig, global_step=total_steps)

    # for param_name, init_param, fast_param in zip(model_output['fast_params'].keys(), model.hypo_module.meta_parameters(), model_output['fast_params'].values()):
    #     writer.add_histogram(prefix + param_name, (init_param - fast_param), total_steps)

    model_log = model_output.get('log', list())
    for type, name, value in model_log:
        if type == 'scalar':
            writer.add_scalar(prefix + name, value, total_steps)
        elif type == 'model_output':
            img = utils.lin2img(value)
            writer.add_image(prefix + name, make_grid(img, scale_each=False, normalize=True),
                             global_step=total_steps)

    min_max_summary(prefix + 'model_out_min_max', model_output['model_out'], writer, total_steps)
    min_max_summary(prefix + 'gt_min_max', gt['rgb'], writer, total_steps)
    min_max_summary(prefix + 'coords_min_max', gt['x'], writer, total_steps)

def celebagan(model, model_input, gt, model_output, writer, total_steps, prefix='train_'):
    # Full forward pass with full mgrid
    # Plot context
    pred_img = utils.lin2img(model_output['rgb'])
    writer.add_image(prefix + 'gt_vs_pred', make_grid(pred_img, scale_each=False, normalize=True),
                     global_step=total_steps)

    model_log = model_output.get('log', list())
    for type, name, value in model_log:
        if type == 'scalar':
            writer.add_scalar(prefix + name, value, total_steps)
        elif type == 'model_output':
            img = utils.lin2img(value)
            writer.add_image(prefix + name, make_grid(img, scale_each=False, normalize=True),
                             global_step=total_steps)

    min_max_summary(prefix + 'model_out_min_max', model_output['model_out'], writer, total_steps)

def audio(model, model_input, gt, model_output, writer, total_steps, prefix='train_', save_latents=True, audiovisual=False):
    # import tensorflow as tf # NOTE: added here to avoid breaking other code if not included in env
    import torchaudio

    device = "cpu"

    gt_wav = torch.squeeze(gt['wav'][:, :, 0], axis=-1)

    # Spectograms always have the same size
    if gt_wav.size(-1) !=  200*41:
        return

    gt_wav = gt_wav.to(device)
    pred_wav = torch.squeeze(model_output['model_out'][:, :, 0], axis=-1)
    data = np.load("instrument.npz")

    mean = torch.Tensor(data['mean'][:-1]).to(device)
    std = torch.Tensor(data['std'][:-1]).to(device)

    # if len(gt_wav.size()) == 2:
    #     pred_wav = pred_wav.unsqueeze(1)
    #     gt_wav = gt_wav.unsqueeze(1)
    # else:
    #     pred_wav = pred_wav.permute(0, 2, 1).contiguous()
    #     gt_wav = gt_wav.permute(0, 2, 1).contiguous()

    pred_wav = pred_wav.to(device)

    print("gt shape: ", gt_wav.shape, " pred shape: ", pred_wav.shape)

    # add spectrogram
    gt_spectrogram = gt_wav.view(200, 41)
    pred_spectrogram = pred_wav.view(200, 41)

    gt_spectrogram = gt_spectrogram * 3 * std[:, None] + mean[:, None]
    pred_spectrogram = pred_spectrogram * 3 * std[:, None] + mean[:, None]

    gt_spectrogram = torch.exp(gt_spectrogram)
    pred_spectrogram = torch.exp(pred_spectrogram)

    gt_spectrogram = torch.cat([gt_spectrogram, torch.zeros_like(gt_spectrogram[-1:, :])], dim=0)
    pred_spectrogram = torch.cat([pred_spectrogram, torch.zeros_like(pred_spectrogram[-1:, :])], dim=0)

    with torch.enable_grad():
        gt_wav = torchaudio.transforms.GriffinLim()(gt_spectrogram)[None, :]
        pred_wav = torchaudio.transforms.GriffinLim()(pred_spectrogram.detach())[None, :]

    # gt_spectrogram = torchaudio.transforms.Spectrogram()(gt_wav)
    # pred_wav = pred_wav.view(*gt_wav.size())
    # pred_spectrogram = torchaudio.transforms.Spectrogram()(pred_wav)

    print("gt shape: ", gt_spectrogram.shape, " pred shape: ", pred_spectrogram.shape)

    output_vs_gt = torch.cat((gt_spectrogram, pred_spectrogram), dim=-1)
    writer.add_image(prefix + 'gt_vs_pred_spec', make_grid(output_vs_gt, scale_each=False, normalize=False),#, cmap="plasma"),
                     global_step=total_steps)

    # add playable audio sample (k per batch concatenated)
    num_save_per_batch = 5
    gt_audio_subset, elmts = utils.stitch_audio_subset(gt_wav, num_sample=num_save_per_batch, elmts=None)
    pred_audio_subset, _ = utils.stitch_audio_subset(pred_wav, elmts=elmts) # use same rand idcs as gt for predictions
    sample_rate = gt['rate'][0].cpu().numpy()

    gt_audio_subset = gt_audio_subset.squeeze()
    writer.add_audio(prefix + 'gt_audio', gt_audio_subset, sample_rate=int(sample_rate),global_step=total_steps)
    # handle nan - update! (should not need to do this...?)
    print("pred subset: ", pred_audio_subset)
    pred_audio_subset[torch.isnan(pred_audio_subset)] = 0.0

    pred_audio_subset = pred_audio_subset.squeeze()
    writer.add_audio(prefix + 'pred_audio', pred_audio_subset, sample_rate=int(sample_rate),global_step=total_steps)

    if prefix == "train_" and save_latents: summarize_latents(writer, model, model_output, total_steps, prefix, n_iter_step=10000,add_statistics=True)

    min_max_summary(prefix + 'model_out_min_max_wav', model_output['model_out'], writer, total_steps)
    min_max_summary(prefix + 'gt_min_max_wav', gt['wav'], writer, total_steps)
    min_max_summary(prefix + 'coords_min_max_wav', gt['x'], writer, total_steps)

    if prefix != "train_" and (not audiovisual):
        model_input_mix = {}
        model_input_mix['context'] = {}
        model_input_mix['context']['idx'] = model_input['context']['idx'][:1, :]
        model_input_mix['context']['x'] = model_input['context']['x'][0]

        if 'audio_coord' in model_input_mix['context']:
            model_input_mix['context']['audio_coord'] = model_input['context']['audio_coord'][0]

        model_output = model(model_input_mix, mix_sample=True, render=True)

        if 'num_pixels' in gt:
            num_pixels = int(gt["num_pixels"].cpu().numpy()[0])
            pred_wav = torch.squeeze(model_output['model_out'][1][:, :, 0], axis=-1).unsqueeze(1)
            pred_wav = pred_wav.to(device)
        else:
            pred_wav = torch.squeeze(model_output['model_out'], axis=-1).unsqueeze(1)
            pred_wav = pred_wav.to(device)

        pred_audio_subset, _ = utils.stitch_audio_subset(pred_wav, elmts=None) # use same rand idcs as gt for predictions

        writer.add_audio(prefix + 'audio_interpolate', pred_audio_subset, sample_rate=int(sample_rate),global_step=total_steps)
        writer.add_histogram(prefix + '_dist_histogram', model_output['dist_hist'], total_steps)


def imnet(model, model_input, gt, model_output, writer, total_steps, prefix='train_'):
    coords = model_input['context']['x']
    min_max_summary(prefix + 'coords_min_max', coords, writer, total_steps)


def film(model, model_input, gt, model_output, writer, total_steps, prefix='train_'):
    images(model, model_input, gt, model_output, writer, total_steps, prefix)

# def audio_visual(model, model_input, gt, model_output, writer, total_steps, prefix='train_'):
#     # create different model_input and model_output to feed to audio/img summaries separately
#
#     import torchaudio
#
#     print("model input: ", model_input.keys(), model_input["context"].keys(), model_input["query"].keys())
#     coords = model_input['context']['x']
#     if coords.size(0) ==  1: # only do for validation
#         num_pixels = int(gt["num_pixels"].cpu().numpy()[0])
#
#         model_output = model(model_input, mix_sample=True, render=True)
#         writer.add_histogram(prefix + '_dist_histogram', model_output['dist_hist'], total_steps)
#         # get rgb
#         sidelen = int(num_pixels**0.5)
#         # rgb = model_output["model_out"][:,:num_pixels, :-1].view(model_output["model_out"].shape(0), sidelen, sidelen, 1).permute(0, 3, 1, 2)
#         rgb = model_output["model_out"][:,:num_pixels, :-1].view(3, sidelen, sidelen, 1).permute(0, 3, 1, 2)
#         writer.add_image(prefix + '_latent_interpolate_rgb', make_grid(rgb, scale_each=False, normalize=True),
#                          global_step=total_steps)
#         # get spectrograms + waveforms
#         pred_wav = torch.squeeze(model_output["model_out"][:, num_pixels:, -1], axis=-1).unsqueeze(1).to("cpu")
#         pred_spec = torchaudio.transforms.Spectrogram()(pred_wav)
#         writer.add_image(prefix + '_latent_interpolate_spec', make_grid(pred_spec, scale_each=False, normalize=False),
#                          global_step=total_steps)
#         pred_audio_subset, _ = utils.stitch_audio_subset(pred_wav, num_sample=-1) # sample all
#         sample_rate = gt['rate'][0].cpu().numpy()
#         writer.add_audio(prefix + '_latent_interpolate_wav', pred_audio_subset, sample_rate=int(sample_rate),
#                          global_step=total_steps)
#
#     # num_pixels = int(gt["num_pixels"].cpu().numpy()[0])
#     # audio_output = {"model_out": model_output["model_out"][:,num_pixels:,-1],"representation": model_output["representation"]}
#     #
#     # print("num pixels: ", num_pixels, " model out shape: ", model_output["model_out"].shape)
#     #
#     # audio(model, model_input, gt, audio_output, writer, total_steps, prefix,summarize_latents=False)
#     # pred_img = model_output["model_out"][:,:num_pixels, :-1]#.unsqueeze(-1)
#     # img_output = {"model_out": pred_img, "rgb": pred_img, "representation": model_output["representation"]}
#     # img_input = {"context": {"x": model_input["context"]["x"][:, :num_pixels, :-1],"y": gt["rgb"], "mask": gt['mask'][:,:num_pixels,:]}}
#     # images(model, img_input, gt, img_output, writer, total_steps, prefix)

def audio_visual(model, model_input, gt, model_output, writer, total_steps, prefix='train_'):
    # create different model_input and model_output to feed to audio/img summaries separately
    # num_pixels = int(gt["num_pixels"].cpu().numpy()[0])
    audio_output = {"model_out": model_output["model_out"][1],"representation": model_output["representation"]}

    audio(model, model_input, gt, audio_output, writer, total_steps, prefix,save_latents=False, audiovisual=True)
    pred_img = model_output["model_out"][0]#.unsqueeze(-1)
    img_output = {"model_out": pred_img, "rgb": pred_img, "representation": model_output["representation"]}
    img_input = {"context": {"x": model_input["context"]["x"],"y": gt["rgb"], "mask": gt['mask'], "idx": model_input["context"]["idx"]}}
    images(model, img_input, gt, img_output, writer, total_steps, prefix)

def audio_visual_gan(model, model_input, gt, model_output, writer, total_steps, prefix='train_'):

    import torchaudio

    # audio data
    num_pixels = int(gt["num_pixels"].cpu().numpy()[0])
    audio_output = model_output['model_out'][:, num_pixels:, -1]
    pred_wav = torch.squeeze(audio_output, axis=-1).unsqueeze(1)
    pred_wav = pred_wav.to("cpu")
    print("pred wav shape: ", pred_wav.shape)
    pred_spectrogram = torchaudio.transforms.Spectrogram()(pred_wav)
    writer.add_image(prefix + 'pred_spec', make_grid(pred_spectrogram, scale_each=False, normalize=False),#, cmap="plasma"),
                     global_step=total_steps)
    # add playable audio sample (k per batch concatenated)
    num_save_per_batch = 5
    pred_audio_subset, _ = utils.stitch_audio_subset(pred_wav, num_sample=num_save_per_batch)
    sample_rate = gt['rate'][0].cpu().numpy()
    writer.add_audio(prefix + 'pred_audio', pred_audio_subset, sample_rate=int(sample_rate),global_step=total_steps)
    min_max_summary(prefix + 'model_out_min_max_wav', audio_output, writer, total_steps)

    # img data
    img_output = model_output["model_out"][:, :num_pixels, :-1]
    pred_img = utils.lin2img(img_output)
    print("pred img shape: ", pred_img.shape)
    writer.add_image(prefix + 'pred_img', make_grid(pred_img, scale_each=False, normalize=True),
                     global_step=total_steps)
    min_max_summary(prefix + 'model_out_min_max', img_output, writer, total_steps)



def variational(model, model_input, gt, model_output, writer, total_steps, prefix='train_'):
    images(model, model_input, gt, model_output, writer, total_steps, prefix)

    writer.add_scalar(prefix + "emp_std", model_output['z'].std(), total_steps)
    writer.add_scalar(prefix + "emp_mean", model_output['z'].mean(), total_steps)

    writer.add_scalar(prefix + "var_std", torch.exp(model_output['logvar']*0.5).mean(), total_steps)
    writer.add_scalar(prefix + "var_mu", model_output['mu'].mean(), total_steps)


def multi_decoder(model, model_input, gt, model_output, writer, total_steps, prefix='train_'):
    images(model, model_input, gt, model_output, writer, total_steps, prefix)

    with torch.enable_grad():
        dummy_model_in = {'context':{}}
        dummy_model_in['context']['x'] = model_input['context']['x'][:1]
        dummy_model_in['context']['idx'] = model_input['context']['idx'][:1]
        dummy_model_out = model(dummy_model_in)

        split_zs = dummy_model_out['split_zs']

        # Loop through outputs and calculate gradient of each pixel w.r.t. model parameters
        all_grads = list()
        for pixel in dummy_model_out['rgb'][0, ...].mean(dim=-1):
            grads = torch.autograd.grad(pixel, split_zs, create_graph=False, retain_graph=True)
            all_grads.append(torch.stack([grad.norm() for grad in grads], dim=0))

        per_latent_images = torch.stack(all_grads, dim=-1).view(len(split_zs), 1, 64, 64)

    writer.add_image(prefix + 'latent_hms', make_grid(per_latent_images, scale_each=False, normalize=True),
                     global_step=total_steps)


def images_depths(model, model_input, gt, model_output, writer, total_steps, prefix='train_'):
    # Full forward pass with full mgrid
    # Plot context

    gt_img = utils.lin2img(gt['depth'])
    pred_img = utils.lin2img(model_output['model_out'][:,:,-1:])
    output_vs_gt = torch.cat((gt_img, pred_img), dim=-1)
    writer.add_image(prefix + 'gt_vs_pred_depth', make_grid(output_vs_gt, scale_each=False, normalize=True),
                     global_step=total_steps)

    gt_img = utils.lin2img(gt['img'])
    pred_img = utils.lin2img(model_output['model_out'][:,:,:-1])
    output_vs_gt = torch.cat((gt_img, pred_img), dim=-1)
    writer.add_image(prefix + 'gt_vs_pred_img', make_grid(output_vs_gt, scale_each=False, normalize=True),
                     global_step=total_steps)

    # fig, ax = plt.subplots()
    # coords = model_input['context']['x']
    # x, y = coords[0, ..., 0].detach().cpu().numpy(), coords[0, ..., 1].detach().cpu().numpy()
    # color = model_input['context']['y'][0].detach().cpu().numpy()
    # color = (color - np.amin(color)) / (np.amax(color) - np.amin(color))

    # ax.scatter(y, x, c=color)
    # writer.add_figure(prefix + 'conditioning_points', fig, global_step=total_steps)

    for param_name, init_param, fast_param in zip(model_output['fast_params'].keys(), model.hypo_module.meta_parameters(), model_output['fast_params'].values()):
        writer.add_histogram(prefix + param_name, (init_param - fast_param), total_steps)

    if 'logits' in model_output:
        _, predicted = torch.max(model_output['logits'], 1)
        top_1, top_5 = utils.accuracy(model_output['logits'], gt['label'], topk=(1, 5))

        writer.add_scalar(prefix + 'top 1', top_1, total_steps)
        writer.add_scalar(prefix + 'top 5', top_5, total_steps)

    model_log = model_output.get('log', list())
    for type, name, value in model_log:
        if type == 'scalar':
            writer.add_scalar(prefix + name, value, total_steps)
        elif type == 'model_output':
            img = utils.lin2img(value)
            writer.add_image(prefix + name, make_grid(img, scale_each=False, normalize=True),
                             global_step=total_steps)

    min_max_summary(prefix + 'img_model_out_min_max', model_output['model_out'][:,:,:-1], writer, total_steps)
    min_max_summary(prefix + 'img_gt_min_max', gt['img'], writer, total_steps)

    min_max_summary(prefix + 'depth_model_out_min_max', model_output['model_out'][:,:,-1:], writer, total_steps)
    min_max_summary(prefix + 'depth_gt_min_max', gt['depth'], writer, total_steps)

    min_max_summary(prefix + 'coords_min_max', gt['x'], writer, total_steps)

def images_gon(model, model_input, gt, model_output, writer, total_steps, prefix='train_'):
    # Full forward pass with full mgrid
    # Plot context
    gt_img = utils.lin2img(gt['img'])

    pred_img = utils.lin2img(model_output['model_out'])

    output_vs_gt = torch.cat((gt_img, pred_img), dim=-1)
    writer.add_image(prefix + 'gt_vs_pred', make_grid(output_vs_gt, scale_each=False, normalize=True),
                     global_step=total_steps)

    # fig, ax = plt.subplots()
    # coords = model_input['context']['x']
    # x, y = coords[0, ..., 0].detach().cpu().numpy(), coords[0, ..., 1].detach().cpu().numpy()
    # color = model_input['context']['y'][0].detach().cpu().numpy()
    # color = (color - np.amin(color)) / (np.amax(color) - np.amin(color))

    # ax.scatter(y, x, c=color)
    # writer.add_figure(prefix + 'conditioning_points', fig, global_step=total_steps)

    if 'logits' in model_output:
        _, predicted = torch.max(model_output['logits'], 1)
        top_1, top_5 = utils.accuracy(model_output['logits'], gt['label'], topk=(1, 5))

        writer.add_scalar(prefix + 'top 1', top_1, total_steps)
        writer.add_scalar(prefix + 'top 5', top_5, total_steps)

    model_log = model_output.get('log', list())
    for type, name, value in model_log:
        if type == 'scalar':
            writer.add_scalar(prefix + name, value, total_steps)
        elif type == 'model_output':
            img = utils.lin2img(value)
            writer.add_image(prefix + name, make_grid(img, scale_each=False, normalize=True),
                             global_step=total_steps)

    min_max_summary(prefix + 'model_out_min_max', model_output['model_out'], writer, total_steps)
    min_max_summary(prefix + 'gt_min_max', gt['img'], writer, total_steps)
    min_max_summary(prefix + 'coords_min_max', gt['x'], writer, total_steps)


def normalize_rgb_signal(signal):
    signal += 1
    signal /= 2.
    signal = torch.clamp(signal, 0., 1.)
    return signal


class VideoSummarizer():
    def __init__(self, num_frames, sidelength):
        self.num_frames = num_frames
        self.sidelength = sidelength

    def __call__(self, model, model_input, gt, model_output, writer, total_steps, prefix='train_'):
        gt_video = utils.lin2video(gt['video_flat'], self.num_frames, self.sidelength)
        gt_mgrid = gt['dense_coords']

        with torch.no_grad():
            model_input['context'] = get_single_batch_item_from_dict(model_input['context'], 0, squeeze=False)
            params = get_single_batch_item_from_dict(model_output['fast_params'], 0, squeeze=False)
            query_x = gt_mgrid[:1, ...]

            sample_model_out = model.forward_with_params(query_x, params)
            pred_video = utils.lin2video(sample_model_out, self.num_frames, self.sidelength)

        # Videos have shape (batch, frames, height, width, channels)
        gt_video = gt_video.permute(0, 1, -1, 2, 3)[:1, ...]
        pred_video = pred_video.permute(0, 1, -1, 2, 3)

        output_vs_gt = torch.cat((gt_video, pred_video), dim=-1)
        output_vs_gt = normalize_rgb_signal(output_vs_gt)
        writer.add_video(prefix + 'gt_video', output_vs_gt, global_step=total_steps)

        min_max_summary(prefix + 'model_out_min_max', model_output['model_out'], writer, total_steps)
        min_max_summary(prefix + 'gt_min_max', gt['video_flat'], writer, total_steps)
        min_max_summary(prefix + 'coords_min_max', gt['x'], writer, total_steps)


def min_max_summary(name, tensor, writer, total_steps):
    writer.add_scalar(name + '_min', tensor.min().detach().cpu().numpy(), total_steps)
    writer.add_scalar(name + '_max', tensor.max().detach().cpu().numpy(), total_steps)


def write_psnr(pred_img, gt_img, writer, iter, prefix):
    batch_size = pred_img.shape[0]

    pred_img = pred_img.detach().cpu().numpy()
    gt_img = gt_img.detach().cpu().numpy()

    psnrs, ssims = list(), list()
    for i in range(batch_size):
        p = pred_img[i].transpose(1, 2, 0)
        trgt = gt_img[i].transpose(1, 2, 0)

        p = (p / 2.) + 0.5
        p = np.clip(p, a_min=0., a_max=1.)

        trgt = (trgt / 2.) + 0.5

        ssim = skimage.measure.compare_ssim(p, trgt, multichannel=True, data_range=1)
        psnr = skimage.measure.compare_psnr(p, trgt, data_range=1)

        psnrs.append(psnr)
        ssims.append(ssim)

    writer.add_scalar(prefix + "psnr", np.mean(psnrs), iter)
    writer.add_scalar(prefix + "ssim", np.mean(ssims), iter)


class TestWriter():
    def __init__(self, target_path, overwrite=True):
        self.target_path = target_path
        self.counter = 0

        if os.path.exists(target_path):
            if overwrite:
                shutil.rmtree(target_path)
            else:
                val = input("The model directory %s exists. Overwrite? (y/n)"%target_path)
                if val == 'y' or overwrite:
                    shutil.rmtree(target_path)

        os.makedirs(target_path)
        self.target_path = Path(target_path)

    def __call__(self, model_output, gt, losses):
        raise NotImplementedError


def get_single_batch_item_from_dict(dict, idx, squeeze=True):
    single_dict = {}
    for key, value in dict.items():
        single_dict[key] = dict[key][idx]

        if not squeeze:
            single_dict[key] = single_dict[key][None, ...]
    return single_dict


def conditional_save(save_fn, path):
    if not os.path.exists(path):
        save_fn()


def convert_torch_dict_to_numpy_dict(dict):
    new_dict = {}
    for key, value in dict.items():
        new_dict[key] = value.detach().cpu().numpy()
    return new_dict


class ImageTestWriter(TestWriter):
    def __init__(self, target_path, with_activations, with_weights, with_labels, overwrite=True):
        self.with_weights = with_weights
        self.with_activations = with_activations
        self.with_labels = with_labels
        super().__init__(target_path, overwrite)

    def __call__(self, model_output, gt, losses):
        pred_imgs = utils.lin2img(model_output['model_out']).cpu().numpy()
        gt_imgs = utils.lin2img(gt['y']).cpu().numpy()

        if self.with_labels:
            gt_labels = gt['label'].cpu().numpy()

        for i in range(gt_imgs.shape[0]):
            # Save out image

            # check number of channels
            if np.shape(pred_imgs[i])[0] == 1:
                pred_img = pred_imgs[i].squeeze(0)
                gt_img = gt_imgs[i].squeeze(0)
            else:
                pred_img = pred_imgs[i].squeeze().transpose(1, 2, 0)
                gt_img = gt_imgs[i].squeeze().transpose(1, 2, 0)

            pred_img += 1.
            pred_img /= 2.
            pred_img = np.clip(pred_img, 0., 1.)

            gt_img += 1.
            gt_img /= 2.
            gt_img = np.clip(gt_img, 0., 1.)

            psnr = skimage.metrics.peak_signal_noise_ratio(gt_img, pred_img)
            losses.update({'psnr':psnr})

            metrics_path = self.target_path / f"{self.counter:05}_metrics.csv"

            if not os.path.exists(metrics_path):
                w = csv.writer(open(metrics_path, "w"))
                for key, val in losses.items():
                    w.writerow([key, val])

            gt_path = self.target_path / f"{self.counter:05}_gt.png"
            pred_path = self.target_path / f"{self.counter:05}_pred.png"

            pred_img = (pred_img * 255).astype(np.uint8)
            gt_img = (gt_img * 255).astype(np.uint8)

            conditional_save(lambda: imageio.imwrite(gt_path, gt_img), gt_path)
            conditional_save(lambda: imageio.imwrite(pred_path, pred_img), pred_path)

            if self.with_activations:
                activation_path = self.target_path / f"{self.counter:05}_activations.pck"
                activations = get_single_batch_item_from_dict(model_output['activations'], i)
                activations = convert_torch_dict_to_numpy_dict(activations)
                conditional_save(lambda: np.save(activation_path, activations), activation_path)

            if self.with_weights:
                weight_path = self.target_path / f"{self.counter:05}_weights.pck"
                weights = get_single_batch_item_from_dict(model_output['fast_params'], i)
                weights = convert_torch_dict_to_numpy_dict(weights)
                conditional_save(lambda: pickle.dump(weights, open(weight_path, "wb")), weight_path)
                # conditional_save(lambda: np.save(weight_path, weights), weight_path)

            if self.with_labels:
                label_path = self.target_path / f"{self.counter:05}_labels"
                conditional_save(lambda: np.save(label_path, gt_labels[i]), label_path)

            self.counter += 1
