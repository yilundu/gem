import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
import utils

import diff_operators
import modules
from collections import OrderedDict


def inner_maml_mse(prediction, gt, mask=None):
    if mask is None:
        loss = ((prediction - gt) ** 2)
    else:
        loss = (mask.cuda() * (prediction - gt) ** 2)
    return loss.sum(0).mean()


def occupancy(prediction, gt, mask=None):
    loss_occ = torch.log(prediction['model_out'] + 1e-5) * gt['occupancy'] + (1 - gt['occupancy']) * torch.log(1 - prediction['model_out'] + 1e-5)
    return {'occupancy': (-loss_occ).mean()}

def audio_L1(model_output, gt, mask=None):
    mask = torch.squeeze(gt['mask'])
    pred_wav = torch.squeeze(model_output['model_out'])
    gt_wav = torch.squeeze(gt['wav'])

    loss = torch.sum(mask*torch.abs(pred_wav - gt_wav)) / torch.sum(mask)
    return {"audio_loss": loss}

def audio_mse(model_output, gt, mask=None):
    mask = torch.squeeze(gt['mask'])
    pred_wav = torch.squeeze(model_output['model_out'])
    gt_wav = torch.squeeze(gt['wav'])

    loss = torch.sum(mask*(pred_wav - gt_wav)**2) / torch.sum(mask)
    return {"audio_loss": loss}

def audio_mse_linear(model_output, gt, mask=None):
    mask = torch.squeeze(gt['mask'])
    pred_wav = torch.squeeze(model_output['model_out'])
    gt_wav = torch.squeeze(gt['wav'])

    loss = torch.sum(mask*(pred_wav - gt_wav)**2) / torch.sum(mask)

    pred_wav = torch.squeeze(model_output['model_out_linear'])
    linear_loss = torch.sum(mask * (pred_wav - gt_wav) ** 2) / torch.sum(mask)

    latents = model_output['latents']
    idx = gt['idx']
    idx_other = gt['idx_other']

    latents_i = latents(idx)
    latents_j = latents(idx_other)

    mse = gt['mse']

    dist = (latents_i - latents_j).pow(2).mean(dim=-1)
    dist = dist[:, 0]

    dist_norm = (dist) / dist.std()
    mse_norm = (mse)  / mse.std()

    return {"audio_loss": loss, "audio_linear_loss": linear_loss, 'mds_loss': (dist_norm - mse_norm).pow(2).mean()}

def inner_maml_mse_subset(prediction, gt, mask=None):
    gt_ch = gt.shape[-1]
    loss = ((prediction[..., :gt_ch] - gt) ** 2)
    return loss.sum(0).mean()


def comp_mse(model_output, gt, mask=None):
    batch_size = gt['rgb'].shape[0]
    loss = ((gt['rgb'] - model_output['rgb']) ** 2)

    lm_coords = model_output['lm_coords'].squeeze()
    lm_loss = 0.
    for instance in range(lm_coords.shape[-1]):
        instance_coord = lm_coords[..., instance] # (-1, 2, 3)
        pairwise = (instance_coord[..., None] - instance_coord[:, :, None, :]).norm(dim=1)
        pairwise_zero = pairwise * (1 - torch.eye(3, 3)[None, ...].repeat(batch_size, 1, 1).cuda())
        lm_loss += torch.max(torch.zeros_like(pairwise).cuda(), 0.05 - pairwise_zero).mean()* 100
    print(lm_loss)
    return {'img_loss': loss.mean(), 'distance': lm_loss}


def image_mse_decay(model_output, gt, mask=None):
    loss = ((model_output['rgb'].cuda() - gt['rgb'].cuda()) ** 2)
    representation_loss = 0.0 * torch.norm(model_output['representation'], dim=-1).mean()
    return {'img_loss': loss.mean(), 'rep_loss': representation_loss.mean()}


def image_mse_manifold(model_output, gt, mask=None):
    loss = ((model_output['rgb'].cuda() - gt['rgb'].cuda()) ** 2)
    z = model_output['z_orig']
    latents = model_output['latents'].weight
    diff = z[:8, None, :] - latents[None, :, :]
    dist = torch.norm(diff, p=2, dim=-1)
    dist_mean = dist.mean(dim=-1)

    dist_small = torch.topk(dist, 300, dim=1, largest=False)[0]
    dist_small = dist_small[:, 1:]

    dist_loss = 1e-3 * (dist_small.mean())
    return {'img_loss': loss.mean(), 'rep_loss': dist_loss}


def image_mse_linear(model_output, gt, mask=None):
    loss = ((model_output['rgb'].cuda() - gt['rgb'].cuda()) ** 2)

    weight = model_output['weights']
    inv_weight = torch.clamp(-weight, 0, 1000)
    loss_weight = inv_weight.mean()


    latents = model_output['latents']
    idx = gt['idx']
    idx_other = gt['idx_other']

    latents_i = latents(idx)
    latents_j = latents(idx_other)

    mse = gt['mse']

    dist = (latents_i - latents_j).pow(2).mean(dim=-1)
    dist = dist[:, 0]

    mse_idx = torch.argsort(mse, dim=0)
    mse = mse[mse_idx[:32]]
    dist = dist[mse_idx[:32]] * model_output['scale_factor'].squeeze()

    dist_norm = (dist) # / dist.std()
    mse_norm = (mse) # / mse.std()

    z = model_output['z_orig']

    loss_linear = ((model_output['rgb_linear'].cuda() - gt['rgb'].cuda()) ** 2)

    return {'img_loss': loss.mean(), 'linear_loss': loss_linear, 'loss_weight': loss_weight, 'mds_loss': torch.abs(dist_norm - mse_norm).mean()}


def occupancy_linear(prediction, gt, mask=None):
    occs = gt['occupancy']
    xs = gt['x']

    weight = prediction['weights']
    inv_weight = torch.clamp(-weight, 0, 1000)
    loss_weight = inv_weight.mean()

    with torch.no_grad():
        xs_perm = torch.cat([xs[1:], xs[:1]], dim=0)

        occs_perm = torch.cat([occs[1:], occs[:1]], dim=0)

        xss = torch.chunk(xs, 4, dim=0)
        xss_perm = torch.chunk(xs_perm, 4, dim=0)
        occss = torch.chunk(occs, 4, dim=0)
        occss_perm = torch.chunk(occs_perm, 4, dim=0)
        mses = []

        for xsi, xsi_perm, occsi, occsi_perm in zip(xss, xss_perm, occss, occss_perm):
            dist = torch.norm(xsi[:, :, None, :] - xsi_perm[:, None, :, :], dim=-1)

            filter_mask = torch.ones_like(dist)

            # Prevent matching of spurious points
            dist = dist + filter_mask * 1000. * (1 - occsi[:, :, None, 0]) + filter_mask * 1000. * (1 - occsi_perm[:, None, :, 0])

            min_dist_perm = (dist.min(dim=1)[0] * occsi_perm[:, :, 0]).sum(dim=1) / occsi_perm[:, :, 0].sum(dim=1)
            min_dist = (dist.min(dim=2)[0] * occsi[:, :, 0]).sum(dim=1) / occsi[:, :, 0].sum(dim=1)

            mse = min_dist + min_dist_perm
            mses.append(mse)

        mse = torch.cat(mses, dim=0)

    latents = prediction['latents']
    idx = gt['idx']
    latents_i = latents(idx)
    latents_j = torch.cat([latents_i[1:], latents_i[:1]], dim=0)

    dist = (latents_i - latents_j).pow(2).mean(dim=-1)
    dist = dist[:, 0]

    mse_idx = torch.argsort(mse, dim=0)
    mse = mse[mse_idx[:32]]
    dist = dist[mse_idx[:32]] * prediction['scale_factor'].squeeze()

    loss_occ = torch.log(prediction['model_out'] + 1e-5) * gt['occupancy'] + (1 - gt['occupancy']) * torch.log(1 - prediction['model_out'] + 1e-5)
    loss_occ_linear = torch.log(prediction['model_out_linear'] + 1e-5) * gt['occupancy'] + (1 - gt['occupancy']) * torch.log(1 - prediction['model_out_linear'] + 1e-5)
    return {'occupancy_linear': (-loss_occ_linear).mean(), 'occupancy': (-loss_occ).mean(), 'loss_weight': loss_weight, 'mds_loss': torch.abs(dist - mse).mean()}


def image_mse(model_output, gt, mask=None):
    if mask is None:
        loss = ((model_output['rgb'].cuda() - gt['rgb'].cuda()) ** 2)

        return {'img_loss': loss.mean()}
    else:
        loss = (mask.cuda() * (model_output['rgb'].cuda() - gt['rgb'].cuda()) ** 2)
        loss /= (3 * mask.sum(dim=0, keepdim=True).sum(dim=-1, keepdim=True).sum(dim=1, keepdim=True)+1)
        return {'img_loss':loss.sum()}


def image_mse_spectral(model_output, gt, mask=None):
    loss = ((model_output['rgb'].cuda() - gt['rgb'].cuda()) ** 2)
    z = model_output['z_orig']
    z_mean = z.mean(dim=-1, keepdim=True)
    z_center = z - z_mean
    z_cov = torch.matmul(z_center.transpose(1, 0), z_center)
    out = torch.matmul(z_cov.transpose(1, 0), z_cov)
    _, s, _ = torch.svd(out, compute_uv=True)

    # target = torch.eye(1024).cuda()
    eig_loss = torch.abs(s - 1).mean()

    return {'img_loss': loss.mean(), 'eig_loss': eig_loss}


def audio_visual_mse(model_output, gt, mask=None):
    # decompose audio and visual loss
    losses = {}

    pred_rgb, pred_wav = model_output['model_out']
    gt_wav = gt['wav'].squeeze(axis=-1)

    pred_wav = pred_wav.squeeze()
    losses["audio_loss"] = ((pred_wav - gt_wav)**2).mean()

    loss =  (pred_rgb.cuda() - gt['rgb'].cuda()) ** 2
    losses['img_loss'] = loss.mean()

    return losses

def audio_visual_mse_image_only(model_output, gt, mask=None):
    # decompose audio and visual loss
    losses = {}

    pred_rgb, pred_wav = model_output['model_out']
    gt_wav = gt['wav'].squeeze(axis=-1)

    loss =  (pred_rgb.cuda() - gt['rgb'].cuda()) ** 2
    losses['img_loss'] = loss.mean()

    return losses


def audio_visual_mse_audio_only(model_output, gt, mask=None):
    # decompose audio and visual loss
    losses = {}

    pred_rgb, pred_wav = model_output['model_out']
    gt_wav = gt['wav'].squeeze(axis=-1)

    pred_wav = pred_wav.squeeze()
    losses["audio_loss"] = ((pred_wav - gt_wav)**2).mean()

    return losses


def audio_visual_mse_linear(model_output, gt, mask=None):
    # decompose audio and visual loss
    losses = {}

    pred_rgb, pred_wav = model_output['model_out']
    gt_wav = gt['wav'].squeeze(axis=-1)

    pred_wav = pred_wav.squeeze()
    losses["audio_loss"] = ((pred_wav - gt_wav)**2).mean()

    loss =  (pred_rgb.cuda() - gt['rgb'].cuda()) ** 2
    losses['img_loss'] = loss.mean()

    pred_rgb, pred_wav = model_output['model_out_linear']
    pred_wav = pred_wav.squeeze()
    gt_wav = gt['wav'].squeeze(axis=-1)
    losses["audio_linear_loss"] = ((pred_wav - gt_wav)**2).mean()

    loss = ((pred_rgb.cuda() - gt['rgb'].cuda()) ** 2)
    losses["img_linear_loss"] = loss.mean()

    weight = model_output['weights']
    inv_weight = torch.clamp(-weight, 0, 1000)
    loss_weight = inv_weight.mean()
    losses['loss_weight'] = loss_weight

    latents = model_output['latents']
    idx = gt['idx']
    idx_other = gt['idx_other']

    latents_i = latents(idx)
    latents_j = latents(idx_other)

    mse = gt['mse']

    dist = (latents_i - latents_j).pow(2).mean(dim=-1)
    dist = dist[:, 0]

    dist_norm = (dist) * model_output['scale_factor'].squeeze()
    mse_norm = (mse)

    losses['mds_loss'] = torch.abs(dist_norm - mse_norm).mean()

    z = model_output['z_orig']
    # l2_loss = torch.norm(z, dim=-1, p=2) * 1e-1

    # losses['l2_loss'] = l2_loss

    return losses


def audio_visual_mse_linear_image_only(model_output, gt, mask=None):
    # decompose audio and visual loss
    losses = {}

    pred_rgb, pred_wav = model_output['model_out']
    loss =  (pred_rgb.cuda() - gt['rgb'].cuda()) ** 2
    losses['img_loss'] = loss.mean()

    return losses

def audio_visual_mse_linear_audio_only(model_output, gt, mask=None):
    # decompose audio and visual loss
    losses = {}

    pred_rgb, pred_wav = model_output['model_out']
    gt_wav = gt['wav'].squeeze(axis=-1)

    pred_wav = pred_wav.squeeze()
    losses["audio_loss"] = ((pred_wav - gt_wav)**2).mean() * 1000

    return losses


def image_mse_grad_penalty(model_output, gt, mask=None):
    if mask is None:
        loss = ((model_output['rgb'].cuda() - gt['rgb'].cuda()) ** 2)
        rgb_loss = torch.abs(model_output['rgb']).mean(dim=1).mean(dim=1)
        z_grad = torch.autograd.grad(rgb_loss.sum(), [model_output['z']], create_graph=True)[0]
        grad_penalty = 1e-3 * torch.norm(z_grad, dim=-1).mean()

        return {'img_loss': loss.mean(), 'grad_penalty': grad_penalty}
    else:
        loss = (mask.cuda() * (model_output['rgb'].cuda() - gt['rgb'].cuda()) ** 2)
        loss /= (3 * mask.sum(dim=0, keepdim=True).sum(dim=-1, keepdim=True).sum(dim=1, keepdim=True)+1)
        return {'img_loss':loss.sum()}


def latent_sparsity_loss(model_output, gt, mask=None):
    loss_dict = image_mse(model_output, gt, gt['mask'])

    latent_loss = 0.
    for z in model_output['split_zs']:
        latent_loss += z.norm(p=1) * 1e2 / z.shape[0]

    loss_dict.update({'latent_loss':latent_loss})
    return loss_dict


def disentangling_loss(model_output, gt, model_input, model, mask=None):
    loss_dict = image_mse(model_output, gt, gt['mask'])

    # Dot product penalty on parameters
    # head_params = torch.stack([torch.cat(head_outs, dim=-1)[0] for head_outs in model_output['net_outs']], dim=0)
    # head_params = F.normalize(head_params, dim=-1)
    # print(head_params.shape)
    # dot_products = torch.matmul(head_params, head_params.t())
    # dot_products_wo_diag = dot_products * (1 - torch.eye(len(model_output['net_outs'])).cuda())
    # dot_products_wo_diag = torch.abs(dot_products_wo_diag) * 1e2

    # # Randomly sample pixels and make sure that they're independent
    dummy_model_in = {'context': {}}
    dummy_model_in['context']['x'] = model_input['context']['x'][:1]
    dummy_model_in['context']['idx'] = model_input['context']['idx'][:1]
    dummy_model_out = model(dummy_model_in)

    split_zs = dummy_model_out['split_zs']

    mean_img = dummy_model_out['rgb'][0, ...].mean(dim=-1)
    independence_loss = 0.
    for idx in np.random.randint(0, mean_img.shape[-1], size=10):
        pixel = mean_img[idx]
        grads = torch.autograd.grad(pixel, split_zs, create_graph=True, retain_graph=True)
        grad_norms = torch.stack([grad.norm(p=1) for grad in grads], dim=0)
        print(grad_norms, F.softmax(grad_norms, dim=0))
        # max, _ = torch.max(grad_norms, dim=0, keepdim=True)
        # mask = grad_norms.lt(max)
        # independence_loss += grad_norms[mask].mean()
        independence_loss += grad_norms * F.softmax(grad_norms, dim=0)

    loss_dict.update({'disentangling':independence_loss})

    # loss_dict.update({'disentangling':dot_products_wo_diag})
    return loss_dict


def gan_loss(input, target_is_real, eps=1e-9):
    # if target_is_real:
    #     return ((input - 0.5) ** 2).mean()
    # else:
    #     return ((input + 0.5) ** 2).mean()
    if target_is_real:
        return -torch.log(input + 1e-3).mean()
    else:
        return -torch.log(1-input + 1e-3).mean()


class LatentGan(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.discriminator = modules.FCBlock(hidden_ch=latent_dim, num_hidden_layers=3, in_features=latent_dim,
                                             out_features=1, outermost_linear=True, nonlinearity='leaky_relu')

    def __call__(self, model_output, gt, gan_weight=1e-5):
        loss_dict = image_mse(model_output, gt, gt['mask'])

        batch_size = model_output['rgb'].shape[0]
        rand_z = torch.randn((batch_size, self.latent_dim)).cuda()

        # Discriminator forward passes
        # Fake forward step
        pred_fake_det = self.discriminator(model_output['z'].detach())
        loss_d_fake = gan_loss(pred_fake_det, False)

        # Real forward step
        pred_real = self.discriminator(rand_z)
        loss_d_real = gan_loss(pred_real, True)
        loss_dict['disc_loss'] = 0.5 * (loss_d_real + loss_d_fake)

        pred_fake = self.discriminator(model_output['z'])
        loss_dict['gen_loss'] = gan_loss(pred_fake, True)

        return loss_dict


def kl_div(model_outputs, prior_std=1.0):
    """Computes varitional loss
    KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
    KL(N(\mu, \sigma), N(0, 0.01)) = \log \frac{0.01}{\sigma} + \frac{\sigma^2 + \mu^2}{2*0.01*0.01} - \frac{1}{2}
    """
    logvar, mu = model_outputs['logvar'], model_outputs['mu']
    return -0.5 * torch.sum(
        1 + logvar - np.log(prior_std * prior_std) - (mu ** 2) / (prior_std * prior_std) - (
            logvar.exp()) / (prior_std * prior_std), dim=-1)


def auto_decoder(model_output, gt, kl_weight=1e-5):
    loss_dict = image_mse(model_output, gt, gt['mask'])
    # loss_dict['kl_loss'] = 0.5 * torch.sum(model_output['z']**2, dim=1)
    # kl_dict = {'logvar':torch.log(model_output['std']*model_output['std']), 'mu':model_output['mu']}
    # loss_dict['kl_loss'] = kl_div(kl_dict) * kl_weight

    return loss_dict


def variational_auto_decoder(model_output, gt, kl_weight=1e-2):
    loss_dict = image_mse(model_output, gt, gt['mask'])
    kl_loss = kl_div(model_output) * kl_weight
    loss_dict.update({'kl':kl_loss})
    return loss_dict


class ImageSemantic():
    def __init__(self, class_weights):
        self.class_weights = class_weights.cuda()

    def __call__(self, model_output, gt, mask=None):
        pred_img = model_output['rgb']
        gt_img = gt['rgb']

        if mask is None:
            img_loss = ((pred_img.cuda() - gt_img.cuda()) ** 2)
        else:
            img_loss = (mask.cuda() * (pred_img.cuda() - gt_img.cuda()) ** 2)

        # maml_loss = 0
        # for input, output in model_output['maml_ins_outs']:
        #     in_ch = input.shape[-1]
        #     maml_loss += ((input-output[..., :in_ch])**2).mean() * 1e-1

        pred_logits = model_output['semantic'].flatten(start_dim=0, end_dim=1)
        labels = gt['semantic'].flatten().long()

        sem_loss = torch.nn.functional.cross_entropy(pred_logits.cuda(), labels.cuda(), weight=self.class_weights)
        # return {"semantic_loss": sem_loss}
        # return {"semantic_loss": sem_loss, 'maml_loss':maml_loss}
        return {'img_loss': img_loss, "semantic_loss": sem_loss}
        # return {'img_loss': img_loss, "semantic_loss": sem_loss, 'maml_loss':maml_loss}


def inner_maml_mse_segmentation(prediction, gt, mask=None):
    prediction = prediction[:, :, :3]
    if mask is None:
        loss = ((prediction - gt) ** 2)
    else:
        loss = (mask.cuda() * (prediction - gt) ** 2)
    return loss.sum(0).mean()


def outer_srns_mse(model_out, gt):
    losses = {}
    trgt_shape = gt['y'].shape
    rgb = model_out['model_out'].cuda().view(trgt_shape)

    losses['img_loss'] = ((rgb - gt['y']) ** 2).mean()
    losses['depth_loss'] = ((model_out['query_ego_out']['depth'] * (model_out['query_ego_out']['depth']<0).float())**2).mean() * 1e4
    return losses


def inner_gon(prediction, gt, mask=None):
    # UPDATE!!
    if mask is None:
        loss = ((prediction - gt) ** 2)
    else:
        loss = (mask.cuda() * (prediction - gt) ** 2)
    return loss.sum(0).mean()


def cross_entropy(model_output, gt):
    pred_logits = model_output['logits']
    labels = gt['label']

    loss_dict =  {'classification':torch.nn.functional.cross_entropy(pred_logits, labels)}
    loss_dict.update(image_mse(model_output, gt))
    # loss_dict['img_loss'] *= 1e-2
    return loss_dict


def val_image_mse(model_output, gt, mask=None):
    model_out =  model_output['rgb'].cpu().numpy()
    gt = gt['rgb'].cpu().numpy()

    model_out += 1.
    model_out /= 2.
    model_out = np.clip(model_out, 0., 1.)

    gt += 1.
    gt /= 2.
    gt = np.clip(gt, 0., 1.)

    val_loss = np.mean((model_out - gt)**2)

    return {'img_loss': val_loss}


def val_image_semantic(model_output, gt, mask=None):

    num_labels = gt["num_extra_channels"][0]

    model_out =  model_output['model_out'][:,:,:-num_labels].cpu().numpy()
    gt_img = gt['y'][:,:,:-1].cpu().numpy()

    model_out += 1.
    model_out /= 2.
    model_out = np.clip(model_out, 0., 1.)

    gt_img += 1.
    gt_img /= 2.
    gt_img = np.clip(gt_img, 0., 1.)

    img_val_loss = np.mean((model_out - gt_img)**2)

    # model_out = model_output['model_out'][:, :, -num_labels:].cpu().numpy()
    # print("shape: ", np.shape(gt['y']), np.shape(model_output['model_out']))
    # gt_sem = gt['y'][:, :, -1:].cpu().numpy()

    pred_logits = model_output['model_out'][:, :, -num_labels:].flatten(start_dim=0, end_dim=1)
    labels = gt['y'][:, :, -1:].flatten().long()
    sem_val_loss = torch.nn.functional.cross_entropy(pred_logits.cuda(), labels.cuda())

    # model_out += 1.
    # model_out /= 2.
    # model_out = np.clip(model_out, 0., 1.)
    #
    # gt_sem += 1.
    # gt_sem /= 2.
    # gt_sem = np.clip(gt_sem, 0., 1.)
    #
    # sem_val_loss = np.mean((model_out - gt_sem) ** 2)

    return {'img_loss': img_val_loss, "semantic_loss": sem_val_loss}


def depth_mse(model_output, gt, mask=None):
    pred_depth = model_output['model_out'][:, :, -1:]
    gt_depth = gt['y'][:, :, -1:]
    if mask is None:
        loss = ((pred_depth.cuda() - gt_depth.cuda()) ** 2)
    else:
        loss = (mask.cuda() * (pred_depth.cuda() - gt_depth.cuda()) ** 2)

    return {'img_loss': loss.mean()}

def val_depth_mse(model_output, gt, mask=None):
    model_out =  model_output['model_out'][:, :, -1:].cpu().numpy()
    gt = gt['y'][:, :, -1:].cpu().numpy()

    model_out += 1.
    model_out /= 2.
    model_out = np.clip(model_out, 0., 1.)

    gt += 1.
    gt /= 2.
    gt = np.clip(gt, 0., 1.)

    val_loss = np.mean((model_out - gt)**2)

    return {'img_loss': val_loss}

def latent_loss(model_output):
    return torch.mean(model_output['latent_vec'] ** 2)


def hypo_weight_loss(model_output):
    weight_sum = 0
    total_weights = 0

    for weight in model_output['hypo_params'].values():
        weight_sum += torch.sum(weight ** 2)
        total_weights += weight.numel()

    return weight_sum * (1 / total_weights)


def image_hypernetwork_loss(mask, kl, fw, model_output, gt):
    return {'img_loss': image_mse(mask, model_output, gt)['img_loss'],
            'latent_loss': kl * latent_loss(model_output),
            'hypo_weight_loss': fw * hypo_weight_loss(model_output)}


def function_mse(model_output, gt):
    return {'func_loss': ((model_output['model_out'] - gt['func']) ** 2).mean()}


def gradients_mse(model_output, gt):
    # compute gradients on the model
    gradients = diff_operators.gradient(model_output['model_out'], model_output['model_in'])
    # compare them with the ground-truth
    gradients_loss = torch.mean((gradients - gt['gradients']).pow(2).sum(-1))
    return {'gradients_loss': gradients_loss}


def gradients_color_mse(model_output, gt):
    # compute gradients on the model
    gradients_r = diff_operators.gradient(model_output['model_out'][..., 0], model_output['model_in'])
    gradients_g = diff_operators.gradient(model_output['model_out'][..., 1], model_output['model_in'])
    gradients_b = diff_operators.gradient(model_output['model_out'][..., 2], model_output['model_in'])
    gradients = torch.cat((gradients_r, gradients_g, gradients_b), dim=-1)
    # compare them with the ground-truth
    weights = torch.tensor([1e1, 1e1, 1., 1., 1e1, 1e1]).cuda()
    gradients_loss = torch.mean((weights * (gradients[0:2] - gt['gradients']).pow(2)).sum(-1))
    return {'gradients_loss': gradients_loss}


def laplace_mse(model_output, gt):
    # compute laplacian on the model
    laplace = diff_operators.laplace(model_output['model_out'], model_output['model_in'])
    # compare them with the ground truth
    laplace_loss = torch.mean((laplace - gt['laplace']) ** 2)
    return {'laplace_loss': laplace_loss}


def wave_pml(model_output, gt):
    source_boundary_values = gt['source_boundary_values']
    x = model_output['model_in']  # (meta_batch_size, num_points, 3)
    y = model_output['model_out']  # (meta_batch_size, num_points, 1)
    squared_slowness = gt['squared_slowness']
    dirichlet_mask = gt['dirichlet_mask']
    batch_size = x.shape[1]

    du, status = diff_operators.jacobian(y, x)
    dudt = du[..., 0]

    if torch.all(dirichlet_mask):
        diff_constraint_hom = torch.Tensor([0])
    else:
        hess, status = diff_operators.jacobian(du[..., 0, :], x)
        lap = hess[..., 1, 1, None] + hess[..., 2, 2, None]
        dudt2 = hess[..., 0, 0, None]
        diff_constraint_hom = dudt2 - 1 / squared_slowness * lap

    dirichlet = y[dirichlet_mask] - source_boundary_values[dirichlet_mask]
    neumann = dudt[dirichlet_mask]

    return {'dirichlet': torch.abs(dirichlet).sum() * batch_size / 1e1,
            'neumann': torch.abs(neumann).sum() * batch_size / 1e2,
            'diff_constraint_hom': torch.abs(diff_constraint_hom).sum()}


def helmholtz_pml(model_output, gt):
    source_boundary_values = gt['source_boundary_values']

    if 'rec_boundary_values' in gt:
        rec_boundary_values = gt['rec_boundary_values']

    wavenumber = gt['wavenumber'].float()
    x = model_output['model_in']  # (meta_batch_size, num_points, 2)
    y = model_output['model_out']  # (meta_batch_size, num_points, 2)
    squared_slowness = gt['squared_slowness'].repeat(1, 1, y.shape[-1] // 2)
    batch_size = x.shape[1]

    full_waveform_inversion = False
    if 'pretrain' in gt:
        pred_squared_slowness = y[:, :, -1] + 1.
        if torch.all(gt['pretrain'] == -1):
            full_waveform_inversion = True
            pred_squared_slowness = torch.clamp(y[:, :, -1], min=-0.999) + 1.
            squared_slowness_init = torch.stack((torch.ones_like(pred_squared_slowness),
                                                 torch.zeros_like(pred_squared_slowness)), dim=-1)
            squared_slowness = torch.stack((pred_squared_slowness, torch.zeros_like(pred_squared_slowness)), dim=-1)
            squared_slowness = torch.where((torch.abs(x[..., 0, None]) > 0.75) | (torch.abs(x[..., 1, None]) > 0.75),
                                           squared_slowness_init, squared_slowness)
        y = y[:, :, :-1]

    du, status = diff_operators.jacobian(y, x)
    dudx1 = du[..., 0]
    dudx2 = du[..., 1]

    a0 = 5.0

    # let pml extend from -1. to -1 + Lpml and 1 - Lpml to 1.0
    Lpml = 0.5
    dist_west = -torch.clamp(x[..., 0] + (1.0 - Lpml), max=0)
    dist_east = torch.clamp(x[..., 0] - (1.0 - Lpml), min=0)
    dist_south = -torch.clamp(x[..., 1] + (1.0 - Lpml), max=0)
    dist_north = torch.clamp(x[..., 1] - (1.0 - Lpml), min=0)

    sx = wavenumber * a0 * ((dist_west / Lpml) ** 2 + (dist_east / Lpml) ** 2)[..., None]
    sy = wavenumber * a0 * ((dist_north / Lpml) ** 2 + (dist_south / Lpml) ** 2)[..., None]

    ex = torch.cat((torch.ones_like(sx), -sx / wavenumber), dim=-1)
    ey = torch.cat((torch.ones_like(sy), -sy / wavenumber), dim=-1)

    A = modules.compl_div(ey, ex).repeat(1, 1, dudx1.shape[-1] // 2)
    B = modules.compl_div(ex, ey).repeat(1, 1, dudx1.shape[-1] // 2)
    C = modules.compl_mul(ex, ey).repeat(1, 1, dudx1.shape[-1] // 2)

    a, _ = diff_operators.jacobian(modules.compl_mul(A, dudx1), x)
    b, _ = diff_operators.jacobian(modules.compl_mul(B, dudx2), x)

    a = a[..., 0]
    b = b[..., 1]
    c = modules.compl_mul(modules.compl_mul(C, squared_slowness), wavenumber ** 2 * y)

    diff_constraint_hom = a + b + c
    diff_constraint_on = torch.where(source_boundary_values != 0.,
                                     diff_constraint_hom - source_boundary_values,
                                     torch.zeros_like(diff_constraint_hom))
    diff_constraint_off = torch.where(source_boundary_values == 0.,
                                      diff_constraint_hom,
                                      torch.zeros_like(diff_constraint_hom))
    if full_waveform_inversion:
        data_term = torch.where(rec_boundary_values != 0, y - rec_boundary_values, torch.Tensor([0.]).cuda())
    else:
        data_term = torch.Tensor([0.])

        if 'pretrain' in gt:  # we are not trying to solve for velocity
            data_term = pred_squared_slowness - squared_slowness[..., 0]

    return {'diff_constraint_on': torch.abs(diff_constraint_on).sum() * batch_size / 1e3,
            'diff_constraint_off': torch.abs(diff_constraint_off).sum(),
            'data_term': torch.abs(data_term).sum() * batch_size / 1}


def sdf(model_output, gt):
    '''
       x: batch of input coordinates
       y: usually the output of the trial_soln function
       '''
    gt_sdf = gt['sdf']
    gt_normals = gt['normals']

    coords = model_output['model_in']
    pred_sdf = model_output['model_out']

    gradient = diff_operators.gradient(pred_sdf, coords)

    # Wherever boundary_values is not equal to zero, we interpret it as a boundary constraint.
    sdf_constraint = torch.where(gt_sdf != -1, pred_sdf, torch.zeros_like(pred_sdf))
    inter_constraint = torch.where(gt_sdf != -1, torch.zeros_like(pred_sdf), torch.exp(-1e2 * torch.abs(pred_sdf)))
    normal_constraint = torch.where(gt_sdf != -1, 1 - F.cosine_similarity(gradient, gt_normals, dim=-1)[..., None],
                                    torch.zeros_like(gradient[..., :1]))
    grad_constraint = torch.abs(gradient.norm(dim=-1) - 1)
    # Exp      # Lapl
    # -----------------
    return {'sdf': torch.abs(sdf_constraint).mean() * 3e3,  # 1e4      # 3e3
            'inter': inter_constraint.mean() * 1e2,  # 1e2                   # 1e3
            'normal_constraint': normal_constraint.mean() * 1e2,  # 1e2
            'grad_constraint': grad_constraint.mean() * 5e1}  # 1e1      # 5e1

# inter = 3e3 for ReLU-PE


class GANLoss(nn.Module):
    def __init__(self):
        super(GANLoss, self).__init__()
        self.eps = 1e-9

    def __call__(self, input, target_is_real):
        if target_is_real:
            return -1.*torch.mean(torch.log(input + self.eps))
        else:
            return -1.*torch.mean(torch.log(1 - input + self.eps))


