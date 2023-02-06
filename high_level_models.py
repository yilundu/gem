import numpy as np
import torch
import conv_modules
import utils
import modules
import meta_modules
from torch import nn
import loss_functions
import torch.nn.functional as F
import random


class SirenImplicitGAN(nn.Module):
    def __init__(self, num_items, test_num_items=0, latent_dim=256, hidden_layers=6, hidden_features=512,
                 in_features=2, out_features=3, first_omega_0=30, share_first_layer=False, pos_encode=True, tanh_output=False, sigmoid_output=False, type='linear', manifold_dim=100):
        super().__init__()
        self.latent_dim = latent_dim
        self.type = type

        self.num_items = num_items
        self.latents = nn.Embedding(num_embeddings=num_items, embedding_dim=self.latent_dim)
        self.latents.weight.data = torch.randn_like(self.latents.weight) * 1e-2

        self.test_num_items = test_num_items

        if test_num_items != 0:
            self.test_latents = nn.Embedding(num_embeddings=test_num_items, embedding_dim=self.latent_dim)
            self.test_latents.weight.data = torch.randn_like(self.test_latents.weight) * 1e-2

        # Determines number of nearby latents define the linear manifold
        self.manifold_dim = manifold_dim

        self.register_parameter('scale_factor',
                                nn.Parameter(torch.ones(1, 1) * 100, requires_grad=True))

        self.img_siren = modules.PosEncodingReLU(in_features=in_features, out_features=out_features, hidden_features=hidden_features,
                                       hidden_layers=hidden_layers, outermost_linear=True,first_omega_0=first_omega_0, tanh_output=tanh_output, sigmoid_output=sigmoid_output, audiovisual=False)

        self.generator = modules.FCBlock(hidden_ch=hidden_features, num_hidden_layers=8, in_features=self.latent_dim,
                                         out_features=self.latent_dim, outermost_linear=True)
        self.reference = modules.FCBlock(hidden_ch=self.latent_dim, num_hidden_layers=3, in_features=self.latent_dim,
                                         out_features=self.latent_dim, outermost_linear=True)

        self.hypernet = meta_modules.LowRankHyperNetwork(hypo_module=self.img_siren, hyper_hidden_layers=0,
                                                         hyper_hidden_features=latent_dim, hyper_in_features=latent_dim)

        self.share_first_layer = share_first_layer
        self.nselect = 1

        self.cuda()

    def freeze_gt(self):
        self.film.requires_grad_(False)

    def forward_with_latent(self, latent, input):
        generator_out = self.reference(latent)
        output = self.hypernet(generator_out)

        if self.audiovisual:
            model_out = self.img_siren(input['context']['x'], input['context']['audio_coord'], params=output['params'], share_first_layer=self.share_first_layer)
        else:
            model_out = self.img_siren(input['context']['x'],
                                       params=output['params'],
                                       share_first_layer=self.share_first_layer)
        return model_out

    def forward_with_intermediate_latent(self, latent, input):
        output = self.hypernet(latent)
        model_out = self.img_siren(input['context']['x'],
                                   params=output['params'],
                                   share_first_layer=self.share_first_layer)
        return model_out

    def forward(self, input, prior_sample=False, mix_sample=False, render=True, closest_idx=False, manifold_model=True):
        out_dict = {}



        if self.test_num_items > 0:
            z = self.test_latents(input['context']['idx'].squeeze())
        else:
            z = self.latents(input['context']['idx'].squeeze())
        out_dict['z_orig'] = z
        out_dict['latents'] = self.latents


        if len(z.size()) > 1 and manifold_model and (self.type == "linear"):
            # Code for the linear loss
            latents = self.latents.weight

            with torch.no_grad():
                # rix = torch.randperm(latents.size(1))[:128]

                chunks = max(z.size(0) // 8, 1)
                zs = torch.chunk(z, chunks, 0)
                idxs = []
                max_idxs = []

                for zi in zs:
                    diff = zi[:, None] - latents[None, :]

                    diff_val = torch.norm(diff, p='fro', dim=-1)
                    _, idx = torch.topk(diff_val, self.manifold_dim+1, dim=-1, largest=False)
                    _, max_idx = torch.topk(diff_val, self.manifold_dim+1, dim=-1)
                    idxs.append(idx)
                    max_idxs.append(max_idx)

                idx = torch.cat(idxs, dim=0)
                max_idx = torch.cat(max_idxs, dim=0)

                # Don't include the current entry
                idx = idx[:, 1:]
                idx = idx[:, :, None].repeat(1, 1, z.size(1))
                max_idx = max_idx[:, :, None].repeat(1, 1, z.size(1))

            s = idx.size()
            idx = idx.view(-1, s[-1])
            latents_dense = torch.gather(latents, 0, idx)

            latents_select = latents_dense.view(s[0], s[1], s[2])

            s = max_idx.size()
            max_idx = max_idx.view(-1, s[-1])
            latents_dense_max = torch.gather(latents, 0, max_idx)
            latents_dense_max = latents_dense_max.view(s[0], s[1], s[2])

            latents_permute = latents_select.permute(0, 2, 1)

            dot_matrix = torch.bmm(latents_select, latents_permute)
            latents_map = torch.bmm(latents_select, z[:, :, None])

            dot_matrix_inv = torch.inverse(dot_matrix)
            weights = torch.bmm(dot_matrix_inv, latents_map)

            # Regenerate with grad
            latents_linear = (weights * latents_select).sum(dim=1)
            generator_out_linear = self.reference(latents_linear)
            output_linear = self.hypernet(generator_out_linear)
            params_linear = output_linear['params']

            model_out_linear = self.img_siren(input['context']['x'], params=params_linear, share_first_layer=self.share_first_layer)
            out_dict['model_out_linear'] = model_out_linear
            out_dict['rgb_linear'] = model_out_linear[..., :3]
            out_dict['latents_dist'] = torch.norm(latents_select[:, :20] - z[:, None, :], p=2, dim=-1)
            out_dict['latents_dist_large'] = torch.norm(latents_dense_max[:, :] - z[:, None, :], p=2, dim=-1)

            # Encd of code linear loss
        elif len(z.size()) > 1 and manifold_model and (self.type == "linear_lle"):
            # Code for the linear loss
            latents = self.latents.weight

            with torch.no_grad():
                # rix = torch.randperm(latents.size(1))[:128]

                chunks = max(z.size(0) // 16, 1)
                zs = torch.chunk(z, chunks, 0)
                idxs = []

                for zi in zs:
                    diff = zi[:, None] - latents[None, :]
                    s = diff.size()
                    diff =  diff.view(s[0], s[1], latent_chunk, -1)

                    diff_val = torch.norm(diff, p='fro', dim=-1)
                    _, idx = torch.topk(diff_val, self.manifold_dim+1, dim=1, largest=False)
                    idxs.append(idx)

                idx = torch.cat(idxs, dim=0)

                # Don't include the current entry
                idx = idx[:, 1:].permute(0, 2, 1).contiguous()
                idx = idx[:, :, :, None].repeat(1, 1, 1, z.size(1))

            s = idx.size()
            idx = idx.view(-1, s[-1])
            latents_dense = torch.gather(latents, 0, idx)

            latents_select = latents_dense.view(s[0], s[1], s[2], s[3])
            select_idx = torch.arange(s[3]).view(latent_chunk, -1).to(latents_select.device)
            select_idx = select_idx[None, :, None, :].repeat(s[0], 1, self.manifold_dim, 1)
            subset_latent = torch.gather(latents_select, -1, select_idx)

            s = subset_latent.size()
            latents_select = subset_latent.view(s[0] * s[1], s[2], s[3])
            z_dense = z.view(-1, latents_select.size(-1))

            latents_center =  latents_select - z_dense[:, None, :]

            latents_permute = latents_center.permute(0, 2, 1)

            dot_matrix = torch.bmm(latents_center, latents_permute)
            ones = torch.ones(latents_center.size(0), latents_center.size(1), 1).to(latents_center.device)

            dot_matrix_inv = torch.inverse(dot_matrix)
            weights = torch.bmm(dot_matrix_inv, ones)
            weights_sum = weights.sum(dim=1, keepdim=True)
            weights = weights / weights_sum

            # Regenerate with grad
            latents_linear = (weights * latents_select).sum(dim=1)
            latents_linear = latents_linear.view(-1, latents.size(-1))

            generator_out_linear = self.reference(latents_linear)
            output_linear = self.hypernet(generator_out_linear)
            params_linear = output_linear['params']

            model_out_linear = self.img_siren(input['context']['x'], params=params_linear, share_first_layer=self.share_first_layer)
            out_dict['model_out_linear'] = model_out_linear
            out_dict['rgb_linear'] = model_out_linear
            out_dict['latents_dist'] = torch.norm(latents_select[:, :100] - z_dense[:, None, :], p=2, dim=-1)
            out_dict['weights'] = weights

            # Encd of code linear loss

        if len(z.size()) == 1:
            z = z[None, :]

        generator_out = self.reference(z)
        generator_out = generator_out
        output = self.hypernet(generator_out)
        out_dict['representation'] = generator_out
        # out_dict['representations'] = output['representations']
        out_dict['z'] = z
        params = output['params']

        if not render:
            return out_dict

        model_out = self.img_siren(input['context']['x'], params=params, share_first_layer=self.share_first_layer)

        out_dict['rgb'] = model_out
        out_dict['model_out'] = model_out
        out_dict['scale_factor'] = self.scale_factor
        return out_dict

