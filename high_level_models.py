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
                 in_features=2, out_features=3, amortized=False, first_omega_0=30, share_first_layer=False, noise=False, pos_encode=False, tanh_output=False, sigmoid_output=False, type='linear', manifold_dim=100, film=False, audiovisual=False):
        super().__init__()
        self.latent_dim = latent_dim
        self.amortized = amortized
        self.type = type

        # overwrite keywords if film
        if film:
            tanh_output = False
            pos_encode = False

        if amortized:
            self.encoder = conv_modules.ConvImgEncoder(channel=3, in_sidelength=64, out_features=latent_dim,
                                                       outermost_linear=True)

        self.num_items = num_items
        self.latents = nn.Embedding(num_embeddings=num_items, embedding_dim=self.latent_dim)
        self.latents.weight.data = torch.randn_like(self.latents.weight) * 1e-2

        self.test_num_items = test_num_items

        if test_num_items != 0:
            self.test_latents = nn.Embedding(num_embeddings=test_num_items, embedding_dim=self.latent_dim)
            self.test_latents.weight.data = torch.randn_like(self.test_latents.weight) * 1e-2

        self.audiovisual = audiovisual

        self.manifold_dim = manifold_dim

        # self.per_element_factor = nn.Parameter(torch.ones((1, self.latent_dim)))
        # self.per_element_bias = nn.Parameter(torch.zeros((1, self.latent_dim)))
        self.register_parameter('per_element_factor',
                                nn.Parameter(torch.ones(1, self.latent_dim) + 1e-2 * torch.randn(1, self.latent_dim), requires_grad=True))
        self.register_parameter('per_element_bias',
                                nn.Parameter(torch.zeros(1, self.latent_dim) + 1e-2 * torch.randn(1, self.latent_dim), requires_grad=True))

        self.register_parameter('scale_factor',
                                nn.Parameter(torch.ones(1, 1) * 100, requires_grad=True))

        if pos_encode:
            self.img_siren = modules.PosEncodingReLU(in_features=in_features, out_features=out_features, hidden_features=hidden_features,
                                           hidden_layers=hidden_layers, outermost_linear=True,first_omega_0=first_omega_0, tanh_output=tanh_output, sigmoid_output=sigmoid_output, audiovisual=audiovisual)
        else:
            self.img_siren = modules.Siren(in_features=in_features, out_features=out_features, hidden_features=hidden_features,
                                           hidden_layers=hidden_layers, outermost_linear=True,first_omega_0=first_omega_0, tanh_output=tanh_output, sigmoid_output=sigmoid_output)

        self.generator = modules.FCBlock(hidden_ch=hidden_features, num_hidden_layers=8, in_features=self.latent_dim,
                                         out_features=self.latent_dim, outermost_linear=True)
        self.reference = modules.FCBlock(hidden_ch=self.latent_dim, num_hidden_layers=3, in_features=self.latent_dim,
                                         out_features=self.latent_dim, outermost_linear=True)

        self.film = film

        if film:
            self.hypernet = meta_modules.FILMNetwork(hypo_module=self.img_siren, latent_dim=latent_dim, num_hidden=3,
                                                    )
        else:
            self.hypernet = meta_modules.LowRankHyperNetwork(hypo_module=self.img_siren, hyper_hidden_layers=0,
                                                             hyper_hidden_features=latent_dim, hyper_in_features=latent_dim)


        self.share_first_layer = share_first_layer
        self.noise = noise
        self.nselect = 1
        self.dist = torch.distributions.dirichlet.Dirichlet(torch.ones(self.nselect))

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

    def gen_noisy_latent(self, input):
        z_orig = z = self.latents(input['context']['idx'])[:, 0, :]
        flip_mask = ((torch.rand_like(z) > 0.5).float() - 0.5) * 2.

        with torch.no_grad():
            latents = self.latents.weight
            rix = torch.randperm(z.size(1))
            chunks = max(z.size(0) // 16, 1)
            zs = torch.chunk(z_orig, chunks, 0)
            idxs = []

            for zi in zs:
                diff = zi[:, None] - latents[None, :]

                diff_val = torch.norm(diff, p='fro', dim=-1)
                _, idx = torch.topk(diff_val, 101, dim=-1, largest=False)
                idxs.append(idx)

            idx = torch.cat(idxs, dim=0)

            # Don't include the current entry'
            idx = idx[:, 1:]
            idx = idx[:, :, None].repeat(1, 1, z.size(1))

            s = idx.size()
            idx = idx.view(-1, s[-1])
            latents_dense = torch.gather(latents, 0, idx)
            latents_select = latents_dense.view(s[0], s[1], s[2])
            # latents_select = torch.stack([torch.gather(latents, 0, idxi) for idxi in idx], dim=0)
            latents_center =  latents_select

            latents_max = latents_center.max(dim=1)[0]
            latents_min = latents_center.min(dim=1)[0]

            latents_diff = latents_max - latents_min

            z_perturb = z + (10.0 + torch.rand_like(z) * 1.2) * latents_diff[:, :] / 16 * flip_mask

            latents_permute = latents_select.permute(0, 2, 1)

            dot_matrix = torch.bmm(latents_select, latents_permute)
            # ones = torch.ones(latents_center.size(0), latents_center.size(1), 1).to(latents_center.device)
            latents_map = torch.bmm(latents_select, z_perturb[:, :, None])

            dot_matrix_inv = torch.inverse(dot_matrix)
            weights = torch.bmm(dot_matrix_inv, latents_map)
            # weights_sum = weights.sum(dim=1, keepdim=True)
            # weights = weights / weights_sum

            # Regenerate with grad
            latents_linear = (weights * latents_select).sum(dim=1)
            z = latents_linear

        return z

    def forward(self, input, prior_sample=False, mix_sample=False, render=True, closest_idx=False, manifold_model=True):
        out_dict = {}

        latent_chunk = 1

        if prior_sample:
            # z = torch.randn((input['context']['idx'].shape[0], self.latent_dim)).to(input['context']['idx'].device)

            sf = 1
            idx = input['context']['idx']
            os = idx.size(0)
            # perm_idx = torch.randperm(29000)[:idx.size(0) * sf].to(idx.device)
            perm_idx = torch.randperm(self.num_items)[:idx.size(0) * sf].to(idx.device)

            z_orig = z = self.latents(perm_idx)

            nselect = self.nselect

            with torch.no_grad():
                latents = self.latents.weight
                rix = torch.randperm(z.size(1))
                chunks = max(z.size(0) // 16, 1)
                zs = torch.chunk(z_orig, chunks, 0)
                idxs = []
                diff_dists = []
                max_dists = []

                for zi in zs:
                    diff = zi[:, None] - latents[None, :]
                    s = diff.size()
                    diff =  diff.view(s[0], s[1], latent_chunk, -1)

                    diff_val = torch.norm(diff, p='fro', dim=-1)
                    diff_dist, idx = torch.topk(diff_val, self.manifold_dim+1, dim=1, largest=False)
                    max_dist, _ = torch.topk(diff_val, self.manifold_dim+1, dim=1, largest=True)

                    diff_dists.append(diff_dist)
                    max_dists.append(max_dist)
                    idxs.append(idx)

                idx = torch.cat(idxs, dim=0)
                diff_dist = torch.cat(diff_dists, dim=0)
                max_dist = torch.cat(max_dists, dim=0)
                max_idx = torch.argsort(max_dist[:, 1], dim=0).squeeze()

                select_idx = max_idx[:os]

                idx = idx[select_idx]
                z = z[select_idx]
                # import pdb
                # pdb.set_trace()
                # print(diff_val)

                # Don't include the current entry'
                idx = idx[:, :].permute(0, 2, 1).contiguous()
                idx = idx[:, :, :nselect, None].repeat(1, 1, 1, z.size(1))

            s = idx.size()
            idx = idx.view(-1, s[-1])
            latents_dense = torch.gather(latents, 0, idx)
            latents_dense = latents_dense.view(s[0], s[1], s[2], s[-1])

            select_idx = torch.arange(s[3]).view(latent_chunk, -1).to(latents_dense.device)
            # select_idx = select_idx[None, :, None, :].repeat(s[0], 1, self.manifold_dim, 1)
            select_idx = select_idx[None, :, None, :].repeat(s[0], 1, nselect, 1)

            subset_latent = torch.gather(latents_dense, -1, select_idx)
            weights = self.dist.sample(torch.Size([subset_latent.size(0)]))
            # weights = torch.rand_like(subset_latent[..., :1])
            # weights = weights / weights.sum(dim=-2, keepdim=True)
            weights = weights.to(subset_latent.device)
            weights = weights[:, None, :, None]

            latents_linear = (weights * subset_latent).sum(dim=-2).view(-1, z.size(-1))

            # ix = random.randint(0, latent_chunk-1)
            # latents_linear = (z + latents_dense[:, ix, 0]) / 2.
            # latents_linear = z

            ############## New sampling code assuming with chunks
            # z_orig = z = self.latents(input['context']['idx'].squeeze())
            # flip_mask = ((torch.rand_like(z) > 0.5).float() - 0.5) * 2.

            # bs = z_orig.size(-1) // latent_chunk
            # i = random.randint(0, 7)
            # z_perturb = z.clone()
            # z_perturb = z + (0.002 * torch.rand_like(z) + 0.003) * flip_mask
            # # z_perturb[i*bs:(i+1)*bs] = z[i*bs:(i+1)*bs] + (0.002 * torch.rand_like(z[i*bs:(i+1)*bs]) + 0.02) * flip_mask[i*bs:(i+1)*bs]

            # z = z_perturb

            # with torch.no_grad():
            #     latents = self.latents.weight
            #     rix = torch.randperm(z.size(1))
            #     chunks = max(z.size(0) // 64, 1)
            #     zs = torch.chunk(z_orig, chunks, 0)
            #     idxs = []

            #     for zi in zs:
            #         diff = zi[:, None] - latents[None, :]
            #         s = diff.size()
            #         diff =  diff.view(s[0], s[1], latent_chunk, -1)

            #         diff_val = torch.norm(diff, p='fro', dim=-1)
            #         _, idx = torch.topk(diff_val, self.manifold_dim+1, dim=1, largest=False)
            #         idxs.append(idx)

            #     idx = torch.cat(idxs, dim=0)

            #     # Don't include the current entry'
            #     idx = idx[:, 1:].permute(0, 2, 1).contiguous()
            #     idx = idx[:, :, :, None].repeat(1, 1, 1, z.size(1))

            # s = idx.size()
            # idx = idx.view(-1, s[-1])
            # latents_dense = torch.gather(latents, 0, idx)

            # latents_select = latents_dense.view(s[0], s[1], s[2], s[3])
            # select_idx = torch.arange(s[3]).view(latent_chunk, -1).to(latents_select.device)
            # select_idx = select_idx[None, :, None, :].repeat(s[0], 1, self.manifold_dim, 1)
            # subset_latent = torch.gather(latents_select, -1, select_idx)

            # s = subset_latent.size()
            # latents_select = subset_latent.view(s[0] * s[1], s[2], s[3])

            # latents_permute = latents_select.permute(0, 2, 1)
            # z_dense = z.view(-1, latents_select.size(-1))
            # latents_map = torch.bmm(latents_select, z_dense[:, :, None])

            # dot_matrix = torch.bmm(latents_select, latents_permute)
            # dot_matrix_inv = torch.inverse(dot_matrix)
            # weights = torch.bmm(dot_matrix_inv, latents_map)

            # # Regenerate with grad
            # latents_linear = (weights * latents_select).sum(dim=1)
            # latents_linear = latents_linear.view(-1, latents.size(-1))


            ############## Old sampling code assuming no chunks
            # latents_select = latents_dense.view(s[0], s[1], s[2])
            # # latents_select = torch.stack([torch.gather(latents, 0, idxi) for idxi in idx], dim=0)
            # latents_center =  latents_select

            # latents_max = latents_center.max(dim=1)[0]
            # latents_min = latents_center.min(dim=1)[0]

            # latents_diff = latents_max - latents_min

            # z_perturb = z + (0.2 + torch.rand_like(z) * 1.2) * latents_diff[:, :] / 16 * flip_mask

            # latents_permute = latents_select.permute(0, 2, 1)

            # dot_matrix = torch.bmm(latents_select, latents_permute)
            # # ones = torch.ones(latents_center.size(0), latents_center.size(1), 1).to(latents_center.device)
            # latents_map = torch.bmm(latents_select, z_perturb[:, :, None])

            # dot_matrix_inv = torch.inverse(dot_matrix)
            # weights = torch.bmm(dot_matrix_inv, latents_map)
            # # weights_sum = weights.sum(dim=1, keepdim=True)
            # # weights = weights / weights_sum

            # # Regenerate with grad
            # latents_linear = (weights * latents_select).sum(dim=1)
            # z = latents_linear


            generator_out = self.reference(latents_linear)
            out_dict['representation'] = generator_out
            output = self.hypernet(generator_out)
            out_dict['representations'] = output['representations']
            params = output['params']

            if closest_idx:
                dist = torch.norm(self.latents.weight[None, :, :] - z[:, None, :], 2, dim=-1)
                sort_idx = torch.argsort(dist, dim=-1)[:, :5].detach().cpu().numpy()
                out_dict['sort_idx'] = sort_idx
        elif mix_sample:
            z = self.latents(input['context']['idx'])[:, 0, :]
            dist = torch.norm(self.latents.weight[None, :, :] - z[:, None, :], dim=-1)

            out_dict['dist_hist'] = dist[0]
            sort_idx = torch.argsort(dist, dim=-1)[:, :5]

            idx_near = sort_idx[:, 1]
            z_other = self.latents(idx_near)
            z_mid = (z + z_other) / 2.
            z = torch.cat([z, z_mid, z_other], dim=0)
            z = z

            generator_out = self.reference(z)
            out_dict['representation'] = generator_out
            output = self.hypernet(generator_out)
            out_dict['representations'] = output['representations']
            params = output['params']
        else:

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

                # latents_select = torch.stack([torch.gather(latents, 0, idxi) for idxi in idx], dim=0)
                # latents_center =  latents_select - z[:, None, :]

                latents_permute = latents_select.permute(0, 2, 1)

                dot_matrix = torch.bmm(latents_select, latents_permute)
                # ones = torch.ones(latents_center.size(0), latents_center.size(1), 1).to(latents_center.device)
                latents_map = torch.bmm(latents_select, z[:, :, None])

                dot_matrix_inv = torch.inverse(dot_matrix)
                weights = torch.bmm(dot_matrix_inv, latents_map)
                # weights_sum = weights.sum(dim=1, keepdim=True)
                # weights = weights / weights_sum

                # Regenerate with grad
                latents_linear = (weights * latents_select).sum(dim=1)
                generator_out_linear = self.reference(latents_linear)
                output_linear = self.hypernet(generator_out_linear)
                params_linear = output_linear['params']

                if self.audiovisual:
                    model_out_linear = self.img_siren(input['context']['x'], audio_coords=input['context']['audio_coord'], params=params_linear, share_first_layer=self.share_first_layer)
                else:
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

                if self.audiovisual:
                    model_out_linear = self.img_siren(input['context']['x'], input['context']['audio_coord'], params=params_linear, share_first_layer=self.share_first_layer)
                else:
                    if self.film:
                        model_out_linear = self.img_siren.forward_with_film(input['context']['x'], params_linear)
                    else:
                        model_out_linear = self.img_siren(input['context']['x'], params=params_linear, share_first_layer=self.share_first_layer)
                out_dict['model_out_linear'] = model_out_linear
                out_dict['rgb_linear'] = model_out_linear
                out_dict['latents_dist'] = torch.norm(latents_select[:, :100] - z_dense[:, None, :], p=2, dim=-1)
                out_dict['weights'] = weights

                # Encd of code linear loss


            if self.amortized:
                z += self.encoder(input['context']['rgb'].permute(0,2,1).view(-1, 3, 64, 64))

            if len(z.size()) == 1:
                z = z[None, :]

            generator_out = self.reference(z)
            generator_out = generator_out
            output = self.hypernet(generator_out)
            out_dict['representation'] = generator_out
            # out_dict['representations'] = output['representations']
            out_dict['z'] = z
            params = output['params']


            if self.amortized:
                z += self.encoder(input['context']['rgb'].permute(0,2,1).view(-1, 3, 64, 64))

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

        if self.audiovisual:
            model_out = self.img_siren(input['context']['x'], audio_coords=input['context']['audio_coord'], params=params, share_first_layer=self.share_first_layer)
        else:
            if self.film:
                model_out = self.img_siren.forward_with_film(input['context']['x'], params)
            else:
                model_out = self.img_siren(input['context']['x'], params=params, share_first_layer=self.share_first_layer)

        out_dict['rgb'] = model_out
        out_dict['model_out'] = model_out
        out_dict['scale_factor'] = self.scale_factor
        return out_dict


class SirenGAN(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.latent_dim = latent_dim

        self.img_siren = modules.Siren(in_features=2, out_features=3, hidden_features=256,
                                       hidden_layers=3, outermost_linear=True, init='custom')
        self.film = meta_modules.FILMNetwork(hypo_module=self.img_siren, latent_dim=latent_dim)
        # self.film = meta_modules.HyperNetwork(hyper_in_features=self.latent_dim, hyper_hidden_layers=3,
        #                                       hyper_hidden_features=256, hypo_module=self.img_siren)
        self.cuda()
        print(self)

    def forward(self, input, render=True, prior_sample=True):
        out_dict = {}
        z = torch.randn((input['context']['x'].shape[0], self.latent_dim)).cuda()
        film_params = self.film(z)
        out_dict['fast_params'] = film_params

        if not render:
            return out_dict

        model_out = self.img_siren(input['context']['x'], params=film_params)
        # model_out = self.img_siren(input['context']['x'], params=film_params)
        out_dict['rgb'] = model_out[..., :3]
        out_dict['model_out'] = model_out
        return out_dict



class SirenVAD(nn.Module):
    def __init__(self, num_items, latent_dim=256, hidden_layers=6, hidden_features=512,
                 in_features=2, out_features=3, first_omega_0=30, share_first_layer=False, gan_training=False):
        super().__init__()
        self.gan_training = gan_training
        self.latent_dim = latent_dim

        self.latents = nn.Embedding(num_embeddings=num_items, embedding_dim=self.latent_dim*2)
        nn.init.normal_(self.latents.weight.data[..., :latent_dim], mean=0, std=0.01)
        nn.init.constant_(self.latents.weight.data[..., latent_dim:], np.log(1e-4))

        self.register_parameter('per_element_factor',
                                nn.Parameter(torch.ones(1, 2*self.latent_dim) + 1e-2 * torch.randn(1, 2*self.latent_dim), requires_grad=True))
        self.register_parameter('per_element_bias',
                                nn.Parameter(torch.zeros(1, 2*self.latent_dim) + 1e-2 * torch.randn(1, 2*self.latent_dim), requires_grad=True))

        self.img_siren = modules.PosEncodingReLU(in_features=in_features, out_features=out_features,
                                                 hidden_features=hidden_features, hidden_layers=hidden_layers,
                                                 outermost_linear=True,first_omega_0=first_omega_0)

        self.fc_decoder = modules.FCBlock(hidden_ch=self.latent_dim, num_hidden_layers=3, in_features=self.latent_dim,
                                          out_features=self.latent_dim, outermost_linear=True, dropout=0.0)
        self.hypernet = meta_modules.LowRankHyperNetwork(hypo_module=self.img_siren, hyper_hidden_layers=0,
                                                         hyper_hidden_features=latent_dim, hyper_in_features=latent_dim)
        self.generator = modules.FCBlock(hidden_ch=512, num_hidden_layers=8, in_features=self.latent_dim,
                                         out_features=self.latent_dim, outermost_linear=True)

        self.share_first_layer = share_first_layer
        self.cuda()

    def forward(self, input, render=True, prior_sample=False, closest_idx=False):
        out_dict = {}

        if prior_sample:
            z = torch.randn((input['context']['idx'].shape[0], self.latent_dim)).to(input['context']['idx'].device)

            if closest_idx:
                dist = torch.norm((self.latents.weight[None, :, :self.latent_dim] * self.per_element_factor[None, :, :self.latent_dim] + self.per_element_bias[None, :, :self.latent_dim]) - z[:, None, :], 2, dim=-1)
                sort_idx = torch.argsort(dist, dim=-1, descending=True)[:, :5].detach().cpu().numpy()
                out_dict['sort_idx'] = sort_idx

            if self.gan_training:
                generator_out = self.generator(z) * 1e-1
                generator_out = self.fc_decoder(generator_out)
                out_dict['representation'] = generator_out
                output = self.hypernet(generator_out)
                out_dict['representations'] = output['representations']
                params = output['params']
            else:
                generator_out = self.fc_decoder(z)
                output = self.hypernet(generator_out)
                params = output['params']
        else:
            latent = self.latents(input['context']['idx'].squeeze()) * self.per_element_factor + self.per_element_bias
            mean = latent[..., :self.latent_dim]
            logvar = latent[..., self.latent_dim:]

            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)

            z = eps * std + mean

            if closest_idx:
                dist = torch.norm((self.latents.weight[None, :, :self.latent_dim] * self.per_element_factor[None, :, :self.latent_dim] + self.per_element_bias[None, :, :self.latent_dim]) - z[:, None, :], 2, dim=-1)
                sort_idx = torch.argsort(dist, dim=-1, descending=True)[:, :5].detach().cpu().numpy()
                out_dict['sort_idx'] = sort_idx

            out_dict.update({'z': z, 'mu': mean, 'logvar': logvar})

            output = self.hypernet(self.fc_decoder(z))
            if self.gan_training:
                out_dict['representation'] = z
                out_dict['representations'] = output['representations']
            params = output['params']

        if not render:
            return out_dict

        model_out = self.img_siren(input['context']['x'], params=params)
        out_dict['rgb'] = model_out[..., :3]
        out_dict['model_out'] = model_out
        return out_dict
