'''Modules for hypernetwork experiments, Paper Sec. 4.4
'''

import torch.nn.functional as F
import torch
from torch import nn
from torchmeta.modules import (MetaModule, MetaSequential)
from collections import OrderedDict
import modules
import numpy as np


class LowRankHyperNetwork(nn.Module):
    def __init__(self, hyper_in_features, hyper_hidden_layers, hyper_hidden_features, hypo_module, linear=False,
                 rank=10):
        '''

        Args:
            hyper_in_features: In features of hypernetwork
            hyper_hidden_layers: Number of hidden layers in hypernetwork
            hyper_hidden_features: Number of hidden units in hypernetwork
            hypo_module: MetaModule. The module whose parameters are predicted.
        '''
        super().__init__()

        self.hypo_parameters = dict(hypo_module.meta_named_parameters())
        self.representation_dim = 0

        self.rank = rank
        self.names = []
        self.nets = nn.ModuleList()
        self.param_shapes = []
        for name, param in self.hypo_parameters.items():
            self.names.append(name)
            self.param_shapes.append(param.size())

            out_features = int(torch.prod(torch.tensor(param.size()))) if 'bias' in name else param.shape[0]*rank + param.shape[1]*rank
            self.representation_dim += out_features

            hn = modules.FCBlock(in_features=hyper_hidden_features, out_features=out_features,
                                 num_hidden_layers=hyper_hidden_layers, hidden_ch=hyper_hidden_features,
                                 outermost_linear=True)
            self.nets.append(hn)

    def forward(self, z):
        '''
        Args:
            z: Embedding. Input to hypernetwork. Could be output of "Autodecoder" (see above)

        Returns:
            params: OrderedDict. Can be directly passed as the "params" parameter of a MetaModule.
        '''
        params = OrderedDict()
        representation = []
        for name, net, param_shape in zip(self.names, self.nets, self.param_shapes):
            low_rank_params = net(z)
            representation.append(low_rank_params.detach())

            if 'bias' in name:
                batch_param_shape = (-1,) + param_shape
                params[name] = low_rank_params.reshape(batch_param_shape)
            else:
                a = low_rank_params[:, :self.rank*param_shape[0]].view(-1, param_shape[0], self.rank)
                b = low_rank_params[:, self.rank*param_shape[0]:].view(-1, self.rank, param_shape[1])
                low_rank_w = a.matmul(b)
                params[name] = self.hypo_parameters[name] * torch.sigmoid(low_rank_w)

        representations = representation
        representation = torch.cat(representation, dim=-1).cuda()
        return {'params':params, 'representation':representation, 'representations': representations}

    def gen_params(self, representation):
        params = OrderedDict()
        start = 0

        for name, net, param_shape in zip(self.names, self.nets, self.param_shapes):
            if 'bias' in name:
                nelem = np.prod(param_shape)
            else:
                nelem = param_shape[0] * self.rank + param_shape[1] * self.rank

            low_rank_params = representation[:, start:start+nelem]

            if 'bias' in name:
                batch_param_shape = (-1,) + param_shape
                params[name] = low_rank_params.reshape(batch_param_shape)
            else:
                a = low_rank_params[:, :self.rank*param_shape[0]].view(-1, param_shape[0], self.rank)
                b = low_rank_params[:, self.rank*param_shape[0]:].view(-1, self.rank, param_shape[1])
                low_rank_w = a.matmul(b)
                params[name] = self.hypo_parameters[name] * torch.sigmoid(low_rank_w)

            start = start + nelem

        return {'params':params, 'representation':representation}


class LowRankDiscriminator(nn.Module):
    def __init__(self, in_features, hypo_module):
        super().__init__()
        self.hypo_parameters = dict(hypo_module.meta_named_parameters())
        self.nbatch = 1
        self.discriminator = modules.FCBlock(in_features=in_features * self.nbatch, out_features=1,
                                             num_hidden_layers=8, hidden_ch=512,
                                             outermost_linear=True, nonlinearity='relu')

        self.nets = nn.ModuleList()
        self.names = []
        self.param_shapes = []
        rank = 10
        for name, param in self.hypo_parameters.items():
            self.names.append(name)
            self.param_shapes.append(param.size())

            out_features = int(torch.prod(torch.tensor(param.size()))) if 'bias' in name else param.shape[0]*rank + param.shape[1]*rank

            hn = modules.FCBlock(in_features=out_features * self.nbatch, out_features=1,
                                 num_hidden_layers=8, hidden_ch=512,
                                 outermost_linear=True)
            self.nets.append(hn)


    def reshape(self, input):
        s = input.size()
        input = input.view(s[0] // self.nbatch, self.nbatch * s[1])
        return input

    def forward(self, input, detach=False):
        '''
        Args:
            z: Embedding. Input to hypernetwork. Could be output of "Autodecoder" (see above)

        Returns:
            params: OrderedDict. Can be directly passed as the "params" parameter of a MetaModule.
        '''
        # representations = input['representations']
        input = input['representation']
        if detach:
            input = input.detach()

        discouts = []
        input = self.reshape(input)
        disc_out = self.discriminator(input)
        discouts.append(disc_out)

        # for representation, name, net, param_shape in zip(representations, self.names, self.nets, self.param_shapes):
        #     if detach:
        #         representation = representation.detach()
        #     representation = self.reshape(representation)
        #     disc_out_i = net(representation)
        #     discouts.append(disc_out_i)

        discouts = torch.sigmoid(torch.cat(discouts, dim=-1))
        return discouts


class FILMNetwork(nn.Module):
    def __init__(self, hypo_module, latent_dim, num_hidden=3):
        '''

        Args:
            hyper_in_features: In features of hypernetwork
            hyper_hidden_layers: Number of hidden layers in hypernetwork
            hyper_hidden_features: Number of hidden units in hypernetwork
            hypo_module: MetaModule. The module whose parameters are predicted.
        '''
        super().__init__()

        hypo_parameters = hypo_module.meta_named_parameters()

        self.names = []
        self.nets = nn.ModuleList()
        self.param_shapes = []
        self.representation_dim = 0
        for name, param in hypo_parameters:
            self.names.append(name)
            self.param_shapes.append(param.size())

            hn = modules.FCBlock(in_features=latent_dim, out_features=int(2*torch.tensor(param.shape[0])),
                                 num_hidden_layers=num_hidden, hidden_ch=latent_dim, outermost_linear=True,
                                 nonlinearity='relu')
            # hn.net[-1].apply(lambda m: hyper_weight_init(m, param.size()[-1]))
            self.nets.append(hn)
            self.representation_dim += int(2*torch.tensor(param.shape[0]))

    def forward(self, z):
        params = []
        representation = []
        for name, net, param_shape in zip(self.names, self.nets, self.param_shapes):
            net_out = net(z)
            layer_params = {}
            layer_params['gamma'] = net_out[:, :param_shape[0]].unsqueeze(1) + 1
            layer_params['beta'] = net_out[:, param_shape[0]:].unsqueeze(1)
            representation.append(layer_params['gamma']-1.)
            representation.append(layer_params['beta'])
            params.append(layer_params)

        representation = torch.cat(representation, dim=-1)

        return {'params':params, 'representation':representation, 'representations': representation}


class HyperNetwork(nn.Module):
    def __init__(self, hyper_in_features, hyper_hidden_layers, hyper_hidden_features, hypo_module, linear=False):
        '''

        Args:
            hyper_in_features: In features of hypernetwork
            hyper_hidden_layers: Number of hidden layers in hypernetwork
            hyper_hidden_features: Number of hidden units in hypernetwork
            hypo_module: MetaModule. The module whose parameters are predicted.
        '''
        super().__init__()

        hypo_parameters = hypo_module.meta_named_parameters()

        self.names = []
        self.nets = nn.ModuleList()
        self.param_shapes = []
        for name, param in hypo_parameters:
            print(name)
            self.names.append(name)
            self.param_shapes.append(param.size())

            if linear:
                hn = modules.BatchLinear(in_features=hyper_in_features,
                                         out_features=int(torch.prod(torch.tensor(param.size()))),
                                         bias=True)
                if 'weight' in name:
                    hn.apply(lambda m: hyper_weight_init(m, param.size()[-1]))
                elif 'bias' in name:
                    hn.apply(lambda m: hyper_bias_init(m))
            else:
                hn = modules.FCBlock(in_features=hyper_in_features, out_features=int(torch.prod(torch.tensor(param.size()))),
                                     num_hidden_layers=hyper_hidden_layers, hidden_ch=hyper_hidden_features,
                                     outermost_linear=True, nonlinearity='relu')
                if 'weight' in name:
                    hn.net[-1].apply(lambda m: hyper_weight_init(m, param.size()[-1]))
                elif 'bias' in name:
                    hn.net[-1].apply(lambda m: hyper_bias_init(m))
            self.nets.append(hn)

    def forward(self, z):
        '''
        Args:
            z: Embedding. Input to hypernetwork. Could be output of "Autodecoder" (see above)

        Returns:
            params: OrderedDict. Can be directly passed as the "params" parameter of a MetaModule.
        '''
        params = OrderedDict()
        for name, net, param_shape in zip(self.names, self.nets, self.param_shapes):
            batch_param_shape = (-1,) + param_shape
            params[name] = net(z).reshape(batch_param_shape)
        return params


############################
# Initialization scheme
def hyper_weight_init(m, in_features_main_net):
    if hasattr(m, 'weight'):
        nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')
        m.weight.data = m.weight.data / 1.e2

    if hasattr(m, 'bias') and m.bias is not None:
        with torch.no_grad():
            m.bias.zero_()


def hyper_bias_init(m):
    if hasattr(m, 'weight'):
        nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')
        m.weight.data = m.weight.data / 1.e2

    if hasattr(m, 'bias') and m.bias is not None:
        with torch.no_grad():
            m.bias.zero_()
