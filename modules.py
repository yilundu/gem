import loss_functions
from torchmeta.modules import (MetaModule, MetaSequential)
from collections import OrderedDict
import math
import numpy as np

import operator
from functools import reduce
from functools import partial

import torch
from torch import nn, einsum
import torch.nn.functional as F


class BatchLinear(nn.Linear, MetaModule):
    '''A linear meta-layer that can deal with batched weight matrices and biases, as for instance output by a
    hypernetwork.'''

    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        weight = params['weight']
        output = input.matmul(weight.permute(*[i for i in range(len(weight.shape) - 2)], -1, -2))

        if self.bias is not None:
            bias = params.get('bias', None)
            output += bias.unsqueeze(-2)

        return output


class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.sigmoid(input)


class FCLayer(MetaModule):
    def __init__(self, in_features, out_features, nonlinearity='relu', dropout=0.0):
        super().__init__()
        self.net = [BatchLinear(in_features, out_features)]
        if nonlinearity == 'relu':
            self.net.append(nn.ReLU(inplace=True))
        elif nonlinearity == 'leaky_relu':
            self.net.append(nn.LeakyReLU(0.2, inplace=True))
        elif nonlinearity == 'silu':
            self.net.append(Swish())

        self.dropout = dropout
        self.dropout_layer = torch.nn.Dropout(dropout, False)

        self.net = MetaSequential(*self.net)

    def forward(self, input, params=None):
        output = self.net(input, params=self.get_subdict(params, 'net'))
        if self.dropout != 0.0:
            output = self.dropout_layer(output)

        return output


class FCBlock(MetaModule):
    def __init__(self,
                 hidden_ch,
                 num_hidden_layers,
                 in_features,
                 out_features,
                 final_bias=True,
                 outermost_linear=False,
                 nonlinearity='relu',
                 dropout=0.0):
        super().__init__()

        self.net = []
        self.net.append(FCLayer(in_features=in_features, out_features=hidden_ch, nonlinearity=nonlinearity, dropout=dropout))

        for i in range(num_hidden_layers):
            self.net.append(FCLayer(in_features=hidden_ch, out_features=hidden_ch, nonlinearity=nonlinearity, dropout=dropout))

        if outermost_linear:
            self.net.append(BatchLinear(in_features=hidden_ch, out_features=out_features, bias=final_bias))
        else:
            self.net.append(FCLayer(in_features=hidden_ch, out_features=out_features, nonlinearity=nonlinearity))

        self.net = MetaSequential(*self.net)
        self.net.apply(init_weights_normal)

    def forward(self, input, params=None):
        return self.net(input, params=self.get_subdict(params, 'net'))


class SineLayer(MetaModule):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.

    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30, trainable_omega_0=False, init='custom'):
        super().__init__()
        print(omega_0)

        self.trainable_omega_0 = trainable_omega_0
        if trainable_omega_0:
            self.omega_0 = nn.Parameter(torch.Tensor([float(omega_0)]))
        else:
            self.register_buffer('omega_0', torch.Tensor([float(omega_0)]))

        self.is_first = is_first

        self.in_features = in_features
        self.linear = BatchLinear(in_features, out_features, bias=bias)

        self.init = init
        self.init_weights()

    def init_weights(self):
        print(self.init)
        if self.init == 'custom':
            with torch.no_grad():
                if self.is_first:
                    self.linear.weight.uniform_(-1 / self.in_features,
                                                 1 / self.in_features)
                else:
                    self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0.detach().cpu().numpy()[0],
                                                 np.sqrt(6 / self.in_features) / self.omega_0.detach().cpu().numpy()[0])
        else:
            nn.init.xavier_normal_(self.linear.weight)

    def forward_with_film(self, input, gamma, beta):
        intermed = self.linear(input)
        if self.init == 'custom':
            return torch.sin(gamma * self.omega_0 * intermed + beta)
        else:
            return torch.sin(intermed)

    def forward(self, input, params=None):
        intermed = self.linear(input, params=self.get_subdict(params, 'linear'))
        if self.init == 'custom':
            return torch.sin(self.omega_0 * intermed)
        else:
            return torch.sin(intermed)


class PosEncoding(MetaModule):
    def __init__(self, in_features, out_features, omega_0=30):
        super().__init__()
        self.linear = BatchLinear(in_features, out_features)
        self.omega_0 = omega_0

    def forward(self, input, params=None):
        if params is None:
            params = self.meta_parameters()

        intermed = self.omega_0 * self.linear(input, params=self.get_subdict(params, 'linear'))
        return torch.cat([torch.sin(intermed), torch.cos(intermed)], dim=-1)


class PosEncodingReLU(MetaModule):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False,
                 first_omega_0=30, share_first_layer=False, tanh_output=False, sigmoid_output=False, audiovisual=False, audio_omega_0=100):
        super().__init__()
        self.net = []
        self.net.append(PosEncoding(in_features=in_features, out_features=hidden_features, omega_0=first_omega_0))
        self.tanh_output = tanh_output
        self.sigmoid_output = sigmoid_output

        for i in range(hidden_layers):
            if not i:
                in_feats = 2*hidden_features
            else:
                in_feats = hidden_features
            self.net.append(FCLayer(in_features=in_feats, out_features=hidden_features))

        if outermost_linear:
            final_linear = BatchLinear(hidden_features, out_features)
            nn.init.xavier_normal_(final_linear.weight)
            self.net.append(final_linear)
        else:
            self.net.append(FCLayer(hidden_features, out_features))

        self.audiovisual = audiovisual
        self.net = nn.ModuleList(self.net)

        if self.audiovisual:
            self.audio_net = []
            self.audio_net.append(PosEncoding(in_features=2, out_features=hidden_features, omega_0=audio_omega_0))

            for i in range(hidden_layers):
                if not i:
                    in_feats = 2*hidden_features
                else:
                    in_feats = hidden_features
                self.audio_net.append(FCLayer(in_features=in_feats, out_features=hidden_features))

            if outermost_linear:
                final_linear = BatchLinear(hidden_features, 1)
                nn.init.xavier_normal_(final_linear.weight)
                self.audio_net.append(final_linear)
            else:
                self.audio_net.append(FCLayer(hidden_features, 1))

            self.audio_net = nn.ModuleList(self.audio_net)

    # def forward(self, coords, params=None):
    #     x = coords
    #
    #     for i, layer in enumerate(self.net):
    #         x = layer(x, params=self.get_subdict(params, f'net.{i}'))
    #     return x


    def forward(self, coords, audio_coords=None, params=None, share_first_layer=False):
        module_params = dict(self.meta_named_parameters())

        if params is None: params = module_params

        x = coords

        # check if use module's init for first layer and hypernet for all else
        if share_first_layer:
            x = self.net[0](x, params=self.get_subdict(module_params, f'net.{0}'))
        else:
            x = self.net[0](x, params=self.get_subdict(params, f'net.{0}'))

        for i, layer in enumerate(self.net):
            if i == 0: continue
            x = layer(x, params=self.get_subdict(params, f'net.{i}'))

        if self.tanh_output:
            x = F.tanh(x)

        if self.sigmoid_output:
            x = F.sigmoid(x)

        if audio_coords is not None:

            audio_x  = audio_coords
            if share_first_layer:
                audio_x = self.audio_net[0](audio_x, params=self.get_subdict(module_params, f'audio_net.{0}'))
            else:
                audio_x = self.audio_net[0](audio_x, params=self.get_subdict(params, f'audio_net.{0}'))

            for i, layer in enumerate(self.audio_net):
                if i == 0: continue
                audio_x = layer(audio_x, params=self.get_subdict(params, f'audio_net.{i}'))

            if self.tanh_output:
                audio_x = F.tanh(audio_x)

            if self.sigmoid_output:
                audio_x = F.sigmoid(audio_x)

            return x, audio_x
        else:
            return x


class Siren(MetaModule):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False,
                 first_omega_0=30, hidden_omega_0=30., hidden_omega_0_mode='static', special_first=True,
                 init='custom', tanh_output=False, sigmoid_output=False):
        super().__init__()
        self.hidden_omega_0 = hidden_omega_0

        self.tanh_output = tanh_output
        self.sigmoid_output = sigmoid_output

        layer = SineLayer

        self.net = []
        self.net.append(layer(in_features, hidden_features,
                              is_first=special_first, omega_0=first_omega_0, init=init))

        for i in range(hidden_layers):
            self.net.append(layer(hidden_features, hidden_features,
                                  is_first=False, omega_0=hidden_omega_0,
                                  trainable_omega_0=hidden_omega_0_mode=='per_layer',
                                  init=init))

        if outermost_linear:
            final_linear = BatchLinear(hidden_features, out_features)

            if init == 'custom':
                with torch.no_grad():
                    final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / 30.,
                                                  np.sqrt(6 / hidden_features) / 30.)
            else:
                nn.init.xavier_normal_(final_linear.weight)


            self.net.append(final_linear)
        else:
            self.net.append(layer(hidden_features, out_features, is_first=False, omega_0=hidden_omega_0))

        self.net = nn.ModuleList(self.net)

    def get_param_count(self):
        count = 0
        for param in self.parameters():
            count += np.prod(param.shape)
        return count

    def forward_with_activations(self, coords, params=None):
        x = coords
        activations = []

        for i, layer in enumerate(self.net):
            x = layer(x, params=self.get_subdict(params, f'net.{i}'))
            activations.append(x)

        return activations

    def forward_with_film(self, coords, film):
        x = coords

        for i, (layer, layer_film) in enumerate(zip(self.net, film)):
            if i < len(self.net) - 1:
                x = layer.forward_with_film(x, layer_film['gamma'], layer_film['beta'])
            else:
                x = layer.forward(x)

        return x

    # def forward(self, coords, params=None):
    #     if params is None:
    #         params = dict(self.meta_named_parameters())
    #
    #     x = coords
    #
    #     for i, layer in enumerate(self.net):
    #         x = layer(x, params=self.get_subdict(params, f'net.{i}'))
    #
    #     return x

    def forward(self, coords, params=None, share_first_layer=False):
        siren_params = dict(self.meta_named_parameters())

        if params is None: params = siren_params

        x = coords

        # check if use siren's init for first layer and hypernet for all else
        if share_first_layer:
            x = self.net[0](x, params=self.get_subdict(siren_params, f'net.{0}'))
        else:
            x = self.net[0](x, params=self.get_subdict(params, f'net.{0}'))

        for i, layer in enumerate(self.net):
            if i == 0: continue
            x = layer(x, params=self.get_subdict(params, f'net.{i}'))

        if self.tanh_output:
            x = F.tanh(x)

        if self.sigmoid_output:
            x = F.sigmoid(x)

        return x


########################
# Encoder modules
class SetEncoder(nn.Module):
    def __init__(self, in_features, out_features,
                 num_hidden_layers, hidden_features):
        super().__init__()

        nl = nn.ReLU(inplace=True)
        weight_init = init_weights_normal

        self.net = [nn.Linear(in_features, hidden_features), nl]
        self.net.extend([nn.Sequential(nn.Linear(hidden_features, hidden_features), nl)
                         for _ in range(num_hidden_layers)])
        self.net.extend([nn.Linear(hidden_features, out_features), nl])
        self.net = nn.Sequential(*self.net)

        self.net.apply(weight_init)

    def forward(self, context_x, context_y, ctxt_mask=None, **kwargs):
        input = torch.cat((context_x, context_y), dim=-1)
        embeddings = self.net(input)

        if ctxt_mask is not None:
            embeddings = embeddings * ctxt_mask
            embedding = embeddings.mean(dim=-2) * (embeddings.shape[-2] / torch.sum(ctxt_mask, dim=-2))
            return embedding
        return embeddings.mean(dim=-2)
    
    
    
#######################
# Wavelet Stuff
class EnvelopedPeriodic(MetaModule):
    def __init__(self,in_features,out_features,env_function, per_function, env_weight_init, per_weight_init):
        super().__init__()

        self.env_function = env_function(in_features,out_features,env_weight_init)
        self.per_linear = BatchLinear(in_features,out_features)
        self.per_linear.apply(per_weight_init)
        self.per_function = per_function

    def forward(self, input, params=None):
        return self.env_function(input, params=self.get_subdict(params, 'env_function'))\
                    * self.per_function(self.per_linear(input, params=self.get_subdict(params, 'per_linear')))

class Sine(nn.Module):
    def __init(self):
        super().__init__()

    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(30 * input)

class Gaussian1d(MetaModule):
    def __init__(self, in_features, out_features, weight_init):
        super().__init__()
        self.linear = BatchLinear(in_features,out_features)
        self.linear.apply(weight_init)

    def forward(self, input, params=None):
        return torch.exp(-torch.pow(self.linear(input, params=self.get_subdict(params, 'linear')), 2) / (20 ** 2))

class Gaussian2d(MetaModule):
    def __init__(self, in_features, out_features, weight_init):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = BatchLinear(in_features, out_features)
        self.linear.apply(weight_init)

    def forward(self, input, params=None):
        return torch.exp(-1 * self.linear(input**2, params=self.get_subdict(params, 'linear')))

    
class EnvelopedPeriodicBlock(MetaModule):
    '''A fully connected neural network that also allows swapping out the weights when used with a hypernetwork.
        Can be used just as a normal neural network though, as well.
        Activations have a periodic factor and an envelope factor whose inputs are determined by two different linear layers.
    '''
    def __init__(self, in_features, out_features, num_hidden_layers, hidden_features,
                 outermost_linear=False, nonlinearity='gabor2d', env_weight_init=None, per_weight_init = None):
        super().__init__()

        self.first_layer_init = None

        # Dictionary that maps nonlinearity name to the respective function, initialization, and, if applicable,
        # special first-layer initialization scheme
        nls_and_inits = {'gabor2d':(Gaussian2d, Sine(), gaussian2d_init, sine_init, first_layer_gaussian2d_init, first_layer_sine_init),
                         'gabor1d':(Gaussian1d, Sine(), gaussian1d_init, sine_init, None, first_layer_sine_init)}

        nl_env, nl_per, nl_env_weight_init, nl_per_weight_init, env_first_layer_init, per_first_layer_init = nls_and_inits[nonlinearity]

        if env_weight_init is not None:  # Overwrite weight init if passed
            self.env_weight_init = env_weight_init
        else:
            self.env_weight_init = nl_env_weight_init

        if per_weight_init is not None:  # Overwrite weight init if passed
            self.per_weight_init = per_weight_init
        else:
            self.per_weight_init = nl_per_weight_init

        if env_first_layer_init is not None:
            self.env_first_layer_init = env_first_layer_init
        else:
            self.env_first_layer_init = self.env_weight_init

        if per_first_layer_init is not None:
            self.per_first_layer_init = per_first_layer_init
        else:
            self.per_first_layer_init = self.per_weight_init

        self.net = []

        self.net.append(
            EnvelopedPeriodic(in_features,hidden_features,nl_env,nl_per,self.env_first_layer_init,self.per_first_layer_init)
        )

        for i in range(num_hidden_layers):
            self.net.append(
                EnvelopedPeriodic(hidden_features,hidden_features,nl_env,nl_per,self.env_weight_init,self.per_weight_init)
            )

        if outermost_linear:
            final_linear = BatchLinear(hidden_features, out_features)
            final_linear.apply(self.per_weight_init)
            self.net.append(final_linear)
        else:
            self.net.append(
                EnvelopedPeriodic(hidden_features,out_features,nl_env,nl_per,self.env_weight_init,self.per_weight_init)
            )

        self.net = MetaSequential(*self.net)

    def forward(self, coords, params=None, **kwargs):
        if params is None:
            params = OrderedDict(self.named_parameters())

        output = self.net(coords, params=self.get_subdict(params, 'net'))
        return output

    

    
################################################################
# fourier layer
################################################################

def compl_mul2d(a, b):
    op = partial(torch.einsum, "bctq,dctq->bdtq")
    return torch.stack([
        op(a[..., 0], b[..., 0]) - op(a[..., 1], b[..., 1]),
        op(a[..., 1], b[..., 0]) + op(a[..., 0], b[..., 1])
    ], dim=-1)


class SpectralConv2d_fast(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d_fast, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, 2))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, 2))

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.rfft(x, 2, normalized=True, onesided=True)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.in_channels, x.size(-2), x.size(-1)//2 + 1, 2, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.irfft(out_ft, 2, normalized=True, onesided=True, signal_sizes=(x.size(-2), x.size(-1)))
        return x


class SimpleBlock2d(nn.Module):
    def __init__(self, modes1, modes2, width, in_channels, out_channels):
        super(SimpleBlock2d, self).__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.fc0 = nn.Linear(in_channels, self.width)

        self.conv0 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)
        self.bn0 = torch.nn.BatchNorm2d(self.width)
        self.bn1 = torch.nn.BatchNorm2d(self.width)
        self.bn2 = torch.nn.BatchNorm2d(self.width)
        self.bn3 = torch.nn.BatchNorm2d(self.width)


        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x):
        batchsize = x.shape[0]
        size_x, size_y = x.shape[1], x.shape[2]

        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        x1 = self.conv0(x)
        x2 = self.w0(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = self.bn0(x1 + x2)
        x = F.relu(x)
        x1 = self.conv1(x)
        x2 = self.w1(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = self.bn1(x1 + x2)
        x = F.relu(x)
        x1 = self.conv2(x)
        x2 = self.w2(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = self.bn2(x1 + x2)
        x = F.relu(x)
        x1 = self.conv3(x)
        x2 = self.w3(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = self.bn3(x1 + x2)


        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

class Net2d(nn.Module):
    def __init__(self, modes, width, in_channels, out_channels):
        super(Net2d, self).__init__()

        self.conv1 = SimpleBlock2d(modes, modes, width, in_channels=in_channels, out_channels=out_channels)

    def forward(self, x):
        x = self.conv1(x)
        return x

    def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))

        return c
    


########################
# Initialization methods
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # For PINNet, Raissi et al. 2019
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    # grab from upstream pytorch branch and paste here for now
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def init_weights_trunc_normal(m):
    # For PINNet, Raissi et al. 2019
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            fan_in = m.weight.size(1)
            fan_out = m.weight.size(0)
            std = math.sqrt(2.0 / float(fan_in + fan_out))
            mean = 0.
            # initialize with the same behavior as tf.truncated_normal
            # "The generated values follow a normal distribution with specified mean and
            # standard deviation, except that values whose magnitude is more than 2
            # standard deviations from the mean are dropped and re-picked."
            _no_grad_trunc_normal_(m.weight, mean, std, -2 * std, 2 * std)


def init_weights_normal(m):
    if hasattr(m, 'weight'):
        nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')

    if hasattr(m, 'bias') and m.bias is not None:
        with torch.no_grad():
            m.bias.data = torch.randn_like(m.bias)*1e-2


def init_weights_selu(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            nn.init.normal_(m.weight, std=1 / math.sqrt(num_input))


def init_weights_elu(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            nn.init.normal_(m.weight, std=math.sqrt(1.5505188080679277) / math.sqrt(num_input))


def init_weights_xavier(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.xavier_normal_(m.weight)


def sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-1 / num_input, 1 / num_input)


###################
# Complex operators
def compl_conj(x):
    y = x.clone()
    y[..., 1::2] = -1 * y[..., 1::2]
    return y


def compl_div(x, y):
    ''' x / y '''
    a = x[..., ::2]
    b = x[..., 1::2]
    c = y[..., ::2]
    d = y[..., 1::2]

    outr = (a * c + b * d) / (c ** 2 + d ** 2)
    outi = (b * c - a * d) / (c ** 2 + d ** 2)
    out = torch.zeros_like(x)
    out[..., ::2] = outr
    out[..., 1::2] = outi
    return out


def compl_mul(x, y):
    '''  x * y '''
    a = x[..., ::2]
    b = x[..., 1::2]
    c = y[..., ::2]
    d = y[..., 1::2]

    outr = a * c - b * d
    outi = (a + b) * (c + d) - a * c - b * d
    out = torch.zeros_like(x)
    out[..., ::2] = outr
    out[..., 1::2] = outi
    return out


def gaussian2d_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, 0.5*np.sqrt(6 / num_input) / 30)

def first_layer_gaussian2d_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-1 / num_input, 0.5*1 / num_input)


def gaussian1d_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            stdv = math.sqrt(num_input)
            m.weight.uniform_(-stdv, stdv)
        if hasattr(m, 'bias'):
            num_input = m.weight.size(-1)
            stdv = math.sqrt(num_input)
            m.bias.data.uniform_(-stdv, stdv)

