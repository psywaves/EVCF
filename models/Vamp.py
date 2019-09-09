from __future__ import print_function

import numpy as np

import math

from scipy.misc import logsumexp


import torch
import torch.utils.data
import torch.nn as nn
from torch.nn import Linear
from torch.autograd import Variable
from torch.nn.functional import normalize

from utils.distributions import log_Bernoulli, log_Normal_diag, log_Normal_standard, log_Logistic_256, log_Softmax
from utils.nn import he_init, GatedDense, NonLinear

from models.Model import Model
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

#=======================================================================================================================
class VAE(Model):
    def __init__(self, args):
        super(VAE, self).__init__(args)

        # encoder: q(z | x)
        modules = [nn.Dropout(p=0.5),
                   NonLinear(np.prod(self.args.input_size), self.args.hidden_size, gated=self.args.gated, activation=nn.Tanh())]
        for _ in range(0, self.args.num_layers - 1):
            modules.append(NonLinear(self.args.hidden_size, self.args.hidden_size, gated=self.args.gated, activation=nn.Tanh()))
        self.q_z_layers = nn.Sequential(*modules)


        self.q_z_mean = Linear(self.args.hidden_size, self.args.z1_size)
        self.q_z_logvar = NonLinear(self.args.hidden_size, self.args.z1_size, activation=nn.Hardtanh(min_val=-12.,max_val=4.))

        # decoder: p(x | z)
        modules = [NonLinear(self.args.z1_size, self.args.hidden_size, gated=self.args.gated, activation=nn.Tanh())]
        for _ in range(0, self.args.num_layers - 1):
            modules.append(NonLinear(self.args.hidden_size, self.args.hidden_size, gated=self.args.gated, activation=nn.Tanh()))
        self.p_x_layers = nn.Sequential(*modules)


        if self.args.input_type == 'binary':
            self.p_x_mean = NonLinear(self.args.hidden_size, np.prod(self.args.input_size), activation=nn.Sigmoid())
        if self.args.input_type == 'multinomial':
            self.p_x_mean = NonLinear(self.args.hidden_size, np.prod(self.args.input_size), activation=None)
        elif self.args.input_type == 'gray' or self.args.input_type == 'continuous':
            self.p_x_mean = NonLinear(self.args.hidden_size, np.prod(self.args.input_size), activation=nn.Sigmoid())
            self.p_x_logvar = NonLinear(self.args.hidden_size, np.prod(self.args.input_size), activation=None)

        # weights initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                he_init(m)

        # add pseudo-inputs for VampPrior
        self.add_pseudoinputs()

    # AUXILIARY METHODS
    def calculate_loss(self, x, beta=1., average=False):
        '''
        :param x: input image(s)
        :param beta: a hyperparam for warmup
        :param average: whether to average loss or not
        :return: value of a loss function
        '''
        # pass through VAE
        x_mean, x_logvar, z_q, z_q_mean, z_q_logvar = self.forward(x)

        # RE
        if self.args.input_type == 'binary':
            RE = log_Bernoulli(x, x_mean, dim=1)
        elif self.args.input_type == 'multinomial':
            RE = log_Softmax(x, x_mean, dim=1) #! Actually not Reconstruction Error but Log-Likelihood
        elif self.args.input_type == 'gray' or self.args.input_type == 'continuous':
            RE = -log_Logistic_256(x, x_mean, x_logvar, dim=1)
        else:
            raise Exception('Wrong input type!')

        # KL
        log_p_z = self.log_p_z(z_q)
        log_q_z = log_Normal_diag(z_q, z_q_mean, z_q_logvar, dim=1)
        KL = -(log_p_z - log_q_z)

        loss = - RE + beta * KL

        if average:
            loss = torch.mean(loss)
            RE = torch.mean(RE)
            KL = torch.mean(KL)

        return loss, RE, KL

    # ADDITIONAL METHODS
    def reconstruct_x(self, x):
        x_mean, _, _, _, _ = self.forward(x)
        return x_mean

    # THE MODEL: VARIATIONAL POSTERIOR
    def q_z(self, x):
        x = self.q_z_layers(x)

        z_q_mean = self.q_z_mean(x)
        z_q_logvar = self.q_z_logvar(x)
        return z_q_mean, z_q_logvar

    # THE MODEL: GENERATIVE DISTRIBUTION
    def p_x(self, z):
        z = self.p_x_layers(z)

        x_mean = self.p_x_mean(z)
        if self.args.input_type == 'binary' or self.args.input_type == 'multinomial':
            x_logvar = 0.
        else:
            x_mean = torch.clamp(x_mean, min=0.+1./512., max=1.-1./512.)
            x_logvar = self.p_x_logvar(z)
        return x_mean, x_logvar

    # the prior
    def log_p_z(self, z):
        # vamp prior
        # z - MB x M
        C = self.args.number_components

        # calculate params
        X = self.means(self.idle_input)

        # calculate params for given data
        z_p_mean, z_p_logvar = self.q_z(X)  # C x M

        # expand z
        z_expand = z.unsqueeze(1)
        means = z_p_mean.unsqueeze(0)
        logvars = z_p_logvar.unsqueeze(0)

        a = log_Normal_diag(z_expand, means, logvars, dim=2) - math.log(C)  # MB x C
        a_max, _ = torch.max(a, 1)  # MB x 1

        # calculte log-sum-exp
        log_prior = a_max + torch.log(torch.sum(torch.exp(a - a_max.unsqueeze(1)), 1))  # MB x 1

        return log_prior

    # THE MODEL: FORWARD PASS
    def forward(self, x):
        # input normalization & dropout
        x = normalize(x, dim=1)

        # z ~ q(z | x)
        z_q_mean, z_q_logvar = self.q_z(x)
        z_q = self.reparameterize(z_q_mean, z_q_logvar) #! train/test distinction -> built into reparameterize function

        # x_mean = p(x|z)
        x_mean, x_logvar = self.p_x(z_q)

        return x_mean, x_logvar, z_q, z_q_mean, z_q_logvar
