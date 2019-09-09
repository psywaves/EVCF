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
# H+Vamp model with 1 hidden layer must use this model
#=======================================================================================================================
class VAE(Model):
    def __init__(self, args):
        super(VAE, self).__init__(args)

        self.args = args

        # encoder: q(z2 | x)
        self.q_z2_layers = nn.Sequential(
            nn.Dropout(p=0.5),
            NonLinear(np.prod(self.args.input_size), self.args.hidden_size, gated=self.args.gated, activation=nn.Tanh()),
        )

        self.q_z2_mean = Linear(self.args.hidden_size, self.args.z2_size)
        self.q_z2_logvar = NonLinear(self.args.hidden_size, self.args.z2_size, activation=nn.Hardtanh(min_val=-12.,max_val=4.))

        # encoder: q(z1 | x, z2)
        self.q_z1_layers_x = nn.Sequential(
            nn.Dropout(p=0.5),
        )
        self.q_z1_layers_z2 = nn.Sequential(
        )
        self.q_z1_layers_joint = nn.Sequential(
            NonLinear(np.prod(self.args.input_size) + self.args.z2_size, self.args.hidden_size, gated=self.args.gated, activation=nn.Tanh())
        )

        self.q_z1_mean = Linear(self.args.hidden_size, self.args.z1_size)
        self.q_z1_logvar = NonLinear(self.args.hidden_size, self.args.z1_size, activation=nn.Hardtanh(min_val=-12.,max_val=4.))

        # decoder: p(z1 | z2)
        self.p_z1_layers = nn.Sequential(
            NonLinear(self.args.z2_size, self.args.hidden_size, gated=self.args.gated, activation=nn.Tanh()),
        )

        self.p_z1_mean = Linear(self.args.hidden_size, self.args.z1_size)
        self.p_z1_logvar = NonLinear(self.args.hidden_size, self.args.z1_size, activation=nn.Hardtanh(min_val=-12.,max_val=4.))

        # decoder: p(x | z1, z2)
        self.p_x_layers_z1 = nn.Sequential(
        )
        self.p_x_layers_z2 = nn.Sequential(
        )
        self.p_x_layers_joint = nn.Sequential(
            NonLinear(self.args.z1_size + self.args.z2_size, self.args.hidden_size, gated=self.args.gated, activation=nn.Tanh())
        )

        if self.args.input_type == 'binary':
            self.p_x_mean = NonLinear(self.args.hidden_size, np.prod(self.args.input_size), activation=nn.Sigmoid())
        if self.args.input_type == 'multinomial':
            self.p_x_mean = NonLinear(self.args.hidden_size, np.prod(self.args.input_size), activation=None)
        elif self.args.input_type == 'gray' or self.args.input_type == 'continuous':
            self.p_x_mean = NonLinear(self.args.hidden_size, np.prod(self.args.input_size), activation=nn.Sigmoid())
            self.p_x_logvar = NonLinear(self.args.hidden_size, np.prod(self.args.input_size), activation=nn.Hardtanh(min_val=-4.5,max_val=0))

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
        x_mean, x_logvar, z1_q, z1_q_mean, z1_q_logvar, z2_q, z2_q_mean, z2_q_logvar, z1_p_mean, z1_p_logvar = self.forward(x)

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
        log_p_z1 = log_Normal_diag(z1_q, z1_p_mean, z1_p_logvar, dim=1)
        log_q_z1 = log_Normal_diag(z1_q, z1_q_mean, z1_q_logvar, dim=1)
        log_p_z2 = self.log_p_z2(z2_q)
        log_q_z2 = log_Normal_diag(z2_q, z2_q_mean, z2_q_logvar, dim=1)
        KL = -(log_p_z1 + log_p_z2 - log_q_z1 - log_q_z2)

        loss = -RE + beta * KL

        if average:
            loss = torch.mean(loss)
            RE = torch.mean(RE)
            KL = torch.mean(KL)

        return loss, RE, KL

    # ADDITIONAL METHODS
    def reconstruct_x(self, x):
        x_mean, _, _, _, _, _, _, _, _, _ = self.forward(x)
        return x_mean

    # THE MODEL: VARIATIONAL POSTERIOR
    def q_z2(self, x):
        x = self.q_z2_layers(x)

        z2_q_mean = self.q_z2_mean(x)
        z2_q_logvar = self.q_z2_logvar(x)
        return z2_q_mean, z2_q_logvar

    def q_z1(self, x, z2):
        x = self.q_z1_layers_x(x)

        z2 = self.q_z1_layers_z2(z2)

        h = torch.cat((x,z2), 1)

        h = self.q_z1_layers_joint(h)

        z1_q_mean = self.q_z1_mean(h)
        z1_q_logvar = self.q_z1_logvar(h)
        return z1_q_mean, z1_q_logvar

    # THE MODEL: GENERATIVE DISTRIBUTION
    def p_z1(self, z2):
        z2 = self.p_z1_layers(z2)

        z1_mean = self.p_z1_mean(z2)
        z1_logvar = self.p_z1_logvar(z2)
        return z1_mean, z1_logvar

    def p_x(self, z1, z2):
        z1 = self.p_x_layers_z1(z1)

        z2 = self.p_x_layers_z2(z2)

        h = torch.cat((z1, z2), 1)

        h = self.p_x_layers_joint(h)

        x_mean = self.p_x_mean(h)
        if self.args.input_type == 'binary' or self.args.input_type == 'multinomial':
            x_logvar = 0.
        else:
            x_mean = torch.clamp(x_mean, min=0.+1./512., max=1.-1./512.)
            x_logvar = self.p_x_logvar(h)
        return x_mean, x_logvar

    # the prior
    def log_p_z2(self, z2):
        # vamp prior
        # z2 - MB x M
        C = self.args.number_components

        # calculate params
        X = self.means(self.idle_input)

        # calculate params for given data
        z2_p_mean, z2_p_logvar = self.q_z2(X)  # C x M

        # expand z
        z_expand = z2.unsqueeze(1)
        means = z2_p_mean.unsqueeze(0)
        logvars = z2_p_logvar.unsqueeze(0)

        a = log_Normal_diag(z_expand, means, logvars, dim=2) - math.log(C)  # MB x C
        a_max, _ = torch.max(a, 1)  # MB
        # calculte log-sum-exp
        log_prior = (a_max + torch.log(torch.sum(torch.exp(a - a_max.unsqueeze(1)), 1)))  # MB

        return log_prior

    # THE MODEL: FORWARD PASS
    def forward(self, x):
        # input normalization
        x = normalize(x, dim=1)

        # z2 ~ q(z2 | x)
        z2_q_mean, z2_q_logvar = self.q_z2(x)
        z2_q = self.reparameterize(z2_q_mean, z2_q_logvar)

        # z1 ~ q(z1 | x, z2)
        z1_q_mean, z1_q_logvar = self.q_z1(x, z2_q)
        z1_q = self.reparameterize(z1_q_mean, z1_q_logvar)

        # p(z1 | z2)
        z1_p_mean, z1_p_logvar = self.p_z1(z2_q)

        # x_mean = p(x|z1,z2)
        x_mean, x_logvar = self.p_x(z1_q, z2_q)

        return x_mean, x_logvar, z1_q, z1_q_mean, z1_q_logvar, z2_q, z2_q_mean, z2_q_logvar, z1_p_mean, z1_p_logvar
