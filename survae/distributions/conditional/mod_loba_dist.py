import math
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.distributions import VonMises
from torch.distributions import Independent
from torch.distributions import Uniform
from survae.distributions.conditional import ConditionalDistribution
from survae.utils import sum_except_batch
import numpy as np


class Loba_dist(ConditionalDistribution):
    """See https://github.com/SB-27182/loba_NN"""

    def __init__(self, radii_net, osci_spread_net, n_var_net, o_mean_net,
                 shift=torch.tensor([[0.0, 0.0]])):

        super(dOscicond_Ovarmean_Nvar_radi_dist, self).__init__()
        self.radii_net = radii_net
        self.osci_spread_net = osci_spread_net
        self.n_var_net = n_var_net
        self.o_mean_net = o_mean_net
        self.shift = shift


    def cond_dist(self, context):
        radii = self.radii_net(context)
        osci_spread = torch.abs(self.osci_spread_net(context))
        n_var = torch.abs(self.n_var_net(context))
        o_mean = self.o_mean_net(context)

        vm = Independent(VonMises(o_mean, osci_spread), 1)
        sample_inds_n = torch.squeeze(vm.sample([1]))

        x = self.shift[:, 0:1] + radii[:, 0:1]*torch.cos(sample_inds_n[:, 0:1])
        y = self.shift[:, 1:2] + radii[:, 1:2]*torch.cos(sample_inds_n[:, 1:2])*torch.sin(sample_inds_n[:, 0:1])
        n_mean = torch.cat((x, y), dim=1)

        return Normal(loc=n_mean, scale=n_var)


    def log_prob(self, x, context):
        dist = self.cond_dist(context)
        return sum_except_batch(dist.log_prob(x))


    def sample(self, context):
        #assert False, "Not Implemented"
        dist = self.cond_dist(context)
        return dist.rsample()


    def latent_sample_nVars(self, context, dims, myVarDim=0, scaleMyVarBy=1.0, scaleOthrVarsBy=0.01,
                      myOsciVarDim=0, scaleMyOsciVarBy=1.0, scaleOthrOsciVarsBy=1.0):
        radii = self.radii_net(context)
        osciVar = torch.abs(self.osci_spread_net(context))
        n_var = torch.abs(self.n_var_net(context))
        o_mean = self.o_mean_net(context)

        # <editor-fold desc="Oscil Variance Scaleing">
        scaleOsciVarDims = np.full([1, dims], scaleOthrOsciVarsBy)
        np.put(scaleOsciVarDims, [myOsciVarDim], scaleMyOsciVarBy)
        scaleOsciVarDims = torch.tensor(scaleOsciVarDims, dtype=torch.float32)
        osciVar = torch.mul(osciVar, scaleOsciVarDims)
        # </editor-fold>

        vm = VonMises(o_mean, osciVar)
        sample_inds = torch.squeeze(vm.sample([1]))

        x = self.shift[:, 0:1] + radii[:, 0:1] * torch.cos(sample_inds[:, 0:1])
        y = self.shift[:, 1:2] + radii[:, 1:2] * torch.cos(sample_inds[:, 1:2]) * torch.sin(sample_inds[:, 0:1])
        n_mean = torch.cat((x, y), dim=1)
        print(n_var)
        # <editor-fold desc="n Variance Scaleing">
        scaleVarDims = np.full([1, dims], scaleOthrVarsBy)
        np.put(scaleVarDims, [myVarDim], scaleMyVarBy)
        scaleVarDims = torch.tensor(scaleVarDims, dtype=torch.float32)
        n_var = torch.mul(n_var, scaleVarDims)
        # </editor-fold>

        dist = Normal(loc=n_mean, scale=n_var)

        z = dist.sample([1])
        return z






    def latent_sample_1DimOscil(self, context, dims, myDim, scaleMyVarBy=1.0, scaleOthrVarsBy=1.0,
                      myOsciVarDim=0, scaleMyOsciVarBy=1.0, scaleOthrOsciVarsBy=1.0):
        """Generates the latent corresponding to the found 1-dim oscilator in the data."""
        radii = self.radii_net(context)
        osciVar = torch.abs(self.osci_spread_net(context))
        n_var = torch.abs(self.n_var_net(context))
        o_mean = self.o_mean_net(context)


        # <editor-fold desc="Oscil Variance Scaleing">
        scaleOsciVarDims = np.full([1, dims], scaleOthrOsciVarsBy)
        np.put(scaleOsciVarDims, [myOsciVarDim], scaleMyOsciVarBy)
        scaleOsciVarDims = torch.tensor(scaleOsciVarDims, dtype=torch.float32)
        osciVar = torch.mul(osciVar, scaleOsciVarDims)
        # </editor-fold>

        vm = VonMises(o_mean, osciVar)
        sample_inds = torch.squeeze(vm.sample([1]))
        my_dim_variable = sample_inds[:, myDim:myDim+1]


        x = self.shift[:, 0:1] + radii[:, 0:1]*torch.cos(my_dim_variable)
        y = self.shift[:, 1:2] + radii[:, 1:2]*torch.sin(my_dim_variable)
        n_mean = torch.cat((x, y), dim=1)

        # <editor-fold desc="n Variance Scaleing">
        scaleVarDims = np.full([1, dims], scaleOthrVarsBy)
        np.put(scaleVarDims, [myDim], scaleMyVarBy)
        scaleVarDims = torch.tensor(scaleVarDims, dtype=torch.float32)
        n_var = torch.mul(n_var, scaleVarDims)
        # </editor-fold>

        dist = Normal(loc=n_mean, scale=n_var)

        z = dist.sample([1])
        return z

    def conditional_variance_avg(self, context):
        """Returns average variance of latent density givent context."""
        n_var = torch.abs(self.n_var_net(context))
        return(torch.mean(n_var, dim=0))


    def sample_with_log_prob(self, context):
        assert False, "Not Implemented"


    def mean(self, context):
        assert False, "Not Implemented"
