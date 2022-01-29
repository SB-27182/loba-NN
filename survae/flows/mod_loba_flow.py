import torch
from torch import nn
from collections.abc import Iterable
from survae.utils import context_size
from survae.distributions import Distribution, ConditionalDistribution
from survae.transforms import Transform, ConditionalTransform


class Loba_flow(ConditionalDistribution):
    """See https://github.com/SB-27182/loba_NN"""

    def __init__(self, base_dist, transforms, context_init=None):
        super(dOsciCond_flow, self).__init__()
        assert isinstance(base_dist, Distribution)
        if isinstance(transforms, Transform): transforms = [transforms]
        assert isinstance(transforms, Iterable)
        assert all(isinstance(transform, Transform) for transform in transforms)
        self.base_dist = base_dist
        self.transforms = nn.ModuleList(transforms)
        self.context_init = context_init
        self.lower_bound = any(transform.lower_bound for transform in transforms)


    def log_prob(self, x, context):
        if self.context_init: context = self.context_init(context)
        log_prob = torch.zeros(x.shape[0], device=x.device)
        for transform in self.transforms:
            if isinstance(transform, ConditionalTransform):
                x, ldj = transform(x, context)
            else:
                x, ldj = transform(x)
            log_prob += ldj
        if isinstance(self.base_dist, ConditionalDistribution):
            log_prob += self.base_dist.log_prob(x, context)
        else:
            log_prob += self.base_dist.log_prob(x)
        return log_prob


    def sample(self, context):
        if self.context_init: context = self.context_init(context)
        if isinstance(self.base_dist, ConditionalDistribution):
            z = self.base_dist.sample(context)
        else:
            z = self.base_dist.sample(context_size(context))
        for transform in reversed(self.transforms):
            if isinstance(transform, ConditionalTransform):
                z = transform.inverse(z, context)
            else:
                z = transform.inverse(z)
        return z




    def sample_cond_latent_nVars(self, context, dims, myDim=0, scaleMyVarBy=1.0, scaleOthrVarsBy=0.01,
                           myOsciVarDim=0, scaleMyOsciVarBy=1.0, scaleOthrOsciVarsBy=1.0,
                           stupidbatchsize=320):
        print("WARNING: YOU MUST UPDATE PARAM:<stupidbatchsize> INSIDE <_new_osciCond_flow.py>")
        if self.context_init: context = self.context_init(context)
        z = self.base_dist.latent_sample_nVars(context, dims,
                                         myVarDim=myDim, scaleMyVarBy=scaleMyVarBy, scaleOthrVarsBy=scaleOthrVarsBy,
                                         myOsciVarDim=myOsciVarDim, scaleMyOsciVarBy=scaleMyOsciVarBy, scaleOthrOsciVarsBy=scaleOthrOsciVarsBy)
        z = torch.squeeze(z)
        x = torch.clone(z)
        for transform in reversed(self.transforms):
            if isinstance(transform, ConditionalTransform):
                x = transform.inverse(x, context)
            else:
                x = transform.inverse(x)

        return x, z




    def latent_sample_1DimOscil(self, context, dims, myDim, scaleMyVarBy=1.00, scaleOthrVarsBy=1.00,
                           myOsciVarDim=0, scaleMyOsciVarBy=5.0, scaleOthrOsciVarsBy=1.0):

        if self.context_init: context = self.context_init(context)
        z = self.base_dist.latent_sample_1DimOscil(context, dims,
                                         myDim=myDim, scaleMyVarBy=scaleMyVarBy, scaleOthrVarsBy=scaleOthrVarsBy,
                                         myOsciVarDim=myOsciVarDim, scaleMyOsciVarBy=scaleMyOsciVarBy, scaleOthrOsciVarsBy=scaleOthrOsciVarsBy)
        z = torch.squeeze(z)
        x = torch.clone(z)
        for transform in reversed(self.transforms):
            if isinstance(transform, ConditionalTransform):
                x = transform.inverse(x, context)
            else:
                x = transform.inverse(x)

        return x, z

    def conditional_variance_avg(self, context):
        """Returns avg variance of the batch; tensor is size[dims]"""
        if self.context_init: context = self.context_init(context)
        return self.base_dist.conditional_variance_avg(context)


    def sample_with_log_prob(self, context):
        raise RuntimeError("ConditionalFlow does not support sample_with_log_prob, see ConditionalInverseFlow instead.")
