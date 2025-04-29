'''
MIT License

Copyright (c) 2022 Jose A. Garrido Torres

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

'''
This file is adapted from the EDBO+ software (https://github.com/doyle-lab-ucla/edboplus/tree/main)
'''

import torch
import gpytorch
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.priors import GammaPrior
from gpytorch.constraints import GreaterThan
import numpy as np

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cpu"),
}

def build_and_optimize_model(train_x, train_y):
    """ Builds model and optimizes it."""

    gp_options = {
        'ls_prior1': 2.0, 'ls_prior2': 0.2, 'ls_prior3': 5.0,
        'out_prior1': 5.0, 'out_prior2': 0.5, 'out_prior3': 8.0,
        'noise_prior1': 1.5, 'noise_prior2': 0.1, 'noise_prior3': 5.0,
        'noise_constraint': 1e-5,
    }

    n_features = np.shape(train_x)[1]

    class ExactGPModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(ExactGPModel, self).__init__(train_x, train_y,
                                               likelihood)
            self.mean_module = gpytorch.means.ConstantMean()

            kernels = MaternKernel(
                ard_num_dims=n_features,
                lengthscale_prior=GammaPrior(gp_options['ls_prior1'],
                                             gp_options['ls_prior2'])
            )

            self.covar_module = ScaleKernel(
                kernels,
                outputscale_prior=GammaPrior(gp_options['out_prior1'],
                                             gp_options['out_prior2']))
            try:
                ls_init = gp_options['ls_prior3']
                self.covar_module.base_kernel.lengthscale = ls_init
            except:
                uniform = gp_options['ls_prior3']
                ls_init = torch.ones(n_features).to(**tkwargs) * uniform
                self.covar_module.base_kernel.lengthscale = ls_init

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    # initialize likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        GammaPrior(gp_options['noise_prior1'], gp_options['noise_prior2'])
    )

    likelihood.noise = gp_options['noise_prior3']
    model = ExactGPModel(train_x, train_y, likelihood).to(**tkwargs)

    model.likelihood.noise_covar.register_constraint(
        "raw_noise", GreaterThan(gp_options['noise_constraint'])
    )

    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
    ], lr=0.1)

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    training_iter = 1000
    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, train_y.squeeze(-1).to(**tkwargs))
        loss.backward()
        optimizer.step()

    model.eval()
    likelihood.eval()
    return model, likelihood  # Optimized model