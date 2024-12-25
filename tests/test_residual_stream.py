import torch
import torch.nn as nn
import unittest
from functools import partial

import sys, os; sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../")
from models.residual_stream import ResidualStreamBlock

class TestResidualStreamBlock(unittest.TestCase):
    """
    Tests for ResidualStreamBlock.

    For now, only checks that output shape is correct.
    """

    def setUp(self):
        self.dim = 64
        self.batch_size = 8
        self.seq_len = 10
        self.x = torch.randn(self.batch_size, self.seq_len, self.dim, requires_grad=True)
        self.model_func = nn.Linear(self.dim, self.dim)

    def test_pre_norm(self):
        norm_config = {'norm_method': 'pre-norm', 'norm_type': 'layernorm'}
        block = ResidualStreamBlock(self.dim, norm_config)
        self._check_module(block, self.x)

    def test_post_norm(self):
        norm_config = {'norm_method': 'post-norm', 'norm_type': 'layernorm'}
        block = ResidualStreamBlock(self.dim, norm_config)
        self._check_module(block, self.x)

    def test_pre_post_norm(self):
        norm_config = {'norm_method': 'pre+post-norm', 'norm_type': 'layernorm'}
        block = ResidualStreamBlock(self.dim, norm_config)
        self._check_module(block, self.x)

    def test_hypersphere_interpolation(self):
        norm_config = {'norm_method': 'hypersphere-interpolation'}
        block = ResidualStreamBlock(self.dim, norm_config)
        self._check_module(block, self.x)

    def test_no_norm(self):
        norm_config = {'norm_method': 'none'}
        block = ResidualStreamBlock(self.dim, norm_config)
        self._check_module(block, self.x)

    def _check_module(self, model, x):

        # check that forward pass works
        output = model(x, self.model_func)

        # check that output is Tensor and shape is correct
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, x.shape)

        # check that backward pass works
        output.sum().backward()
        self.assertIsNotNone(self.x.grad)

        # grad check (check that gradients are correct through numerical approximation)
        self._grad_check(model, self.model_func, x)

    def _grad_check(self, model, model_func, x):

        # clone all models and inputs to double precision (so that numerical approximation is more accurate)
        x = x.double()
        model = model.double()
        model_func = model_func.double()

        # function to check gradients of
        func = partial(model, model_func=model_func)

        # check that gradients computed through backward pass are correct through numerical approximation
        self.assertTrue(torch.autograd.gradcheck(func, x, fast_mode=True))

        # check gradients of gradients
        self.assertTrue(torch.autograd.gradgradcheck(func, x, fast_mode=True))

if __name__ == '__main__':
    unittest.main()
