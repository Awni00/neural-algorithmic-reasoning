import torch
import torch.nn as nn
import unittest
from functools import partial

import sys, os; sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../")
from models.residual_stream import ResidualStreamBlock, HypersphereLERP, HypersphereSLERP, AdaptiveHypersphereSLERP

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

class TestHypersphereInterpolation(unittest.TestCase):
    """
    Tests for HypersphereLERP, HypersphereSLERP, and AdaptiveHypersphereSLERP.

    Checks that output shape is correct, backward pass works, gradients are numerically correct, and output is unit-norm.
    """

    def setUp(self):
        self.dim = 64
        self.batch_size = 8
        self.seq_len = 10
        self.x = torch.randn(self.batch_size, self.seq_len, self.dim, requires_grad=True)
        self.y = torch.randn(self.batch_size, self.seq_len, self.dim, requires_grad=True)

    def test_hypersphere_lerp(self):
        for lerp_weight_constraint in ['none', 'sigmoid', 'abs', 'clamp']:
            with self.subTest(lerp_weight_constraint=lerp_weight_constraint):
                model = HypersphereLERP(self.dim, lerp_weight_constraint=lerp_weight_constraint)
                self._check_module(model, self.x, self.y)

    def test_hypersphere_slerp(self):
        for single_weight in [True, False]:
            with self.subTest(single_weight=single_weight):
                model = HypersphereSLERP(self.dim, single_weight=single_weight)
                self._check_module(model, self.x, self.y)

    def test_adaptive_hypersphere_slerp(self):
        for single_weight in [True, False]:
            with self.subTest(single_weight=single_weight):
                model = AdaptiveHypersphereSLERP(self.dim, single_weight=single_weight)
                self._check_module(model, self.x, self.y)

    def _check_module(self, model, x, y):

        # check that forward pass works
        output = model(x, y)

        # check that output is Tensor and shape is correct
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, x.shape)

        # check that output is unit-norm
        self.assertTrue(torch.allclose(output.norm(p=2, dim=-1), torch.ones_like(output.norm(p=2, dim=-1))))

        # check that backward pass works
        output.sum().backward()
        self.assertIsNotNone(self.x.grad)
        self.assertIsNotNone(self.y.grad)

        # grad check (check that gradients are correct through numerical approximation)
        self._grad_check(model, x, y)

    def _grad_check(self, model, x, y):

        # clone all models and inputs to double precision (so that numerical approximation is more accurate)
        x = x.double()
        y = y.double()
        model = model.double()

        # function to check gradients of
        func = partial(model)

        # check that gradients computed through backward pass are correct through numerical approximation
        self.assertTrue(torch.autograd.gradcheck(func, (x, y), fast_mode=True))

        # check gradients of gradients
        self.assertTrue(torch.autograd.gradgradcheck(func, (x, y), fast_mode=True))

if __name__ == '__main__':
    unittest.main()
