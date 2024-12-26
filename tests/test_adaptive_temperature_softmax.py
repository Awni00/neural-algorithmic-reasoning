import unittest
import torch
from functools import partial

import sys, os; sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../")
from models.adaptive_temperature_softmax import AdaptiveTemperatureSoftmax

class TestAdaptiveTemperatureSoftmax(unittest.TestCase):
    """
    Tests for AdaptiveTemperatureSoftmax.

    Checks for shape, backward pass, and gradient computation.
    """

    def setUp(self):
        self.batch_size = 8
        self.seq_len = 10
        self.num_classes = 16
        self.logits = torch.randn(self.batch_size, self.seq_len, self.num_classes, requires_grad=True)
        self.model = AdaptiveTemperatureSoftmax()

    def test_forward_train(self):
        self.model.train()
        self._check_module(self.model, self.logits)

    def test_forward_eval(self):
        self.model.eval()
        self._check_module(self.model, self.logits)

    def _check_module(self, model, logits):

        # check that forward pass works
        output = model(logits)

        # check that output is Tensor and shape is correct
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, logits.shape)

        # check that backward pass works
        output.sum().backward()
        self.assertIsNotNone(self.logits.grad)

        # grad check (check that gradients are correct through numerical approximation)
        self._grad_check(model, logits)

    def _grad_check(self, model, logits):

        # clone all models and inputs to double precision (so that numerical approximation is more accurate)
        logits = logits.double()
        model = model.double()

        # function to check gradients of
        func = partial(model)

        # check that gradients computed through backward pass are correct through numerical approximation
        self.assertTrue(torch.autograd.gradcheck(func, logits, fast_mode=True))

        # check gradients of gradients
        self.assertTrue(torch.autograd.gradgradcheck(func, logits, fast_mode=True))


if __name__ == '__main__':
    unittest.main()