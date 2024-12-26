import torch
import torch.nn as nn

class AdaptiveTemperatureSoftmax(nn.Module):
    """
    Softmax with adaptive temperature scaling.

    At inference time, will adaptive scale the temperature of the softmax distribution based on the Shannon entropy.

    See: "softmax is not enough (for sharp out-of-distribution)" -- Velickovic et al (2024) -- https://arxiv.org/abs/2410.01104
    """

    def __init__(self, adaptive_training_temperature=False, dim=-1):
        super(AdaptiveTemperatureSoftmax, self).__init__()

        # this is a polynomial fit for estimating the optimal temperature as a function of the Shannon entropy
        # computed by fitting a degree 4 polynomial to estimate 1/temp = poly(entropy)
        self.poly_fit = [-0.037, 0.481, -2.3, 4.917, -1.791] # coefficients of the polynomial fit (first is a_4, last is a_0)

        # whether to adaptively scale the temperature during training as well (not just at inference time)
        # not sure if this makes sense
        self.adaptive_training_temperature = adaptive_training_temperature

        self.dim = dim # dimension along which to compute the softmax

    def forward(self, logits):

        # during training, will use the standard softmax, unless adaptive_training_temperature is set
        if self.training:
            if self.adaptive_training_temperature:
                return self.adaptive_temperature_softmax(logits)
            else:
                return torch.nn.functional.softmax(logits, dim=self.dim)

        # at inference time, will use the adaptive temperature softmax
        else:
            return self.adaptive_temperature_softmax(logits)

    def adaptive_temperature_softmax(self, logits):

        # compute softmax distribution of logits
        original_probs = torch.nn.functional.softmax(logits, dim=self.dim)
        # shape: [batch_size, ..., num_classes]

        # compute shannon entropy of softmax distribution
        entropy = torch.sum(-original_probs * torch.log(original_probs + 1e-9), dim=self.dim, keepdim=True)
        # shape: [batch_size, ..., 1]

        # beta = 1 / temperature = poly(entropy)
        beta_polynomial_fit = polyval_horner(self.poly_fit, entropy)

        # only correct temperature for high-entropy heads (> 0.5 entropy) and never increase entropy (beta >= 1)
        beta = torch.where(
            entropy > 0.5, # don't overcorrect low-entropy heads
            torch.maximum(beta_polynomial_fit, torch.ones_like(entropy)), # never increase entropy
            1.0
        )

        return torch.nn.functional.softmax(logits * beta, dim=self.dim)


def polyval(coefs, x):
    coefs = coefs[::-1] # [a_n, a_{n-1}, ..., a_0] -> [a_0, a_1, ..., a_n]
    val = 0
    for idx, coef in enumerate(coefs):
        val += coef * x**idx
    return val
    # return sum([coefs[len(coefs) - i] * x**i for i in range(len(coefs))])


def polyval_horner(coefs, x):
    # Horner's method for evaluating a polynomial (https://en.wikipedia.org/wiki/Horner%27s_method)
    # slightly faster than the naive method
    val = 0
    for coef in coefs:
        val = val * x + coef
    return val