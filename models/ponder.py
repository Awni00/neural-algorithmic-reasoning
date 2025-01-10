import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math

from .transformer_blocks import create_norm

class PonderNetworkWrapper(nn.Module):

    def __init__(self,
            block,
            n_classes,
            halt_sequencewise=False,
            ponder_kl_div_loss_weight = 0.01,
            ponder_prior_lambda = 0.2,
            ponder_epsilon = 0.05,
            halt_prob_normalization = 'condition_on_halting' # condition_on_halting, halt_last
            ):
        """A wrapper that creates a recurrent network with a (pondering) adaptive computation mechanism from a given recurrent block.

        At each step, the network applies the block to the current hidden states, and predicts a halting probability for each element in the sequence.

        Halting is performed probabilistically based on the predicted halted probabilities. During training, logits are predicted at each step and the loss is the expected cross-entropy loss over all steps, weighted by the halting probabilities.

        Parameters
        ----------
        block : nn.Module
            The module to be used as the recurrent block. Maps [B, T, D] -> [B, T, D]. E.g., a Transformer block. Can be causal or not.
        n_classes : int
            Number of output classes.
        halt_sequencewise : bool, optional
            Whether to halt the entire sequence based on average-pooled states. Default is False.
        ponder_kl_div_loss_weight : float, optional
            weight for the KL(Geometric(lambda) prior || halting_probs) regularization loss. Added to cross-entropy loss. Default is 0.01.
        ponder_prior_lambda : float, optional
            lambda parameter of Geometric prior. Used in computed KL regularization. Corresponds to an average number of steps given by 1 / lambda. Default is 0.2.
        ponder_epsilon : float, optional
            Used to compute train_max_steps. train_max_steps, the maximum number steps trained for, is computed as integer such that the geometric prior halts before train_max_steps steps with probability at least 1 - ponder_epsilon. Default is 0.05.
        halt_prob_normalization : str, optional
            How to normalize halting probabilities. Since model is only trained up to train_max_steps, determined by ponder_epsilon and ponder_prior_lambda,
            halting probabilities need to be normalized so that they form a probability distribution that sums to 1. Otherwise, model may choose to halt after train_max_steps
            in order to avoid loss. Options are 'condition_on_halting' and 'halt_last'.
            - condition_on_halting: divides halting probabilities by their sum. This corresponds to conditioning on the event that the model halts in train_max_steps.
            - halt_last: sets the last halting probability to 1 - sum(halting_probs[:-1]), enforcing that the model halts in the last step if not halted already.
            Default is 'condition_on_halting'.
        """

        super().__init__()

        self.block = block # A transformer block mapping [B, T, D] -> [B, T, D]

        # PonderBlockWrapper outputs halting probabilities in addition to updated hidden states
        self.ponder_block = PonderBlockWrapper(block, halt_sequencewise=halt_sequencewise)
        self.n_classes = n_classes # number of classes for final prediction head
        self.dim = block.d_model # hidden dimension of the transformer block
        self.halt_sequencewise = halt_sequencewise
        self.halt_prob_normalization = halt_prob_normalization

        ## calculate max pondering steps based on ponder_epsilon
        # calculated as the number of steps needed to reach ponder_epsilon probability of halting under a geometric distribution prior with parameter ponder_prior_lambda
        self.train_max_steps = calc_geometric_percentile(ponder_prior_lambda, ponder_epsilon)
        self.geometric_prior = calc_geometric(torch.full((self.train_max_steps,), ponder_prior_lambda)) # shape: [train_max_steps]
        print(f'self.train_max_steps: {self.train_max_steps}, capturing {self.geometric_prior.sum():.2f} probability mass in un-normalized geometric prior') # should capture > 1 - ponder_epsilon probability mass

        # normalize so that prior is proper probability distribution that sums to 1
        self.geometric_prior = self.normalize_halt_probs(self.geometric_prior, dim = 0) # shape: [train_max_steps]

        self.ponder_prior_lambda = ponder_prior_lambda
        self.ponder_kl_div_loss_weight = ponder_kl_div_loss_weight

        ## final prediction head
        # add normalization before predicting halting logits, if not post-norm
        if block.norm_config['norm_method'] == 'pre-norm':
            self.prelogits_norm = create_norm(self.dim, block.norm_config['norm_type'])
        else:
            self.prelogits_norm = torch.nn.Identity()

        self.to_logits = nn.Sequential(
            self.prelogits_norm,
            nn.Linear(self.dim, self.n_classes), # [B, T, D] -> [B, T, n_classes]
            )

    def normalize_halt_probs(self, halting_probs, dim = 1):
        if self.halt_prob_normalization == 'condition_on_halting':
            # divide by sum of halting probabilities: p_i = p_i / sum(p_1, ..., p_n)
            halting_probs = halting_probs / halting_probs.sum(dim = dim, keepdim = True)
        elif self.halt_prob_normalization == 'halt_last':

            # set last halting probability to 1 - sum(halting_probs[:-1])
            slices = [slice(None)] * halting_probs.dim()  # Create a list of slices for all dimensions
            slices[dim] = slice(None, -1) # for dim dimension, corresponding to pondering steps, select all but the last step

            # get halting probabilities for all but the last step
            halting_probs_except_last = halting_probs[tuple(slices)] # shape: [B, train_max_steps - 1, T, D]

            # compute halting probability for last step as 1 - sum(halting_probs[..., :-1, ...]) = 1 - sum(halting_probs_except_last)
            # this corresponds to all remaining probability mass being assigned to the last step
            last_prob = 1 - halting_probs_except_last.sum(dim=dim, keepdim=True) # shape: [B, 1, T, D]

            # reassemble halting probabilities
            halting_probs = torch.cat([halting_probs_except_last, last_prob], dim=dim) # shape: [B, train_max_steps, T, D]

        else:
            raise ValueError(f"Invalid halt_prob_normalization: {self.halt_prob_normalization}")
        return halting_probs


    def forward(self, x, labels=None, **kwargs):

        if self.training:
            assert labels is not None, "Labels must be provided during training."
            return self.forward_train(x, labels, **kwargs)
        else:
            return self.forward_inference(x, **kwargs)


    def forward_train(self, x, labels, **kwargs):

        hidden_states = []
        halting_logits = []

        # run network for train_max_steps pondering steps, predicting halting probabilities at each step
        for _ in range(self.train_max_steps):
            x, halt_logits = self.ponder_block(x, **kwargs)

            hidden_states.append(x)
            halting_logits.append(halt_logits)

        # pack halting probs and hidden states across pondering steps
        halting_logits = torch.stack(halting_logits, dim = 1) # shape: [B, train_max_steps, T]
        hidden_states = torch.stack(hidden_states, dim = 1) # shape: [B, train_max_steps, T, D]

        # compute halting probabilities and normalize
        halting_conditional_probs = halting_logits.sigmoid() # shape: [B, train_max_steps, T] (or [B, train_max_steps] if halt_sequencewise=True)
        halting_probs = calc_geometric(halting_conditional_probs, dim = 1) # shape: [B, train_max_steps, T] (or [B, train_max_steps] if halt_sequencewise=True)
        halting_probs = self.normalize_halt_probs(halting_probs, dim = 1) # shape: [B, train_max_steps, T] (or [B, train_max_steps] if halt_sequencewise=True)
        if self.halt_sequencewise:
            halting_probs = halting_probs.unsqueeze(-1) # shape: [B, train_max_steps, 1]

        # predict logits from hidden states at each step
        logits = self.to_logits(hidden_states) # shape: [B, train_max_steps, T, n_classes]

        # calculate KL divergence with geometric prior as regularization
        kl_div_loss = self.compute_kl_div_loss(halting_probs) # shape: [] (scalar)

        # calculate cross-entropy loss weighted by halting probabilities
        halt_weighted_ce_loss, ce_loss = self.compute_weighted_ce_loss(logits, labels, halting_probs) # shape: [] (scalar)

        # total loss = cross-entropy task loss + KL regularization
        loss = halt_weighted_ce_loss + self.ponder_kl_div_loss_weight * kl_div_loss

        intermediate_outputs = dict(
            hiddens = hidden_states,
            logits = logits,
            halting_conditional_probs = halting_conditional_probs,
            halting_probs = halting_probs,
            ce_loss = ce_loss,
            halt_weighted_ce_loss = halt_weighted_ce_loss,
            kl_div_loss = kl_div_loss,
            loss = loss,
        )

        return loss, intermediate_outputs

    def compute_weighted_ce_loss(self, logits, labels, halting_probs):
        """
        Computes the cross-entropy loss weighted by halting probabilities.

        \sum_{t=1}^{max_steps} p_t * CE(y, y_hat(t)), where p_t is the halting probability at step t, y is the true label, and y_hat(t) is the predicted logits at step t.

        Parameters
        ----------
        logits : torch.Tensor
            Predicted logits of shape [B, train_max_steps, T, n_classes].
        labels : torch.Tensor
            True labels of shape [B, T].
        halting_probs : torch.Tensor
            Halting probabilities of shape [B, train_max_steps, T].
        """
        # calculate cross entropy task loss

        B, L, T, C = logits.size()

        labels = repeat(labels, 'b n -> b l n', l = self.train_max_steps) # shape: [B, T] -> [B, train_max_steps, T]

        # compute cross-entropy loss without reduction
        # need more complex reduction scheme (first expected loss across pondering steps, weighted by halting_probs, then average over batch and sequence length)
        ce_loss = torch.nn.functional.cross_entropy(logits.view(-1, C), labels.contiguous().view(-1), reduction='none').view(B, L, T) # shape: [B, train_max_steps, T]
        # logits.view(-1, self.n_classes): [B * train_max_steps * T, n_classes]; labels.contiguous().view(-1): [B * train_max_steps * T,]

        # compute loss weighted by halting probabilities
        halt_weighted_ce_loss = (ce_loss * halting_probs).sum(dim=1) # shape: [B, T]

        # average across batch and sequence length
        halt_weighted_ce_loss = halt_weighted_ce_loss.mean() # shape: [] (scalar)

        return halt_weighted_ce_loss, ce_loss

    def compute_kl_div_loss(self, halting_probs):
        """
        Computes the KL divergence between the halting probabilities and the geometric prior.

        Parameters
        ----------
        halting_probs : torch.Tensor
            Halting probabilities of shape [B, train_max_steps, T].
        """
        # calculate KL divergence with geometric prior as regularization

        geometric_prior_dist = self.geometric_prior # shape: [train_max_steps]

        # if halting token-wise (e.g., a causal model), geoemtric prior on halting probabilities is applied to each token, i.e., repeated across sequence length
        if not self.halt_sequencewise:
            T = halting_probs.size(2)
            geometric_prior_dist = repeat(geometric_prior_dist, 'l -> (l n)', n = T) # shape: [train_max_steps * T]
            halting_probs_flattened = rearrange(halting_probs, '... l n -> ... (l n)') # shape: [B, train_max_steps, T] -> [B, train_max_steps * T]
        # halting sequence-wise
        else:
            halting_probs_flattened = halting_probs # shape: [B, train_max_steps]

        # NOTE: consider passing halting_probs_flattened in log space to avoid numerical instability
        # "It is recommended to pass certain distributions (like softmax) in the log space to avoid numerical issues caused by explicit log."
        # (although not sure how re-normalization would work)
        kl_div_loss = torch.nn.functional.kl_div(
            torch.log(geometric_prior_dist), # log probabilities of geometric prior
            halting_probs_flattened, # halting probabilities (not log since log_target=False)
            reduction='batchmean',
            log_target=False
        ) # shape: [] (scalar)

        return kl_div_loss


    def forward_inference(self, x, **kwargs):
        """
        Run the network in inference mode, halting stochastically based on predicted halting probabilities.

        In case of token-wise halting, each token can halt independently based on its own halting probability.
        In this case, each time a token halts, it "finalizes" its own prediction logit,
        but its state continues to be updated in order for other tokens to attend to the updated version of the halted token.
        In case of sequence-wise halting, a halting probability for the entire sequence is predicted using average-pooled states.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [B, T, D].

        Returns
        -------
        logits : torch.Tensor
            Predicted logits of shape [B, T, n_classes].
        """

        steps_hidden_states = []
        steps_halt_logits = []
        steps_halt = []

        for step in range(self.train_max_steps):

            x, halt_logits = self.ponder_block(x)
            # x.shape: [B, T, D]
            # halt_logits.shape: [B, T]

            steps_hidden_states.append(x)
            steps_halt_logits.append(halt_logits)

            # calculating halting probs
            halt_probs = halt_logits.sigmoid() # shape: [B, T], or [B] if halt_sequencewise=True

            # flip a coin for each sequence in batch (and each token if token-wise halting) to determine whether to halt at this step
            # randomly halt w.p. halting_probs
            halt_now = torch.rand_like(halt_probs) <= halt_probs # shape: [B, T], or [B] if halt_sequencewise=True

            # stack the halting signal across layers and determine whether to stop early
            steps_halt.append(halt_now)

            ## break if halting has been sampled for all layers
            # check if halted at any step, for each sequence in batch (and each token if token-wise halting)
            step_was_halted = torch.any(torch.stack(steps_halt), dim = 0) # shape: [B, T]

            # stop if all batches and tokens have halted
            if torch.all(step_was_halted):
                break


        # stack hidden states and boolean halting signals across pondering steps
        steps_hidden_states = torch.stack(steps_hidden_states, dim = 1) # shape: [B, n_steps, T, D]
        steps_halt = torch.stack(steps_halt, dim = 1) # shape: [B, n_steps, T]

        # calculate the index of the first halt signal, and make it the last layer if none of them halted
        halt_times = get_first_halt(steps_halt, dim = 1) # shape: [B, T]

        halt_time_hidden_states = gather_halt_time_hidden_states(steps_hidden_states, halt_times) # shape: [B, T, D]

        # calculate logits
        logits = self.to_logits(halt_time_hidden_states) # shape: [B, T, n_classes]

        outputs = dict(
            steps_hidden_states = steps_hidden_states, # hidden states at each step; shape: [B, n_steps, T, D]
            steps_halt_logits = steps_halt_logits, # halting logits at each step; shape: [B, n_steps, T]
            steps_halt = steps_halt, # halting signals at each step; shape: [B, n_steps, T]
            halt_times = halt_times, # index of the first halt signal; shape: [B, T]
            halt_time_hidden_states = halt_time_hidden_states, # hidden states at the time of halting; shape: [B, T, D]
            logits = logits # predicted logits; shape: [B, T, n_classes]
           )

        return logits, outputs

class PonderBlockWrapper(nn.Module):

    def __init__(self, block, halt_sequencewise=False):
        """
        A module wrapper that adds a pondering mechanism to a given block.

        The output of the resulting pondering-block is the processed hidden states produced by `block` as well as the halting logits.

        Parameters
        ----------
        block : nn.Module
            The block to be wrapped.
        halt_sequencewise : bool, optional
            Whether to average-pool sequence and produce a single halting probability for the entire sequence.
            When False, a separate halting probability is produced for each element in the sequence.
            E.g., in a causal model, the halting probability for each token is a function of the past.
            Must be False if using a causal model. By default False.
        """


        super().__init__()
        self.block = block
        self.dim = block.d_model
        assert not (block.causal and halt_sequencewise), "Must use halt_sequencewise=False when using a causal model"
        self.halt_sequencewise = halt_sequencewise

        # add normalization before predicting halting logits, if not post-norm
        if block.norm_config['norm_method'] == 'pre-norm':
            self.prehaltlogits_norm = create_norm(self.dim, block.norm_config['norm_type'])
        else:
            self.prehaltlogits_norm = torch.nn.Identity()

        self.to_halt_logits = nn.Sequential(
            self.prehaltlogits_norm,
            nn.Linear(self.dim, 1), # [B, T, D] -> [B, T, 1] (if halt_sequencewise=True, [B, D] -> [B, 1])
            Rearrange('... () -> ...') # [B, T, 1] -> [B, T] (if halt_sequencewise=True, [B, 1] -> [B])
            )

    def forward(self, x, **kwargs):
        # x: [B, T, D]

        y = self.block(x, **kwargs) # shape: [B, T, D]

        if self.halt_sequencewise:
            halt_input = y.mean(dim=1) # [B, T, D] -> [B, D]
        else:
            halt_input = y # shape: [B, T, D]

        halting_logits = self.to_halt_logits(halt_input) # shape: [B, T] (or if halt_sequencewise=True, [B])

        return y, halting_logits

# -------------------------

# Helper functions for pondering mechanism

def get_first_halt(halt_signals, dim=1):
    """
    Given a tensor of halt signals, returns the index of the first halt signal in each sequence.

    Parameters
    ----------
    halt_signals : torch.Tensor
        Boolean tensor of shape [B, T] indicating whether to halt at each step.

    Returns
    -------
    first_halt : torch.Tensor
        Index of the first halt signal in each sequence, or -1 if no halt signal.
    """

    first_halt = (halt_signals.cumsum(dim = dim) == 0).sum(dim = 1).clamp(max = halt_signals.size(1) - 1)
    return first_halt


def gather_halt_time_hidden_states(hiddens, halt_times):
    """
    Given a tensor of hidden states and the index of the first halt signal in each sequence, selects the hidden states at the time of halting.

    Parameters
    ----------
    hiddens : torch.Tensor
        Hidden states of shape [B, N, T, D].
    halt_times : torch.Tensor
        Index of the first halt signal. Shape either [B, T] or [B], depending on whether sequence-wise or token-wise halting is used. values between 0 and N-1.

    Returns
    -------
    halt_time_hidden_states : torch.Tensor
        Hidden states at the time of halting, of shape [B, T, D].
    """

    B, N, T, D = hiddens.size()

    if halt_times.dim() == 1:
        # create indices for batch and time (seqlen) dimensions
        batch_indices = torch.arange(B) # shape: [B]

        # select the hidden states at the time of halting
        halt_time_hidden_states = hiddens[batch_indices, halt_times, :]

    elif halt_times.dim() == 2:
        # create indices for batch and time (seqlen) dimensions
        batch_indices = torch.arange(B).unsqueeze(1) # shape: [B, 1]
        time_indices = torch.arange(T).unsqueeze(0) # shape: [1, T]

        # expand batch and time indices to match the shape of halt_times
        batch_indices = batch_indices.expand(B, T)  # Shape: [B, T]
        time_indices = time_indices.expand(B, T)  # Shape: [B, T]

        # select the hidden states at the time of halting
        halt_time_hidden_states = hiddens[batch_indices, halt_times, time_indices, :]

    return halt_time_hidden_states

# Quote from PonderNet paper, regarding sequence-wise vs toekn-wise halting:
# "ACT considers the case of sequential data, where the step function can ponder dynamically for each new item in the input sequence.
# Given the introduction of attention mechanisms in the recent years (e.g. Transformers; Vaswani et al., 2017) that can process arrays with dynamic shapes,
# we suggest that pondering should be done holistically instead of independently for each item in the sequence. This can be useful in learning e.g. how
# many message-passing steps to do in a graph network (Velickovic et al., 2019)."
# This seems to imply that they always use sequence-wise halting in their Transformer-based experiments. They don't share code though,
# so the exact implementation they used is unclear.
# This is more clear from the description in the appendix on experimental details, esp. when describing experiments with encoder-decoder models:
# "The number of layers in the encoder and the decoder was learnt, but constrained to be the same. This number was identified as the “pondering time” in our PonderNet architecture...
# he prediction was computed by applying the decoder layer an equal number of times to the pondering step, that is y_{n+1} = decoder(...(decoder(h_{n+1}))."
# Here, we implement a way to halt either sequence-wise or token-wise. For token-wise halting, each token can halt independently based on its own halting probability, which is a function of its own state.
# E.g., for an autoregressive model, the last token choses how many "decoder" steps to take before halting.

def pad_to(t, padding, dim = -1, value = 0.):
    if dim > 0:
        dim = dim - t.ndim
    zeroes = -dim - 1
    return torch.nn.functional.pad(t, (*((0, 0) * zeroes), *padding), value = value)

def safe_cumprod(t, eps = 1e-10, dim = -1):
    t = torch.clip(t, min = eps, max = 1.)
    return torch.exp(torch.cumsum(torch.log(t), dim = dim))

def exclusive_cumprod(t, dim = -1):
    cum_prod = safe_cumprod(t, dim = dim)
    return pad_to(cum_prod, (1, -1), value = 1., dim = dim)

def calc_geometric(conditional_halting_probs, dim = -1):
    """
    Given tensor where conditional_halting_probs[..., t, ...] is the probability of halting at step t given that halting has not yet occured,
    calculates the probability of at step l.

    Result is a tensor where result[..., t, ...] is the probability of halting at step t.

    This is calculated as result[..., t, ...] prod_{s=1}^{t-1} (1 - conditional_halting_probs[..., s, ...]) * conditional_halting_probs[..., t, ...]
    """

    return exclusive_cumprod(1 - conditional_halting_probs, dim = dim) * conditional_halting_probs

def calc_geometric_percentile(lamda, epsilon):
    """
    calculates minimal t such that P(Geometric(lamda) <= t) >= 1 - epsilon
    <=> 1 - (1 - lamda)^t >= 1 - epsilon => t = ceil(log(epsilon) / log(1 - lamda))

    Used to set the maximum number of steps to take in the pondering mechanism.
    """

    return math.ceil(math.log(epsilon) / math.log(1 - lamda))
