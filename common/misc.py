from typing import Type

import torch
import numpy as np
import gym

import torch.nn.functional as F

from torch.autograd import Variable

Tensor = Type[torch.Tensor]


def soft_update(target: torch.nn.Module, source: torch.nn.Module, tau: float = 1.0):
    """Perform DDPG soft update (move target params toward source based on weight factor tau).

    Reference:
        https://github.com/ikostrikov/pytorch-ddpg-naf/blob/master/ddpg.py#L11

    :param torch.nn.Module target: Net to copy parameters to
    :param torch.nn.Module source: Net whose parameters to copy
    :param float tau: Range form 0 to 1, weight factor for update
    """

    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def onehot_from_logits(logits, eps=0.0):
    """Given a batch of logits, return one-hot sample using epsilon greedy strategy (based on given epsilon). When eps=0., it performs as an argmax operator."""

    # get best (according to current policy) actions in one-hot form
    one_hots = torch.zeros_like(logits, device=logits.device)
    argmax_acs = torch.argmax(logits, dim=-1, keepdim=True)
    one_hots.scatter_(-1, argmax_acs, 1.0)
    if eps == 0.0:
        return one_hots
    # get random actions in one-hot form
    rand_acs = Variable(
        torch.eye(logits.shape[1], device=logits.device)[
            [np.random.choice(range(logits.shape[1]), size=logits.shape[0])]
        ],
        requires_grad=False,
    )
    # chooses between best and random actions using epsilon greedy
    return torch.stack(
        [
            argmax_acs[i] if r > eps else rand_acs[i]
            for i, r in enumerate(torch.rand(logits.shape[0]))
        ]
    )


def sample_gumbel(shape, eps=1e-20, tens_type=torch.FloatTensor, device="cpu"):
    """Sample from Gumbel(0, 1).

    Note:
        modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
    """

    # U = Variable(tens_type(*shape).uniform_(), requires_grad=False)
    U = torch.rand(shape, requires_grad=False, device=device)
    return -torch.log(-torch.log(U + eps) + eps).detach()


def gumbel_softmax_sample(logits, temperature, explore: bool):
    """Draw a sample from the Gumbel-Softmax distribution.

    Note:
        modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
    """

    y = logits
    if explore:
        y = y + sample_gumbel(
            logits.shape, tens_type=type(logits.data), device=logits.device
        )
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(
    logits: Tensor, temperature: float = 1.0, hard: bool = False, explore: bool = True
):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.

    Note:
        modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb

    :param DataTransferType logits: Unnormalized log-probs.
    :param float temperature: Non-negative scalar.
    :param bool hard: If ture take argmax, but differentiate w.r.t. soft sample y
    :returns [batch_size, n_class] sample from the Gumbel-Softmax distribution. If hard=True, then the returned sample
        will be one-hot, otherwise it will be a probability distribution that sums to 1 across classes
    """
    y = gumbel_softmax_sample(logits, temperature, explore)
    if hard:
        y_hard = onehot_from_logits(y)
        y = (y_hard - y).detach() + y
    return y


def masked_logits(
    logits: Tensor, mask: Tensor = None, explore: bool = False, normalize: bool = False
):
    if explore:
        logits += sample_gumbel(logits.shape, tens_type=type(logits.data))
    if mask is not None:
        if not isinstance(mask, torch.Tensor):
            mask = torch.FloatTensor(mask, device=logits.device)
        inf_mask = torch.maximum(torch.log(mask), torch.ones_like(logits) * -1e9)
        logits += inf_mask
    if normalize:
        logits = F.normalize(logits, p=2.0, dim=-1)
    return logits


def masked_gumbel_softmax(
    logits: Tensor, mask: Tensor, hard: bool = True, explore: bool = False
) -> Tensor:
    """Generate masked gumbel softmax without param noise.

    Args:
        logits (Tensor): Unnormalized log-probs.
        mask (Tensor): Mask tensor

    Returns:
        Tensor: Generate masked gumbel softmax (onehot)
    """
    logits = masked_logits(logits, mask, explore)
    y = F.softmax(logits, dim=-1)
    if hard:
        y_hard = onehot_from_logits(logits)
        y = (y_hard - y).detach() + y
    return y


def clip_action(
    logits: torch.Tensor,
    action_space: gym.Space,
    exploration: bool = False,
    action_mask: torch.Tensor = None,
    std: float = 0.1,
) -> torch.Tensor:
    """Clip raw logits with action space definition to legal actions.

    Args:
        logits (torch.Tensor): Logits need to be clipped.
        action_space (gym.Space): Action space.
        exploration (bool, optional): Enable exploration or not. Defaults to False.
        action_mask (torch.Tensor, optional): Action mask. Defaults to None.

    Returns:
        torch.Tensor: Action tensor.
    """

    eps = (
        0.0 if not exploration else torch.normal(mean=torch.zeros_like(logits), std=std)
    )
    logits += eps
    # XXX(ming): seems that there is no action mask for the continuous case
    if action_mask is not None:
        logits *= action_mask
    logits = torch.max(
        torch.min(logits, torch.as_tensor(action_space.high, dtype=logits.dtype)),
        torch.as_tensor(action_space.high, dtype=logits.dtype),
    )
    return logits


def cumulative_td_errors(
    start: int, end: int, offset: int, value, td_errors, ratios, gamma: float
):
    v = np.zeros_like(value)
    assert end - offset > start, (start, end, offset)
    for s in range(start, end - offset):
        pi_of_c = 1.0
        trace_errors = [td_errors[s]]
        for t in range(s + 1, s + offset):
            pi_of_c *= ratios[t - 1]
            trace_errors.append(gamma ** (t - start) * pi_of_c * td_errors[t])
        v[s] = value[s] + np.sum(trace_errors)
    return v


class OUNoise:
    """Ornstein-Uhlenbeck Noise.

    Reference:
        https://github.com/songrotek/DDPG/blob/master/ou_noise.py
    """

    def __init__(
        self,
        action_dimension: int,
        scale: float = 0.1,
        mu: float = 0.0,
        theta: float = 0.15,
        sigma: float = 0.2,
    ):
        """Initialize an OUNoise instance.

        Args:
            action_dimension (int): The action dimension.
            scale (float, optional): The factor used to scale the noise output. Defaults to 0.1.
            mu (float, optional): Expectness value. Defaults to 0.
            theta (float, optional): The parameter used to produce the difference from mu to x. Defaults to 0.15.
            sigma (float, optional): Variance scale. Defaults to 0.2.
        """

        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        """Reset the state which related to the action's dimension and expectness."""

        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self) -> float:
        """Compute the noise.

        Returns:
            float: Noise output.
        """

        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state * self.scale
