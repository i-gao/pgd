"""
Series of attack implementations.
Borrows heavily from https://github.com/MadryLab/robustness
"""

import torch
from utils import IMAGENET_MEAN, IMAGENET_STD, normalize_images, unnormalize_images

class Attack:
    def __init__(self, orig_input, eps, step_size):
        self.orig_input = orig_input
        self.eps = eps
        self.step_size = step_size

class LinfStep(Attack):
    """
    Credit: https://github.com/MadryLab/robustness/blob/a9541241defd9972e9334bfcdb804f6aefe24dc7/robustness/attack_steps.py
    Attack step for :math:`\ell_\infty` threat model. Given :math:`x_0`
    and :math:`\epsilon`, the constraint set is given by:

    .. math:: S = \{x | \|x - x_0\|_\infty \leq \epsilon\}
    """

    def project(self, x):
        """ """
        diff = x - self.orig_input
        diff = torch.clamp(diff, -self.eps, self.eps)
        out = torch.clamp(diff + self.orig_input, 0, 1)
        return out

    def step(self, x, g):
        """ """
        step = torch.sign(g) * self.step_size
        return x + step

    def init_random_x(self, x):
        """ """
        new_x = x + 2 * (torch.rand_like(x) - 0.5) * self.eps
        return torch.clamp(new_x, 0, 1)
    
    def __str__(self):
        return f"LInfAttack"


class L2Step(Attack):
    """
    Credit: https://github.com/MadryLab/robustness/blob/a9541241defd9972e9334bfcdb804f6aefe24dc7/robustness/attack_steps.py
    Attack step for :math:`\ell_2` threat model. Given :math:`x_0`
    and :math:`\epsilon`, the constraint set is given by:

    .. math:: S = \{x | \|x - x_0\|_2 \leq \epsilon\}
    """

    def project(self, x):
        """ """
        diff = x - self.orig_input
        diff = diff.view(-1, *diff.shape[3:])
        diff = diff.renorm(p=2, dim=0, maxnorm=self.eps)
        diff = diff.view(*x.shape)
        out = torch.clamp(self.orig_input + diff, 0, 1)
        return out

    def init_random_x(self, x):
        """ """
        l = len(x.shape) - 1
        rp = torch.randn_like(x)
        rp_norm = rp.view(rp.shape[0], -1).norm(dim=1).view(-1, *([1] * l))
        return torch.clamp(x + self.eps * rp / (rp_norm + 1e-10), 0, 1)

    def step(self, x, g):
        """ """
        l = len(x.shape) - 1
        g_norm = torch.norm(g.view(g.shape[0], -1), dim=1).view(-1, *([1] * l))
        scaled_g = g / (g_norm + 1e-10)
        return x + scaled_g * self.step_size

    def __str__(self):
        return f"L2Attack"

class UnconstrainedStep(Attack):
    """
    Credit: https://github.com/MadryLab/robustness/blob/a9541241defd9972e9334bfcdb804f6aefe24dc7/robustness/attack_steps.py
    Unconstrained threat model, :math:`S = [0, 1]^n`.
    """

    def project(self, x):
        """ """
        out = torch.clamp(x, 0, 1)
        return out

    def step(self, x, g):
        """ """
        return x + g * self.step_size

    def init_random_x(self, x):
        """ """
        new_x = x + (torch.rand_like(x) - 0.5).renorm(
            p=2, dim=0, maxnorm=self.step_size
        )
        return torch.clamp(new_x, 0, 1)

    def __str__(self):
        return f"UnconstrainedAttack"
    
class GradientStep(Attack):
    """
    Absolutely no constraints
    """
    def project(self, x):
        """ """
        return x

    def step(self, x, g):
        """ """
        return x + g * self.step_size

    def init_random_x(self, x):
        """ """
        return x

    def __str__(self):
        return f"GradientDescent"
