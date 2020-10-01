import numpy as np
import torch

from interface.attack import AttackInterface


class FGSMAttack(AttackInterface):
    def __call__(self, epsilon, x, grad):
        perturbed = x + epsilon * grad.sign()
        perturbed = torch.clamp(perturbed, np.min(x.detach().numpy()), np.max(x.detach().numpy()))

        return perturbed
