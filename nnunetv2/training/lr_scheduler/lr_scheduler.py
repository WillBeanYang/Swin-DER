import math
import warnings
from typing import List

from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Optimizer

class PolyLRScheduler(_LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        initial_lr: float,
        max_steps: int,
        exponent: float = 0.9,
        current_step: int = None
    ) -> None:
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.max_steps = max_steps
        self.exponent = exponent
        self.ctr = 0
        super().__init__(optimizer, current_step if current_step is not None else -1, False)

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1

        new_lr = self.initial_lr * (1 - current_step / self.max_steps) ** self.exponent
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

class LinearWarmupCosineAnnealingLR(_LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        max_steps: int,
        initial_lr: float,
        warmup_start_lr: float = 0.0,
        eta_min: float = 0.0,
        current_step: int = None,
    ) -> None:
        """
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            warmup_epochs (int): Maximum number of iterations for linear warmup
            max_epochs (int): Maximum number of iterations
            warmup_start_lr (float): Learning rate to start the linear warmup. Default: 0.
            initial_lr (float): Initial Learning rate.
            eta_min (float): Minimum learning rate. Default: 0.
            last_epoch (int): The index of last epoch. Default: -1.
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.initial_lr = initial_lr
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        self.ctr = 0

        super().__init__(optimizer, current_step if current_step is not None else -1, False)

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1
        if current_step < self.warmup_steps:
            new_lr = self.warmup_start_lr + current_step * (self.initial_lr - self.warmup_start_lr) / (self.warmup_steps - 1)
        else:
            new_lr = self.eta_min + 0.5 * (self.initial_lr - self.eta_min) * (1 + math.cos(math.pi * (current_step - self.warmup_steps) / (self.max_steps - self.warmup_steps)))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

class CosineAnnealingLR(_LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        initial_lr: float,
        T_max: int,
        eta_min: float = 0.0,
        current_step: int = None,
    ) -> None:
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.T_max = T_max
        self.eta_min = eta_min
        self.ctr = 0
        super().__init__(optimizer, current_step if current_step is not None else -1, False)

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1

        new_lr = self.eta_min + (self.initial_lr - self.eta_min) * (1 + math.cos(math.pi * current_step / self.T_max)) / 2
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
