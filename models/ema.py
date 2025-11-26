import math
from copy import deepcopy
import torch
import torch.nn as nn


class EmaModel(nn.Module):
    
    def __init__(self, cfg, model, decay=0.9998, updates=0):
        """
        Args:
            model (nn.Module): model to apply EMA.
            decay (float): ema decay reate.
            updates (int): counter of EMA updates.
        """
        super(EmaModel, self).__init__()
        # Create EMA(FP16)
        self.cfg = cfg 
        self.model = deepcopy(model).eval()
        self.register_buffer("updates", torch.tensor(updates, dtype=torch.long))
        self._decay_base = cfg.trainer.ema_decaybase

        for p in self.model.parameters():
            p.requires_grad_(False)

    def decay(self, x):
        return self._decay_base * (1 - torch.exp(-x / self.cfg.trainer.ema_tau))
    
    @torch.no_grad()
    def update(self, model_normal):
        self.updates += 1
        decay_value = self.decay(self.updates)

        state_dict = model_normal.state_dict()
        for k, v in self.model.state_dict().items():
            ## in dpp mode, the weight are start with module.
            if v.dtype.is_floating_point:
                v *= decay_value
                v += (1.0 - decay_value) * state_dict[k].detach()
