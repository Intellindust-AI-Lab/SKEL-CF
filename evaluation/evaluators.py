import torch
from collections import defaultdict

from .metrics import *


class EvaluatorBase():
    ''' To use this class, you should inherit it and implement the `eval` method. '''
    def __init__(self):
        self.accumulator = defaultdict(list)

    def eval(self, **kwargs):
        ''' Evaluate the metrics on the data. '''
        raise NotImplementedError

    def get_results(self, chosen_metric=None):
        ''' Get the current mean results. '''
        # Only chosen metrics will be compacted and returned.
        compacted = self._compact_accumulator(chosen_metric)
        ret = {}
        for k, v in compacted.items():
            ret[k] = v.mean(dim=0).item()
        return ret

    def _compact_accumulator(self, chosen_metric=None):
        ''' Compact the accumulator list and return the compacted results. '''
        ret = {}
        for k, v in self.accumulator.items():
            # Only chosen metrics will be compacted.
            if chosen_metric is None or k in chosen_metric:
                ret[k] = torch.cat(v, dim=0)
                self.accumulator[k] = [ret[k]]
        return ret

