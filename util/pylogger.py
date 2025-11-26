import functools
import logging
import torch


def rank_zero_only(fn):
    
    @functools.wraps(fn)
    def wrapped_fn(*args, **kwargs):
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            return fn(*args, **kwargs)
    return wrapped_fn


def get_pylogger(name=__name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""
    
    logger = logging.getLogger(name)
    logging_levels = ("debug", "info", "warning", "error", "exception", "fatal", "critical")
 
    if not hasattr(logger, "_is_wrapped_by_rank_zero"):
        logging_levels = ("debug", "info", "warning", "error", "exception", "fatal", "critical")
        for level in logging_levels:
            original_fn = getattr(logger, level)
            wrapped_fn = rank_zero_only(original_fn)
            setattr(logger, level, wrapped_fn)
        logger._is_wrapped_by_rank_zero = True  

    return logger