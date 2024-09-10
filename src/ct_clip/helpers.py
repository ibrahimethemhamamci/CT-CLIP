import torch
import torch.nn.functional as F


def exists(val):
    return val is not None


def identity(t, *args, **kwargs):
    return t


def default(val, d):
    return val if exists(val) else d


def cycle(dl):
    while True:
        for data in dl:
            yield data


def noop(*args, **kwargs):
    pass


def cast_tuple(t):
    return t if isinstance(t, (tuple, list)) else (t,)


def pair(val):
    ret = (val, val) if not isinstance(val, tuple) else val
    assert len(ret) == 2
    return ret


def l2norm(t: torch.Tensor) -> torch.Tensor:
    return F.normalize(t, dim=-1)


def log(t: torch.Tensor, eps=1e-20) -> torch.Tensor:
    return torch.log(t + eps)
