import functools
import operator

import sympy
import torch
import numpy


@functools.lru_cache(None)
def has_triton():
    try:
        import triton

        return triton is not None
    except (ImportError, ModuleNotFoundError):
        return False


@functools.lru_cache(None)
def has_torchvision_roi_align():
    try:
        from torchvision.ops import roi_align  # noqa

        return roi_align is not None and hasattr(
            getattr(torch.ops, "torchvision", None), "roi_align"
        )
    except (ImportError, ModuleNotFoundError):
        return False


def conditional_product(*args):
    return functools.reduce(operator.mul, [x for x in args if x])


def sympy_product(it):
    return functools.reduce(operator.mul, it, sympy.Integer(1))


def unique(it):
    return {id(x): x for x in it}.values()

def rankmin(x):
    # https://stackoverflow.com/questions/5284646/rank-items-in-an-array-using-python-numpy-without-sorting-array-twice
    u, inv, counts = numpy.unique(x, return_inverse=True, return_counts=True)
    csum = numpy.zeros_like(counts)
    csum[1:] = counts[:-1].cumsum()
    return csum[inv]