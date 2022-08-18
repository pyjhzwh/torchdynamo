import logging

import triton
from triton import Config
from triton import cdiv
from triton import heuristics
from triton import next_power_of_2
from triton.ops.matmul import get_configs_io_bound
from triton.ops.matmul_perf_model import early_config_prune as mm_early_config_prune

from torchinductor import config
from torchinductor.triton_ops.mm_perf_model import estimate_matmul_time
from torchinductor.utils import conditional_product

from .conv_perf_model import early_config_prune as conv_early_config_prune
from .conv_perf_model import estimate_conv_time

log = logging.getLogger(__name__)


class Autotuner(triton.code_gen.Autotuner):
    """
    Customized triton autotuner
    """

    def _bench(self, *args, config, **kwargs):
        try:
            return super()._bench(*args, config=config, **kwargs)
        except triton.code_gen.OutOfResources as e:
            log.warning("OutOfResources: %s %s", e, config)
            return (float("inf"), float("inf"), float("inf"))


def autotune(configs, key, prune_configs_by=None, reset_to_zero=None):
    """
    A copy of triton.autotune that calls our subclass above
    """

    def decorator(fn):
        def wrapper(kernel):
            return Autotuner(
                kernel, fn.arg_names, configs, key, reset_to_zero, prune_configs_by
            )

        fn.kernel_decorators.append(wrapper)
        return fn

    return decorator


def triton_config(size_hints, x, y=None, z=None, num_stages=1):
    """
    Construct a pointwise triton config with some adjustment heuristics
    based on size_hints. Size_hints is a tuple of numels in each tile
    dimension and will be rounded up to the nearest power of 2.
    """
    # Ideally we want to read this from some device config
    maxGridSize = [2147483647, 65535, 65535]

    target = conditional_product(x, y, z)
    if conditional_product(*size_hints) < target:
        target //= 8

    # shrink sizes to size hints
    x = min(x, size_hints[0])
    if y:
        y = min(y, size_hints[1])
    if z:
        z = min(z, size_hints[2])

    # if we are below original block size, scale up where we can;
    # or if the calculated grid size is larger than the limit, we bump up the corresponding dimension
    while x < size_hints[0] and (
        x * maxGridSize[0] < size_hints[0] or conditional_product(x, y, z) < target
    ):
        x *= 2
    while (
        y
        and y < size_hints[1]
        and (
            y * maxGridSize[1] < size_hints[1] or conditional_product(x, y, z) < target
        )
    ):
        y *= 2
    while (
        z
        and z < size_hints[2]
        and (
            z * maxGridSize[2] < size_hints[2] or conditional_product(x, y, z) < target
        )
    ):
        z *= 2

    cfg = {"XBLOCK": x}
    if y:
        cfg["YBLOCK"] = y
    if z:
        cfg["ZBLOCK"] = z
    num_warps = next_power_of_2(min(max(conditional_product(x, y, z) // 256, 1), 8))
    return Config(cfg, num_warps=num_warps, num_stages=num_stages)


def triton_config_reduction(size_hints, x, r, num_stages=2):
    """
    Construct a reduction triton config with some adjustment heuristics
    based on size_hints. Size_hints is a tuple of numels in each tile
    dimension and will be rounded up to the nearest power of 2.
    """

    target = conditional_product(x, r)
    if conditional_product(*size_hints) < target:
        target //= 8

    # shrink sizes to size hints
    x = min(x, size_hints[0])
    r = min(r, size_hints[1])

    # if we are below original block size, scale up where we can
    while x < size_hints[0] and conditional_product(x, r) < target:
        x *= 2
    while r < size_hints[1] and conditional_product(x, r) < target:
        r *= 2

    cfg = {"XBLOCK": x, "RBLOCK": r}
    num_warps = next_power_of_2(min(max(conditional_product(x, r) // 128, 1), 8))
    return Config(cfg, num_warps=num_warps, num_stages=num_stages)


def triton_config_tiled_reduction(size_hints, x, y, r, num_stages=2):
    """
    Construct a tile reduction triton config with some adjustment
    heuristics based on size_hints. Size_hints is a tuple of numels in
    each tile dimension and will be rounded up to the nearest power of 2.
    """

    target = conditional_product(x, y, r)
    if conditional_product(*size_hints) < target:
        target //= 8

    # shrink sizes to size hints
    x = min(x, size_hints[0])
    y = min(y, size_hints[1])
    r = min(r, size_hints[2])

    # if we are below original block size, scale up where we can
    while x < size_hints[0] and conditional_product(x, y, r) < target:
        x *= 2
    while r < size_hints[2] and conditional_product(x, y, r) < target:
        r *= 2
    while y < size_hints[1] and conditional_product(x, y, r) < target:
        y *= 2

    cfg = {"XBLOCK": x, "YBLOCK": y, "RBLOCK": r}
    num_warps = next_power_of_2(min(max(conditional_product(x, y, r) // 256, 1), 8))
    return Config(cfg, num_warps=num_warps, num_stages=num_stages)


def apply_triton_config(config):
    """
    Decorator that applies a fixed triton config using triton.heuristics.
    """

    def getter(name):
        def get(args):
            return config.kwargs[name]

        return get

    return heuristics({name: getter(name) for name in config.kwargs.keys()})


def pointwise_heuristics(size_hints):
    """
    Construct @triton.heuristics() based on size_hints.
    """

    if len(size_hints) == 1:
        return apply_triton_config(triton_config(size_hints, 1024))
    if len(size_hints) == 2:
        if not config.triton.autotune:
            return apply_triton_config(triton_config(size_hints, 64, 64))
        return autotune(
            [
                triton_config(size_hints, 32, 32),
                triton_config(size_hints, 8, 256),
                triton_config(size_hints, 256, 8),
                triton_config(size_hints, 1, 1024),
                triton_config(size_hints, 1024, 1),
            ],
            key=["xnumel", "ynumel"],
        )
    if len(size_hints) == 3:
        if not config.triton.autotune:
            return apply_triton_config(triton_config(size_hints, 16, 16, 16))
        return autotune(
            [
                triton_config(size_hints, 16, 16, 16),
                triton_config(size_hints, 64, 8, 8),
                triton_config(size_hints, 8, 64, 8),
                triton_config(size_hints, 8, 8, 64),
                triton_config(size_hints, 1024, 1, 1),
                triton_config(size_hints, 1, 1024, 1),
                triton_config(size_hints, 1, 1, 1024),
            ],
            key=["xnumel", "ynumel", "znumel"],
        )
    raise NotImplementedError(f"size_hints: {size_hints}")


def reduction_heuristics(size_hints):
    """args to @triton.heuristics()"""

    if len(size_hints) == 2:
        if not config.triton.autotune:
            return apply_triton_config(triton_config_reduction(size_hints, 32, 128))
        return autotune(
            [
                triton_config_reduction(size_hints, 64, 64),
                #                triton_config_reduction(size_hints, 16, 64),
                #                triton_config_reduction(size_hints, 32, 128),
                triton_config_reduction(
                    size_hints, 128, 8
                ),  # this one is the best for outer reduction
                triton_config_reduction(
                    size_hints, 8, 512
                ),  # this and the next one seem very similar but both are needed for perf
                triton_config_reduction(size_hints, 1, 2048, num_stages=1),
            ],
            key=["xnumel", "rnumel"],
        )
    """
    # This is not tested yet:
    if len(size_hints) == 3:
        if not config.triton.autotune:
            return apply_triton_config(
                triton_config_tiled_reduction(size_hints, 16, 16, 16)
            )
        return autotune(
            [
                triton_config_tiled_reduction(size_hints, 16, 16, 16),
                triton_config_tiled_reduction(size_hints, 1, 32, 128),
                triton_config_tiled_reduction(size_hints, 32, 1, 128),
                triton_config_tiled_reduction(size_hints, 1, 1, 2048, num_stages=1),
            ],
            key=["xnumel", "ynumel", "rnumel"],
        )
    """
    raise NotImplementedError(f"size_hints: {size_hints}")


def triton_conv_config(size_hints, x, y=None, z=None, num_stages=1):
    """
    Construct a pointwise triton config with some adjustment heuristics
    based on size_hints. Size_hints is a tuple of numels in each tile
    dimension and will be rounded up to the nearest power of 2.
    """
    # Ideally we want to read this from some device config
    maxGridSize = [2147483647, 65535, 65535]

    target = conditional_product(x, y, z)
    if conditional_product(*size_hints) < target:
        target //= 8

    # shrink sizes to size hints
    x = min(x, size_hints[0])
    if y:
        y = min(y, size_hints[1])
    if z:
        z = min(z, size_hints[2])

    # if we are below original block size, scale up where we can;
    # or if the calculated grid size is larger than the limit, we bump up the corresponding dimension
    while x < size_hints[0] and (
        x * maxGridSize[0] < size_hints[0] or conditional_product(x, y, z) < target
    ):
        x *= 2
    while (
        y
        and y < size_hints[1]
        and (
            y * maxGridSize[1] < size_hints[1] or conditional_product(x, y, z) < target
        )
    ):
        y *= 2
    while (
        z
        and z < size_hints[2]
        and (
            z * maxGridSize[2] < size_hints[2] or conditional_product(x, y, z) < target
        )
    ):
        z *= 2

    cfg = {"BLOCK_M": x}
    if y:
        cfg["BLOCK_N"] = y
    if z:
        cfg["BLOCK_K"] = z
    print(x, y, z)
    num_warps = next_power_of_2(min(max(conditional_product(x, y, z) // 256, 1), 8))
    return Config(cfg, num_warps=num_warps, num_stages=num_stages)


def conv_mm_configs(BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages, **kwargs):
    block_dict = {"BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N, "BLOCK_K": BLOCK_K, **kwargs}
    return triton.Config(block_dict, num_stages=num_warps, num_warps=num_stages)


def conv_autotune(size_hints=None):
    BLOCK_max = [512] * 3
    if size_hints:
        assert len(size_hints) == 3
        for i in range(len(size_hints)):
            if size_hints[i]:
                BLOCK_max[i] = max(min(size_hints[i], BLOCK_max[i]), 16)
    configs = [
        conv_mm_configs(256, 64, 32, 8, 2),
        conv_mm_configs(256, 32, 64, 4, 4),
        conv_mm_configs(256, 32, 32, 4, 4),
        conv_mm_configs(128, 128, 32, 8, 4),
        conv_mm_configs(128, 64, 32, 8, 4),
        conv_mm_configs(128, 32, 64, 4, 4),
        conv_mm_configs(128, 16, 32, 4, 4),
        conv_mm_configs(64, 128, 64, 8, 4),
        conv_mm_configs(64, 128, 32, 8, 4),
        conv_mm_configs(64, 128, 32, 4, 4),
        conv_mm_configs(64, 64, 32, 4, 4),
        conv_mm_configs(64, 32, 32, 4, 4),
    ]
    configs = list(
        filter(
            lambda c: c.kwargs["BLOCK_M"] <= BLOCK_max[0]
            and c.kwargs["BLOCK_N"] <= BLOCK_max[1]
            and c.kwargs["BLOCK_K"] <= BLOCK_max[2],
            configs,
        )
    )
    key = [
        "BATCH",
        "IN_C",
        "IN_H",
        "IN_W",
        "KERNEL_N",
        "KERNEL_H",
        "KERNEL_W",
        "OUT_H",
        "OUT_W",
        # parameters of conv
        "stride_h",
        "stride_w",
        "padding_h",
        "padding_w",
        "dilation_h",
        "dilation_w",
        "output_padding_h",
        "output_padding_w",
        "groups",
        # input strides
        "stride_xc",
        "stride_xh",
        "stride_xw",
    ]
    prune_configs_by = {
        "early_config_prune": conv_early_config_prune,
        "perf_model": estimate_conv_time,
        "top_k": 10,
    }
    return autotune(configs, key, prune_configs_by=prune_configs_by)


def mm_heuristics():
    mm_heuristic = heuristics(
        {
            "EVEN_K": lambda args: args["K"] % (args["BLOCK_K"] * args["SPLIT_K"]) == 0,
        }
    )
    return mm_heuristic


def mm_autotune(size_hints=None, get_io_bound_configs=False):
    BLOCK_max = [512] * 3
    if size_hints:
        assert len(size_hints) == 3
        for i in range(len(size_hints)):
            if size_hints[i]:
                BLOCK_max[i] = max(min(size_hints[i], BLOCK_max[i]), 16)
    configs = [
        # basic configs for compute-bound matmuls
        conv_mm_configs(256, 128, 32, 8, 3, SPLIT_K=1),
        conv_mm_configs(256, 64, 32, 4, 4, SPLIT_K=1),
        conv_mm_configs(128, 256, 32, 8, 3, SPLIT_K=1),
        conv_mm_configs(128, 128, 32, 4, 4, SPLIT_K=1),
        conv_mm_configs(128, 64, 32, 4, 4, SPLIT_K=1),
        conv_mm_configs(128, 32, 32, 4, 4, SPLIT_K=1),
        conv_mm_configs(64, 256, 32, 4, 4, SPLIT_K=1),
        conv_mm_configs(64, 128, 32, 4, 4, SPLIT_K=1),
        conv_mm_configs(64, 32, 32, 2, 5, SPLIT_K=1),
        # good for int8
        conv_mm_configs(256, 128, 128, 8, 3, SPLIT_K=1),
        conv_mm_configs(256, 64, 128, 4, 4, SPLIT_K=1),
        conv_mm_configs(128, 256, 128, 8, 3, SPLIT_K=1),
        conv_mm_configs(128, 128, 128, 4, 4, SPLIT_K=1),
        conv_mm_configs(128, 64, 64, 4, 4, SPLIT_K=1),
        conv_mm_configs(128, 32, 64, 4, 4, SPLIT_K=1),
        conv_mm_configs(64, 256, 128, 4, 4, SPLIT_K=1),
        conv_mm_configs(64, 128, 64, 4, 4, SPLIT_K=1),
        conv_mm_configs(64, 32, 64, 2, 5, SPLIT_K=1),
    ]
    if get_io_bound_configs:
        configs += get_configs_io_bound()
    configs = list(
        filter(
            lambda c: c.kwargs["BLOCK_M"] <= BLOCK_max[0]
            and c.kwargs["BLOCK_N"] <= BLOCK_max[1]
            and c.kwargs["BLOCK_K"] <= BLOCK_max[2],
            configs,
        )
    )
    key = ["M", "N", "K"]
    prune_configs_by = {
        "early_config_prune": mm_early_config_prune,
        "perf_model": estimate_matmul_time,
        "top_k": 10,
    }
    return autotune(configs, key, prune_configs_by=prune_configs_by)


def grid(xnumel, ynumel=None, znumel=None):
    """Helper function to compute triton grids"""

    def grid_fn(meta):
        result = [cdiv(xnumel, meta["XBLOCK"])]
        if ynumel:
            result.append(cdiv(ynumel, meta["YBLOCK"]))
            if znumel:
                result.append(cdiv(znumel, meta["ZBLOCK"]))
        return result

    return grid_fn
