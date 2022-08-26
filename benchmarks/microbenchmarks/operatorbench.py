#!/usr/bin/env python
import click
import numpy as np
import torch
import triton
from operator_inp_utils import OperatorInputsLoader

from torchdynamo.optimizations.backends import cudagraphs_inner
from torchdynamo.testing import same
from torchinductor import config as inductor_config
from torchinductor.compile_fx import compile_fx
from torchinductor.decomposition import decompositions
from torchinductor.lowering import fallbacks
from torchinductor.lowering import lowerings
from torchinductor.utils import gen_gm_and_inputs

aten = torch.ops.aten


def compute_speedups(repeats, models, example_inputs, accuracy_checking=False):
    expected = models[0](*example_inputs)
    if accuracy_checking:
        for model in models[1:]:
            actual = model(*example_inputs)
            assert same(actual, expected), expected[0] - actual[0]

    timings = np.zeros((repeats, len(models)), np.float64)
    for rep in range(repeats):
        # interleave the runs to handle frequency scaling and load changes
        for m, model in enumerate(models):
            # do_bench() clears L2 cache to hide the latency of CPU launch time
            # along with cuda synchronization
            median_ms, _, _ = triton.testing.do_bench(lambda: model(*example_inputs))
            timings[rep, m] = median_ms
    return np.median(timings, axis=0)


def strip_overloads(gm):
    """
    Modifies the target of graph nodes in :attr:`gm` to strip overloads.
    Args:
        gm(fx.GraphModule): The input Fx graph module to be modified
    """
    for node in gm.graph.nodes:
        if isinstance(node.target, torch._ops.OpOverload):
            node.target = node.target.overloadpacket
    gm.recompile()


def convert_to_jit(gm, gm_args):
    strip_overloads(gm)
    try:
        return torch.jit.script(gm)
    except Exception:
        pass
    return torch.jit.trace(gm, gm_args)


def microbenchmark(target, args, kwargs, dtype, accuracy_checking):
    gm, gm_args = gen_gm_and_inputs(target, args, kwargs)
    torch.jit._builtins._register_builtin(
        torch.ops.aten.convolution_backward.default, "aten::convolution_backward"
    )
    compiled_fn = compile_fx(gm, gm_args)
    cudagraphs_eager = cudagraphs_inner(gm, gm_args, copy_outputs=False)
    g = convert_to_jit(gm, gm_args)
    cudagraphs_jit = cudagraphs_inner(g, gm_args, copy_outputs=False)

    repeats = 3
    medians = compute_speedups(
        repeats,
        [cudagraphs_eager, cudagraphs_jit, compiled_fn],
        gm_args,
    )
    return medians


def skip_operator(operator):
    nyi_strings = (
        "aten.gather.default",
        "nll_loss",
        "aten.index",
        "aten.scatter_",
        "masked_fill_.Scalar",
    )

    if any(nyi_string in str(operator) for nyi_string in nyi_strings):
        # maybe disable aten.native_layer_norm.default
        # TODO - inputs cannot be randomly initialized, causes cyda failures
        print(f"Skipping {operator}, input generator nyi")
        return True

    # not covered by other non-compute operator heuristics
    if operator == torch.ops.aten._unsafe_view.default:
        print(f"Skipping {operator}, non compute operator")
        return True

    # some of inductor registered to the OpOverload, some registered to OpOverloadPacket
    op_impls = [operator]
    if isinstance(operator, torch._ops.OpOverload):
        op_impls.append(operator.overloadpacket)

    if any(op in fallbacks for op in op_impls):
        print(f"Skipping {operator}, no inductor impl")
        return True

    if all(op not in decompositions and op not in lowerings for op in op_impls):
        print(f"Skipping {operator}, no inductor impl")
        return True

    if inductor_config.triton.convolution == "aten" and "convolution" in str(operator):
        return True

    if inductor_config.triton.mm == "aten" and operator in (
        aten.mm.default,
        aten.bmm.default,
        aten.addmm.default,
        aten.matmul.default,
    ):
        return True

    return False


@click.command()
@click.option(
    "--suite",
    help="suite to load inps from: options: timm, huggingface, torchbench",
    default="torchbench",
)
@click.option("--op", help="operator overload to benchmark")
@click.option("--dtype", help="dtype to benchmark")
@click.option("--max-samples", help="max samples per op", default=15)
@click.option("--accuracy-checking", help="check accuracy", default=False)
def benchmark(suite, op, dtype, max_samples, accuracy_checking):
    assert suite in ("timm", "huggingface", "torchbench"), f"got {suite}"
    if suite == "timm":
        loader = OperatorInputsLoader.get_timm_loader()
    elif suite == "huggingface":
        loader = OperatorInputsLoader.get_huggingface_loader()
    else:
        loader = OperatorInputsLoader.get_torchbench_loader()

    assert dtype in ("float16", "float32"), f"got {dtype}"

    if op == "all":
        filename = f"timings_{suite}_{op.replace('.', '_')}{dtype}.txt"
        f = open(filename, "a")

    dtype = torch.float16 if dtype == "float16" else torch.float32

    if op == "all":
        ops = loader.get_all_ops()
    else:
        ops = [eval(op)]

    for operator in ops:
        if skip_operator(operator):
            continue

        print(f"Running {operator}")
        inp_gen = loader.get_inputs_for_operator(operator, dtype=dtype)
        timings = []

        for i in range(min(max_samples, 1000000)):
            print(f"Iter {i}")
            try:
                inps = next(inp_gen)
                if inps is None:
                    break
                args, kwargs = inps
            except StopIteration:
                break
            try:
                # aten, nvfuser, inductor
                timings.append(
                    microbenchmark(operator, args, kwargs, dtype, accuracy_checking)
                )
            except Exception as e:
                print(f"error {operator}")
                print(e)
                pass

        if not timings:
            continue

        timings = torch.tensor(timings).T
        q = torch.tensor([0.2, 0.5, 0.8], dtype=torch.float64)
        output = f"\n{operator}:\nNVFUSER Speedups : {(torch.quantile(timings[0] / timings[1], q)).tolist()}"
        output = f"{output}\nInductor Speedups : {(torch.quantile(timings[0] / timings[2], q)).tolist()}"
        if op == "all":
            f.write(output)
        print(output)

    if op == "all":
        f.close()


if __name__ == "__main__":
    benchmark()
