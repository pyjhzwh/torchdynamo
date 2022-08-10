import torch
import torchinductor
import triton
import torchinductor.triton_ops
from torchdynamo.testing import rand_strided

device="cuda"
dtype=torch.float32
a_shape, b_shape = [2048, 768], [768, 768]
a_stride, b_stride = [768, 1], [1, 768]
# a_shape, b_shape = [2048, 768], [768, 3072]
# a_stride, b_stride = [768, 1], [1, 768]
# a_shape, b_shape = [2048, 3072], [3072, 768]
# a_stride, b_stride = [3072, 1], [1, 3072]
a = rand_strided(a_shape, a_stride, device=device, dtype=dtype)
b = rand_strided(b_shape, b_stride, device=device, dtype=dtype)
c = torch.empty_strided((a_shape[0], b_shape[1]), (b_shape[1],1), device=device, dtype=dtype)
print(a.shape, b.shape, c.shape, a.stride(), b.stride(), c.stride())

runnable_kernel = torchinductor.triton_ops.matmul_out
run_args = (a, b, c)
run_kwargs = {}
timing, _, _ = triton.testing.do_bench(lambda: runnable_kernel(*run_args, **run_kwargs))
print("triton", timing)
runnable_kernel = torch.ops.aten.mm.out
run_args = (a, b)
run_kwargs = {"out": c}
timing, _, _ = triton.testing.do_bench(lambda: runnable_kernel(*run_args, **run_kwargs))
print("aten", timing)
