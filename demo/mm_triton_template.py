import torch
import torchdynamo
import torchinductor.config

torchinductor.config.debug = True
torchinductor.config.triton.dense_indexing = True
torchinductor.config.triton.mm = "triton"
torch.manual_seed(0) 

# The flag below controls whether to allow TF32 on matmul.
torch.backends.cuda.matmul.allow_tf32 = True


@torchdynamo.optimize("inductor")
def mm_relu(a, b):
    y = torch.mm(a, b)
    return torch.relu(y)

shape = ([128, 9216], [9216, 4096])
dtype = torch.float16
M, K = shape[0]
_, N = shape[1]
torch.manual_seed(0)
# allocate inputs
a = torch.randn(shape[0], device="cuda", dtype=dtype)
b = torch.randn(shape[1], device="cuda", dtype=dtype)

c = mm_relu(a, b)