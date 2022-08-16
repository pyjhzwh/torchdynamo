import torch
import torchdynamo
import torchinductor.config


torchinductor.config.debug = True
torchinductor.config.triton.dense_indexing = True
torchinductor.config.triton.convolution = "triton"
torch.manual_seed(0)

@torchdynamo.optimize("inductor")
def conv_torchinductor(x, w, bias, stride, padding, dilation, groups):
    y =  torch.conv2d(x, w, None, stride, padding, dilation, groups)
    return y


dilation = (1, 1)
groups = 1
dtype = torch.float32
BLOCK_M, BLOCK_N, BLOCK_K, NSTAGE, NWARP = 128, 128, 64, 2, 4
BATCH = 128
IN_H, IN_W, IN_C, KERNEL_H, KERNEL_W, KERNEL_N, stride, padding = \
    64, 32, 32, 64, 1, 1, (1, 1), (0, 0)


# allocate inputs, nchw
x = torch.randn((BATCH, IN_C, IN_H, IN_W), dtype=dtype, device='cuda')
w = torch.randn((KERNEL_N, IN_C // groups, KERNEL_H, KERNEL_W),
                dtype=dtype, device='cuda')

y = conv_torchinductor(x, w, None, stride, padding, dilation, groups)