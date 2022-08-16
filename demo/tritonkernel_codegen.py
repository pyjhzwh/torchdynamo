import torch
import torchdynamo
import torchdynamo.config
import torchinductor.config

torchinductor.config.debug = True
torchdynamo.config.debug = True

@torchdynamo.optimize("inductor", nopython=True)
def add_relu(t1, t2):
    return torch.relu(t1 + t2)

s0, s1, s2 = 32, 64, 128
t1 = torch.randn((s0, s1, s2), dtype=torch.float32, device="cuda")
t2 = torch.randn((s0, s2, s1), dtype=torch.float32, device="cuda")

y = add_relu(t1.permute(0, 2, 1), t2)