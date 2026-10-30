#!/usr/bin/env python3
"""Build a tiny random RWKV7MOE-shaped state dict for smoke-testing the converter + the
llama.cpp graph builder end-to-end (correct shapes/plumbing, not trained/meaningful weights)."""
import torch

torch.manual_seed(0)

N_EMBD = 64
N_LAYER = 2
HEAD_SIZE = 32
N_HEAD = N_EMBD // HEAD_SIZE
DIM_FFN = N_EMBD * 4
N_ROUTED = 4
VOCAB = 65536

D_DECAY = 24
D_AAA = 40   # deliberately != D_MV, to exercise the layer-0 v1/v2 rank-mismatch fix
D_MV = 16
D_GATE = 48


def lin(out_f, in_f):
    return torch.randn(out_f, in_f) * 0.02


sd = {}
sd["emb.weight"] = torch.randn(VOCAB, N_EMBD) * 0.02
sd["blocks.0.ln0.weight"] = torch.ones(N_EMBD)
sd["blocks.0.ln0.bias"] = torch.zeros(N_EMBD)
sd["ln_out.weight"] = torch.ones(N_EMBD)
sd["ln_out.bias"] = torch.zeros(N_EMBD)
sd["head.weight"] = lin(VOCAB, N_EMBD)

for i in range(N_LAYER):
    a = f"blocks.{i}.att."
    f = f"blocks.{i}.ffn."

    sd[f"blocks.{i}.ln1.weight"] = torch.ones(N_EMBD)
    sd[f"blocks.{i}.ln1.bias"] = torch.zeros(N_EMBD)
    sd[f"blocks.{i}.ln2.weight"] = torch.ones(N_EMBD)
    sd[f"blocks.{i}.ln2.bias"] = torch.zeros(N_EMBD)

    for c in ("r", "w", "k", "v", "a", "g"):
        sd[a + f"x_{c}"] = torch.rand(1, 1, N_EMBD) * 0.1

    sd[a + "w0"] = torch.randn(1, 1, N_EMBD) * 0.1
    sd[a + "w1"] = torch.randn(N_EMBD, D_DECAY) * 0.02
    sd[a + "w2"] = torch.randn(D_DECAY, N_EMBD) * 0.02

    sd[a + "a0"] = torch.randn(1, 1, N_EMBD) * 0.1
    sd[a + "a1"] = torch.randn(N_EMBD, D_AAA) * 0.02
    sd[a + "a2"] = torch.randn(D_AAA, N_EMBD) * 0.02

    sd[a + "v0"] = torch.randn(1, 1, N_EMBD) * 0.1 + 1.0
    sd[a + "v1"] = torch.randn(N_EMBD, D_MV) * 0.02
    sd[a + "v2"] = torch.randn(D_MV, N_EMBD) * 0.02

    sd[a + "g1"] = torch.randn(N_EMBD, D_GATE) * 0.02
    sd[a + "g2"] = torch.randn(D_GATE, N_EMBD) * 0.02

    sd[a + "k_k"] = torch.ones(1, 1, N_EMBD) * 0.85
    sd[a + "k_a"] = torch.ones(1, 1, N_EMBD)
    sd[a + "r_k"] = torch.zeros(N_HEAD, HEAD_SIZE)

    sd[a + "receptance.weight"] = lin(N_EMBD, N_EMBD)
    sd[a + "key.weight"] = lin(N_EMBD, N_EMBD)
    sd[a + "value.weight"] = lin(N_EMBD, N_EMBD)
    sd[a + "output.weight"] = lin(N_EMBD, N_EMBD)
    sd[a + "ln_x.weight"] = torch.ones(N_EMBD)
    sd[a + "ln_x.bias"] = torch.zeros(N_EMBD)

    sd[f + "x_k"] = torch.rand(1, 1, N_EMBD) * 0.1
    sd[f + "shared_key.weight"] = lin(DIM_FFN, N_EMBD)
    sd[f + "shared_value.weight"] = lin(N_EMBD, DIM_FFN)
    for e in range(N_ROUTED):
        sd[f + f"routed_key.{e}.weight"] = lin(DIM_FFN, N_EMBD)
        sd[f + f"routed_value.{e}.weight"] = lin(N_EMBD, DIM_FFN)
    sd[f + "router.weight"] = lin(N_ROUTED, N_EMBD)
    sd[f + "routing_bias"] = torch.zeros(N_ROUTED)

torch.save(sd, "tiny_rwkv7moe.pth")
print("wrote tiny_rwkv7moe.pth:", sum(v.numel() for v in sd.values()), "params")
print(f"N_EMBD={N_EMBD} N_LAYER={N_LAYER} HEAD_SIZE={HEAD_SIZE} DIM_FFN={DIM_FFN} N_ROUTED={N_ROUTED} VOCAB={VOCAB}")
