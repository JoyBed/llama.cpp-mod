#!/usr/bin/env python3
"""
Standalone GGUF converter for the bespoke RWKV7MOE architecture (dense RWKV7 time-mix +
DeepSeek-V3-style shared-expert/routed-expert/aux-loss-free-bias sparse MoE channel-mix).

Written against the matching llama.cpp patch (src/models/rwkv7moe.cpp, arch string
"rwkv7moe") rather than through convert_hf_to_gguf.py's ModelBase registry, because this
is a bespoke checkpoint format, not an AutoModel/HF config -- going through gguf.GGUFWriter
directly keeps full control over tensor names/shapes without inventing a fake HF config.json.

Expects a single merged PyTorch state dict (plain torch.save'd dict of str -> Tensor) with
BlinkDL/RWKV-PEFT-style key names, e.g.:
  emb.weight
  blocks.0.ln0.{weight,bias}
  blocks.{i}.ln1.{weight,bias}, blocks.{i}.ln2.{weight,bias}
  blocks.{i}.att.{x_r,x_w,x_k,x_v,x_a,x_g,w0,w1,w2,a0,a1,a2,v0,v1,v2,g1,g2,k_k,k_a,r_k,
                  receptance.weight,key.weight,value.weight,output.weight,ln_x.{weight,bias}}
  blocks.{i}.ffn.x_k
  blocks.{i}.ffn.shared_key.weight, blocks.{i}.ffn.shared_value.weight
  blocks.{i}.ffn.routed_key.{e}.weight, blocks.{i}.ffn.routed_value.{e}.weight   (e=0..n_routed-1)
  blocks.{i}.ffn.router.weight
  blocks.{i}.ffn.routing_bias
  ln_out.{weight,bias}
  head.weight

If your checkpoint is still the sharded dense_shared.pt + layer_N.pt form that
build_moe_checkpoint.py produces (see moe_build/), merge it into one state dict first (and
merge any LoRA/PEFT adapter into base weights) -- this script intentionally does not know
about adapters, only the final flat weight tensors.

Usage:
  python3 convert_rwkv7moe_to_gguf.py \\
      --checkpoint merged_sft_model.pth \\
      --vocab-dir /path/to/dir/containing/rwkv_vocab_v20230424.txt \\
      --outfile rwkv7moe.gguf \\
      [--n-routed 43] [--top-k 2] [--outtype f16|f32|bf16]
"""
import argparse
import ast
import re
import sys
from pathlib import Path

import numpy as np
import torch


def load_gguf():
    # allow running against the repo's own vendored gguf-py without installing it
    here = Path(__file__).resolve().parent
    candidate = here.parent / "llama.cpp" / "gguf-py"
    if candidate.is_dir():
        sys.path.insert(0, str(candidate))
    import gguf  # noqa: E402
    return gguf


def build_rwkv_world_vocab(gguf, vocab_path: Path, vocab_size: int):
    tokens: list[bytes] = ["<s>".encode("utf-8")]
    toktypes: list[int] = [gguf.TokenType.CONTROL]

    with open(vocab_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            parts = line.split(" ")
            assert len(parts) >= 3
            token, token_len = ast.literal_eval(" ".join(parts[1:-1])), int(parts[-1])
            token = token.encode("utf-8") if isinstance(token, str) else token
            assert isinstance(token, bytes)
            assert len(token) == token_len
            token_text = repr(token)[2:-1]  # "b'\\xff'" -> "\\xff"
            tokens.append(token_text.encode("utf-8"))
            toktypes.append(gguf.TokenType.NORMAL)

    remainder = vocab_size - len(tokens)
    assert remainder >= 0, f"vocab_size {vocab_size} smaller than loaded vocab ({len(tokens)})"
    for i in range(len(tokens), vocab_size):
        tokens.append(f"[PAD{i}]".encode("utf-8"))
        toktypes.append(gguf.TokenType.UNUSED)

    return tokens, toktypes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="merged state-dict .pth/.pt file")
    ap.add_argument("--vocab-dir", required=True, help="directory containing rwkv_vocab_v20230424.txt")
    ap.add_argument("--outfile", required=True, help="output .gguf path")
    ap.add_argument("--n-routed", type=int, default=43, help="number of routed experts per layer")
    ap.add_argument("--top-k", type=int, default=2, help="experts used per token")
    ap.add_argument("--outtype", choices=["f32", "f16", "bf16"], default="f16")
    ap.add_argument("--head-size", type=int, default=64)
    ap.add_argument("--head-size-divisor", type=float, default=8.0,
                     help="matches args.head_size_divisor used for ln_x eps at training time")
    ap.add_argument("--chat-template", default=None,
                     help="path to a jinja chat template; defaults to llama.cpp's own "
                          "models/templates/llama-cpp-rwkv-world.jinja, which already matches "
                          "our System:/User:/Assistant: SFT format")
    args = ap.parse_args()

    gguf = load_gguf()

    print(f"loading checkpoint {args.checkpoint} ...")
    sd: dict[str, torch.Tensor] = torch.load(args.checkpoint, map_location="cpu", weights_only=True)

    n_embd = sd["emb.weight"].shape[1]
    vocab_size = sd["emb.weight"].shape[0]
    n_layer = max(int(m.group(1)) for k in sd if (m := re.match(r"blocks\.(\d+)\.", k))) + 1
    dim_ffn = sd["blocks.0.ffn.shared_key.weight"].shape[0]
    n_routed = args.n_routed
    top_k = args.top_k
    head_size = args.head_size
    layer_norm_eps = 1e-5 * (args.head_size_divisor ** 2)

    print(f"n_embd={n_embd} n_layer={n_layer} dim_ffn={dim_ffn} vocab={vocab_size} "
          f"n_routed={n_routed} top_k={top_k} head_size={head_size}")

    # sanity-check every layer actually has n_routed experts and consistent shapes
    for i in range(n_layer):
        p = f"blocks.{i}.ffn."
        assert f"{p}routed_key.{n_routed - 1}.weight" in sd, f"layer {i} missing routed expert {n_routed - 1}"
        assert f"{p}routed_key.{n_routed}.weight" not in sd, f"layer {i} has more than {n_routed} routed experts"

    np_dtype = {"f32": np.float32, "f16": np.float16, "bf16": np.float32}[args.outtype]
    # bf16 cast is handled by gguf.GGUFWriter's add_tensor raw_dtype path; keep it simple and
    # let f16/f32 cover the common export cases. bf16 falls back to f32 storage here.
    gguf_ftype = {"f32": gguf.LlamaFileType.ALL_F32,
                  "f16": gguf.LlamaFileType.MOSTLY_F16,
                  "bf16": gguf.LlamaFileType.MOSTLY_BF16}[args.outtype]

    def to_np(t: torch.Tensor, force_f32: bool = False) -> np.ndarray:
        t = t.detach()
        if force_f32 or args.outtype == "bf16":
            # bf16 output type is handled by gguf.GGUFWriter's raw_dtype path elsewhere; here
            # we just avoid truncating precision, same as upstream's _tensor_requires_f32.
            return t.to(torch.float32).numpy()
        if args.outtype == "f16":
            return t.to(torch.float16).numpy().astype(np.float16)
        return t.to(torch.float32).numpy()

    writer = gguf.GGUFWriter(args.outfile, "rwkv7moe", endianess=gguf.GGUFEndian.LITTLE)

    # ---- hparams ----
    writer.add_context_length(1048576)  # RWKV isn't context-limited
    writer.add_embedding_length(n_embd)
    writer.add_block_count(n_layer)
    writer.add_layer_norm_eps(layer_norm_eps)
    writer.add_wkv_head_size(head_size)
    writer.add_feed_forward_length(dim_ffn)
    writer.add_file_type(gguf_ftype)
    writer.add_head_count(0)  # required by llama.cpp's generic loader, unused by RWKV
    writer.add_token_shift_count(2)  # RWKV7 (not ARWKV7) uses separate att/ffn token-shift state
    writer.add_expert_count(n_routed)
    writer.add_expert_used_count(top_k)

    writer.add_decay_lora_rank(sd["blocks.0.att.w1"].shape[1])
    writer.add_iclr_lora_rank(sd["blocks.0.att.a1"].shape[1])
    writer.add_value_residual_mix_lora_rank(sd["blocks.1.att.v1"].shape[1] if n_layer > 1 else sd["blocks.0.att.v1"].shape[1])
    writer.add_gate_lora_rank(sd["blocks.0.att.g1"].shape[1])

    # ---- vocab ----
    vocab_path = Path(args.vocab_dir) / "rwkv_vocab_v20230424.txt"
    tokens, toktypes = build_rwkv_world_vocab(gguf, vocab_path, vocab_size)
    writer.add_tokenizer_model("rwkv")
    writer.add_token_list(tokens)
    writer.add_token_types(toktypes)
    writer.add_bos_token_id(0)
    writer.add_eos_token_id(0)
    writer.add_unk_token_id(0)
    writer.add_pad_token_id(0)
    # RWKV-world convention (matches llama.cpp's dense RWKV7/ARWKV7 converters): id 0 is
    # <|endoftext|>/pad, so it's a poor practical stop token for chat. Token 261 is '\n\n'
    # in this vocab and doubles as end-of-turn -- this lines up exactly with our own SFT
    # data format (System:/User:/Assistant: blocks separated by "\n\n", explicit eos id 0
    # only at the very end), so '\n\n' is also the right per-turn stop signal here.
    writer.add_eot_token_id(261)
    template_path = Path(args.chat_template) if args.chat_template else (
        Path(__file__).resolve().parent.parent / "llama.cpp" / "models" / "templates" / "llama-cpp-rwkv-world.jinja"
    )
    if template_path.is_file():
        writer.add_chat_template(template_path.read_text(encoding="utf-8"))
    else:
        print(f"warning: chat template not found at {template_path}, skipping add_chat_template")

    # ---- tensors ----
    # Any 1-D tensor (biases, lerp/mix vectors, per-channel scalars), any "_norm" weight/bias,
    # and a short explicit list of 2-D+ tensors that feed a ggml_mul (not ggml_mul_mat) --
    # TIME_MIX_W1/W2 (decay lora pair), TIME_MIX_LERP_FUSED, FFN_GATE_INP (router) -- must stay
    # F32 regardless of --outtype. This mirrors upstream convert_hf_to_gguf.py's
    # ModelBase._tensor_requires_f32() exactly. Skipping this (as an earlier version of this
    # script did) produces a GGUF that loads fine but crashes at first forward pass with
    # ggml's binary-ops dispatcher aborting on "unsupported types: dst: f32, src0: f32,
    # src1: f16" -- ggml_mul's CPU kernel only implements the (f16, f32) operand order, not
    # (f32, f16), and our (and upstream's) ggml_mul calls always put the f32 activation first
    # and the weight tensor second, so any of these ending up in a lower-precision dtype than
    # f32 breaks that specific op.
    def add(name: str, t: torch.Tensor, force_f32: bool = False):
        force_f32 = force_f32 or t.dim() <= 1 or name.endswith("_norm.weight") or name.endswith("_norm.bias")
        writer.add_tensor(name, to_np(t, force_f32=force_f32))

    add("token_embd.weight", sd["emb.weight"])

    add("token_embd_norm.weight", sd["blocks.0.ln0.weight"])
    add("token_embd_norm.bias",   sd["blocks.0.ln0.bias"])

    add("output_norm.weight", sd["ln_out.weight"])
    add("output_norm.bias",   sd["ln_out.bias"])
    add("output.weight",      sd["head.weight"])

    for i in range(n_layer):
        a = f"blocks.{i}.att."
        f = f"blocks.{i}.ffn."
        blk = f"blk.{i}."

        add(blk + "attn_norm.weight",   sd[f"blocks.{i}.ln1.weight"])
        add(blk + "attn_norm.bias",     sd[f"blocks.{i}.ln1.bias"])
        add(blk + "attn_norm_2.weight", sd[f"blocks.{i}.ln2.weight"])
        add(blk + "attn_norm_2.bias",   sd[f"blocks.{i}.ln2.bias"])

        # time-mix: dense, unchanged from the RWKV7 donor model.
        #
        # w0/a0/v0 are plain per-channel vectors stored as (1,1,n_embd); flatten to 1-D.
        # w1/a1/v1/g1 are stored as (n_embd, rank) and used as `x @ w1` (i.e. already
        # "input-major"); w2/a2/v2/g2 are stored as (rank, n_embd) and used as
        # `(...) @ w2`. ggml's create_tensor({n_embd, rank}) for *1 and {rank, n_embd}
        # for *2 expects the opposite physical layout GGUFWriter's automatic shape
        # reversal would give these as-is, so both need an explicit .transpose(0, 1)
        # before writing -- exactly the same transpose conversion/rwkv.py applies to
        # the (numerically equivalent, differently-sourced) fla-hub checkpoints.
        add(blk + "time_mix_w0.weight", sd[a + "w0"].reshape(-1))
        add(blk + "time_mix_w1.weight", sd[a + "w1"].transpose(0, 1), force_f32=True)
        add(blk + "time_mix_w2.weight", sd[a + "w2"].transpose(0, 1), force_f32=True)

        add(blk + "time_mix_a0.weight", sd[a + "a0"].reshape(-1))
        add(blk + "time_mix_a1.weight", sd[a + "a1"].transpose(0, 1))
        add(blk + "time_mix_a2.weight", sd[a + "a2"].transpose(0, 1))

        add(blk + "time_mix_v0.weight", sd[a + "v0"].reshape(-1))
        if i == 0:
            # llama.cpp's rwkv7(-moe) loader creates layer 0's time_mix_v1/v2 with the
            # ICLR lora rank (n_lora_iclr), not the value-residual-mix rank
            # (n_lora_value_res_mix) used at every other layer -- see rwkv7-base.cpp's
            # load_arch_tensors `if (i == 0) { ... n_lora_iclr ... }` branch, inherited
            # unchanged by rwkv7moe.cpp. Our training code (att.py) does NOT special-case
            # layer 0's v1/v2 shape -- it always allocates them at the value-residual-mix
            # rank, same as every layer. That's fine: layer 0's forward() takes the
            # `v_first = v` branch and never reads v0/v1/v2 at all, so their contents are
            # dead weights. We can't write our real (wrongly-shaped) v1/v2 into a
            # dim-checked GGUF tensor, so -- matching the upstream dense-RWKV7 converter's
            # own hack of reusing a1/a2 as dummy layer-0 v1/v2 -- substitute zero tensors
            # shaped to what the loader actually expects (n_lora_iclr), which is provably
            # harmless since nothing ever reads them.
            add(blk + "time_mix_v1.weight", torch.zeros_like(sd[a + "a1"]).transpose(0, 1))
            add(blk + "time_mix_v2.weight", torch.zeros_like(sd[a + "a2"]).transpose(0, 1))
        else:
            add(blk + "time_mix_v1.weight", sd[a + "v1"].transpose(0, 1))
            add(blk + "time_mix_v2.weight", sd[a + "v2"].transpose(0, 1))

        add(blk + "time_mix_g1.weight", sd[a + "g1"].transpose(0, 1))
        add(blk + "time_mix_g2.weight", sd[a + "g2"].transpose(0, 1))

        # time_mix_lerp_fused: stack [x_r, x_w, x_k, x_v, x_a, x_g] -> (6, 1, 1, n_embd)
        lerp = torch.stack([sd[a + f"x_{c}"].reshape(-1) for c in ("r", "w", "k", "v", "a", "g")], dim=0)
        lerp = lerp.reshape(6, 1, 1, n_embd)
        add(blk + "time_mix_lerp_fused.weight", lerp, force_f32=True)

        add(blk + "time_mix_k_k.weight", sd[a + "k_k"].reshape(-1))
        add(blk + "time_mix_k_a.weight", sd[a + "k_a"].reshape(-1))
        add(blk + "time_mix_r_k.weight", sd[a + "r_k"].reshape(-1))

        add(blk + "time_mix_key.weight",         sd[a + "key.weight"])
        add(blk + "time_mix_value.weight",       sd[a + "value.weight"])
        add(blk + "time_mix_receptance.weight",  sd[a + "receptance.weight"])

        add(blk + "time_mix_ln.weight", sd[a + "ln_x.weight"])
        add(blk + "time_mix_ln.bias",   sd[a + "ln_x.bias"])
        add(blk + "time_mix_output.weight", sd[a + "output.weight"])

        # channel-mix: token-shift lerp stays dense; key/value become the MoE tensors.
        add(blk + "channel_mix_lerp_k.weight", sd[f + "x_k"].reshape(-1))

        # router.weight is a plain nn.Linear(n_embd, n_routed) weight -- shape
        # (n_routed, n_embd), same (out, in) layout as every other dense weight here,
        # so no manual transpose needed (GGUFWriter's automatic shape reversal handles it).
        add(blk + "ffn_gate_inp.weight", sd[f + "router.weight"], force_f32=True)
        add(blk + "exp_probs_b.bias", sd[f + "routing_bias"])

        up_exps = torch.stack([sd[f + f"routed_key.{e}.weight"] for e in range(n_routed)], dim=0)     # [n_routed, dim_ffn, n_embd]
        down_exps = torch.stack([sd[f + f"routed_value.{e}.weight"] for e in range(n_routed)], dim=0) # [n_routed, n_embd, dim_ffn]
        add(blk + "ffn_up_exps.weight", up_exps)
        add(blk + "ffn_down_exps.weight", down_exps)

        add(blk + "ffn_up_shexp.weight",   sd[f + "shared_key.weight"])
        add(blk + "ffn_down_shexp.weight", sd[f + "shared_value.weight"])

        print(f"  wrote layer {i}/{n_layer}")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()
    print(f"done -> {args.outfile}")


if __name__ == "__main__":
    main()
