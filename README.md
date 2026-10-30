# llama.cpp RWKV7-MoE

A patch adding support for a bespoke architecture to [llama.cpp](https://github.com/ggml-org/llama.cpp):
**dense RWKV7 time-mix + a DeepSeek-V3-style sparse MoE channel-mix** (always-on shared
expert + top-k of N routed experts + aux-loss-free bias load balancing), built by
sparse-upcycling a dense RWKV7 0.4B donor model. New arch string: `rwkv7moe`.

Built for the [OpenMythos](#) project — a ~9.1B-capacity (blended active compute close to a
1-1.5B dense model) RWKV7-MoE, trained via RWKV-PEFT, destined to run via `llama-server` on a
Raspberry Pi and serve openwebui over its OpenAI-compatible API (bypassing ollama's vendored
llama.cpp fork entirely, since it has no MoE-channel-mix support).

## What's patched

Dense RWKV7 time-mix is reused completely unchanged (`llm_build_rwkv7_base`). Channel-mix is
replaced with llama.cpp's existing generic `build_moe_ffn` machinery — no new ggml ops needed,
just a new architecture wiring together primitives that already exist for other MoE archs
(Qwen2MoE, DeepSeek-V2/V3, etc.):

- `src/llama-arch.h` / `src/llama-arch.cpp` — new `LLM_ARCH_RWKV7MOE` enum value, name
  registration (`"rwkv7moe"`), marked recurrent (shares RWKV's token-shift/time-mix state).
- `src/models/models.h` — `llama_model_rwkv7moe` class declaration.
- `src/models/rwkv7moe.cpp` — the actual implementation: hparam/tensor loading, graph builder.
- `src/llama-model.cpp` — factory dispatch + RoPE-type switch (RWKV uses none).

No tensor-name or MoE-FFN-builder changes were needed — `build_moe_ffn`'s
`LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT` gating mode is an exact match for this model's
training-time routing (bias added to raw router logits for top-k *selection* only; mixing
weights are softmax over just the selected logits, bias-free) — see the comments in
`src/models/rwkv7moe.cpp` for the derivation against the reference PyTorch module
(`RWKV_CMix_x070_MoE` in RWKV-PEFT's `ffn_moe.py`).

## Converter

`convert/convert_rwkv7moe_to_gguf.py` — standalone, does **not** go through
`convert_hf_to_gguf.py`'s `ModelBase` registry (this isn't an AutoModel/HF-config checkpoint).
Writes GGUF directly via `gguf.GGUFWriter`. Expects a single merged PyTorch state dict with
RWKV-PEFT/BlinkDL-style key names (`blocks.{i}.att.*`, `blocks.{i}.ffn.{shared,routed,router,
routing_bias}`, etc. — see the script's docstring for the full key list).

Two bugs caught and fixed during validation (both would have hard-failed on a real checkpoint):

1. **Layer-0 `time_mix_v1`/`v2` rank mismatch.** llama.cpp's loader (inherited from dense
   RWKV7) allocates layer 0's `v1`/`v2` at the ICLR lora rank, but the training code allocates
   them at the value-residual-mix rank for *every* layer including 0. Harmless in practice —
   layer 0's forward pass never reads `v1`/`v2` (it takes the `v_first = v` branch) — so the
   converter substitutes correctly-shaped zero tensors for layer 0 specifically.
2. **`exp_probs_b` tensor name has no `ffn_` prefix** in llama.cpp's naming convention
   (`blk.%d.exp_probs_b`, not `blk.%d.ffn_exp_probs_b` — the one inconsistent one of the
   MoE tensor set).
3. **Blanket f16 cast crashes the very first forward pass on a real-dimensioned model.**
   An earlier version of the converter cast every tensor to the target dtype uniformly. This
   loads fine and even runs on a *tiny* synthetic checkpoint, but aborts on anything real-sized
   with `ggml_abort: unsupported types: dst: f32, src0: f32, src1: f16` inside
   `ggml_compute_forward_mul`. Root cause: ggml's CPU binary-ops dispatcher only implements
   `ggml_mul` for the `(f16, f32)` operand order, never `(f32, f16)` — and every `ggml_mul`
   call here (and in dense RWKV7, verbatim-reused) puts the f32 activation first and the
   weight tensor second. Fix: force F32 for any 1-D tensor, any `_norm.weight`/`_norm.bias`,
   and the specific 2-D+ tensors that feed a `ggml_mul` rather than a `ggml_mul_mat`
   (`time_mix_w1`/`w2`, `time_mix_lerp_fused`, `ffn_gate_inp`) — this exactly mirrors
   upstream `convert_hf_to_gguf.py`'s `ModelBase._tensor_requires_f32()`. Caught by
   reproducing on a *real-dimensioned* (not the original tiny) synthetic checkpoint after the
   first real-checkpoint export's `llama-perplexity` cross-check crashed with that exact
   message — the original tiny checkpoint's small tensors happened to never trigger the
   specific code path that exposes it.
4. **Global `layer_norm_eps` was 64x too large, silently corrupting every regular LayerNorm
   in the network.** Training (`rwkvt/rwkv7/block.py`) defines `ln0`/`ln1`/`ln2`/`ln_out` as
   plain `nn.LayerNorm(n_embd)` with no eps override — PyTorch default `1e-5`. A *different*,
   deliberately-flagged quirk (`rwkvt/rwkv7/att.py`: `eps=(1e-5)*(head_size_divisor**2)`,
   commented `"!!! notice eps value !!!"`) applies only to `ln_x`, the GroupNorm inside
   attention. The converter computed that `6.4e-4` ln_x-specific value and wrote it as the
   GGUF's *global* `layer_norm_eps`, which `build_norm()` applies to every other LayerNorm
   call (~100 across 24 layers) — while llama.cpp's `ln_x` GroupNorm is hardcoded separately
   as `64e-5f` in `rwkv7-base.cpp` and never reads this GGUF key at all. Net effect: the one
   norm meant to get the scaled value never used this setting, and every norm that should
   have gotten plain `1e-5` got 64x too much instead. The model loaded and ran with no crash
   and no shape mismatch — the architecture, quantization, and crash-path tests all passed —
   but every layer's normalization was cumulatively wrong.

   **Caught by the first real numeric cross-check against the trained checkpoint** (three-way
   perplexity: PyTorch merged model vs. f16 GGUF vs. Q4_K_M GGUF, all on the same eval text).
   Before the fix: PyTorch ppl=15.95, f16-GGUF ppl=15702.79, Q4_K_M-GGUF ppl=29121.87 — both
   GGUF variants broken roughly equally badly, which is what pointed at a conversion-time bug
   rather than a quantization-specific one (quantization noise alone doesn't explain a
   1000x perplexity blowup that's already present at f16). After the fix: PyTorch ppl=15.95,
   f16-GGUF ppl=13.13, Q4_K_M-GGUF ppl=13.85 — all in the same ballpark (spread is normal
   variance on a 226-token sample). Confirmed qualitatively too: greedy completion of "The
   capital of France is" went from degenerate repetition (`or or or or...`) to the correct
   `The capital of France is Paris.`
   Fix: `layer_norm_eps = 1e-5` (both converter copies), leaving `ln_x`'s hardcoded C++ value
   untouched since it was already correct and independent of this metadata key.

## Status

Validated against both a small synthetic checkpoint (`convert/make_tiny_test_ckpt.py`,
deliberately built with mismatched lora ranks per layer to exercise bug #1 above) and a
real-dimensioned synthetic checkpoint (same generator, parameterized to the real model's
n_embd=1024, head_size=64, dim_ffn=4096, n_routed=43 — needed to catch bug #3, which the tiny
checkpoint's small tensor sizes never triggered). **Also validated end-to-end against the
actual trained checkpoint** (bug #4 above) and on real Raspberry Pi hardware:

- [x] Compiles clean, zero warnings on the new file.
- [x] `llama-cli`: loads, runs multi-token generation (64+ tokens), recurrent state stable.
- [x] `llama-quantize` → Q4_K_M: per-expert 3D tensors (`ffn_up_exps`/`ffn_down_exps`)
      quantize correctly; quantized model loads and generates.
- [x] `llama-server`: full `/v1/chat/completions` HTTP round-trip — the actual path openwebui
      will hit.
- [x] `llama-perplexity`: real multi-token forward pass (batched prefill, not just
      single-token decode steps) on both f16 and Q4_K_M, at real-model dimensions.
- [x] Chat template: wired to llama.cpp's own `models/templates/llama-cpp-rwkv-world.jinja`
      (`System:`/`User:`/`Assistant:` + `\n\n`, which happens to already match this project's
      SFT data format) plus EOT token 261 (`\n\n`).
- [x] Numeric correctness against the real trained (distill-pretrain) checkpoint: perplexity
      within normal variance of the PyTorch reference, both f16 and Q4_K_M (see bug #4).
- [x] Real-hardware speed benchmark, Raspberry Pi (4 threads, Q4_K_M, 9.11B total / sparse
      MoE-active params, 5.35 GiB on disk): prompt processing 16.47 t/s, generation 6.2 t/s.

## Building

```sh
cmake -B build -DLLAMA_CURL=OFF
cmake --build build --target llama-cli llama-server llama-quantize -j$(nproc)
```

## Converting a checkpoint

```sh
python3 convert/convert_rwkv7moe_to_gguf.py \
  --checkpoint merged_sft_model.pth \
  --vocab-dir /path/to/dir/containing/rwkv_vocab_v20230424.txt \
  --outfile rwkv7moe.gguf \
  --n-routed 43 --top-k 2 --outtype f16
```

Then optionally quantize:

```sh
./build/bin/llama-quantize rwkv7moe.gguf rwkv7moe-q4km.gguf Q4_K_M
```

And serve:

```sh
./build/bin/llama-server -m rwkv7moe-q4km.gguf --port 8080
```

## Applying this patch

```sh
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
git checkout 889f2ec   # see UPSTREAM_BASE.txt for the exact base commit this was built against
git apply /path/to/llama.cpp-mod/0001-rwkv7moe-arch-registration.patch
cp /path/to/llama.cpp-mod/src/models/rwkv7moe.cpp src/models/rwkv7moe.cpp
```

Then build as described above. `src/CMakeLists.txt` globs `models/*.cpp` already, so no
CMake changes are needed for the new file to be picked up.
