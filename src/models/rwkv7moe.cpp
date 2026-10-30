#include "models.h"

// RWKV7MOE: identical dense time-mix to RWKV7, but the channel-mix FFN is a DeepSeek-V3-style
// sparse MoE (always-on shared expert + top-k of N routed experts + aux-loss-free bias for
// load-balanced selection). See rwkvt/rwkv7/ffn_moe.py (RWKV_CMix_x070_MoE) in the training
// codebase for the reference PyTorch implementation this mirrors bit-for-bit:
//   - a single token-shift lerp ("xk") feeds the router, the shared expert and every routed
//     expert alike.
//   - the shared expert is an unrouted clone of the same channel-mix FFN shape, always active.
//   - routed experts: top_k of n_expert selected by (router_logits + routing_bias); the mixing
//     weights are softmax(router_logits) over just the selected top_k (bias-free, matches
//     LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT exactly: raw logits as "probs", bias added
//     only to the selection step, softmax applied only after gathering the top_k).
//   - activation is ReLU^2 (no gating multiply), same as dense RWKV7 channel-mix.

void llama_model_rwkv7moe::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS,                hparams.f_norm_eps, false);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,            hparams.f_norm_rms_eps, false);
    ml.get_key(LLM_KV_WKV_HEAD_SIZE,                          hparams.wkv_head_size);
    ml.get_key(LLM_KV_ATTENTION_DECAY_LORA_RANK,              hparams.n_lora_decay);
    ml.get_key(LLM_KV_ATTENTION_ICLR_LORA_RANK,               hparams.n_lora_iclr);
    ml.get_key(LLM_KV_ATTENTION_VALUE_RESIDUAL_MIX_LORA_RANK, hparams.n_lora_value_res_mix);
    ml.get_key(LLM_KV_ATTENTION_GATE_LORA_RANK,               hparams.n_lora_gate, false);
    ml.get_key(LLM_KV_TOKEN_SHIFT_COUNT,                      hparams.token_shift_count, false);

    // n_expert / n_expert_used_arr / feed_forward_length are already read generically for
    // every architecture in llama_model::load_hparams() before this is called.

    type = LLM_TYPE_UNKNOWN;
}

void llama_model_rwkv7moe::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    // Block 0, LN0
    tok_norm   = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD_NORM, "weight", 0), {n_embd}, 0);
    tok_norm_b = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD_NORM, "bias",   0), {n_embd}, 0);

    // output
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output_norm_b = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "bias"), {n_embd}, 0);
    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, n_vocab}, 0);

    const int n_lora_decay = hparams.n_lora_decay;
    const int n_lora_iclr = hparams.n_lora_iclr;
    const int n_lora_value_res_mix = hparams.n_lora_value_res_mix;
    const int n_lora_gate = hparams.n_lora_gate;
    const int attn_hidden_size = n_embd;
    const int ffn_size = hparams.n_ff_arr[0];

    GGML_ASSERT(hparams.n_expert > 0 && "RWKV7MOE requires expert_count to be set");

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm   = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
        layer.attn_norm_b = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "bias", i),   {n_embd}, 0);

        layer.attn_norm_2   = create_tensor(tn(LLM_TENSOR_ATTN_NORM_2, "weight", i), {n_embd}, 0);
        layer.attn_norm_2_b = create_tensor(tn(LLM_TENSOR_ATTN_NORM_2, "bias", i),   {n_embd}, 0);

        layer.time_mix_w0 = create_tensor(tn(LLM_TENSOR_TIME_MIX_W0, "weight", i), {n_embd}, 0);
        layer.time_mix_w1 = create_tensor(tn(LLM_TENSOR_TIME_MIX_W1, "weight", i), {n_embd, n_lora_decay}, 0);
        layer.time_mix_w2 = create_tensor(tn(LLM_TENSOR_TIME_MIX_W2, "weight", i), {n_lora_decay, n_embd}, 0);

        layer.time_mix_a0 = create_tensor(tn(LLM_TENSOR_TIME_MIX_A0, "weight", i), {n_embd}, 0);
        layer.time_mix_a1 = create_tensor(tn(LLM_TENSOR_TIME_MIX_A1, "weight", i), {n_embd, n_lora_iclr}, 0);
        layer.time_mix_a2 = create_tensor(tn(LLM_TENSOR_TIME_MIX_A2, "weight", i), {n_lora_iclr, n_embd}, 0);

        if (i == 0) {
            // actually not used
            layer.time_mix_v0 = create_tensor(tn(LLM_TENSOR_TIME_MIX_V0, "weight", i), {n_embd}, 0);
            layer.time_mix_v1 = create_tensor(tn(LLM_TENSOR_TIME_MIX_V1, "weight", i), {n_embd, n_lora_iclr}, 0);
            layer.time_mix_v2 = create_tensor(tn(LLM_TENSOR_TIME_MIX_V2, "weight", i), {n_lora_iclr, n_embd}, 0);
        } else {
            layer.time_mix_v0 = create_tensor(tn(LLM_TENSOR_TIME_MIX_V0, "weight", i), {n_embd}, 0);
            layer.time_mix_v1 = create_tensor(tn(LLM_TENSOR_TIME_MIX_V1, "weight", i), {n_embd, n_lora_value_res_mix}, 0);
            layer.time_mix_v2 = create_tensor(tn(LLM_TENSOR_TIME_MIX_V2, "weight", i), {n_lora_value_res_mix, n_embd}, 0);
        }

        layer.time_mix_g1 = create_tensor(tn(LLM_TENSOR_TIME_MIX_G1, "weight", i), {n_embd, n_lora_gate}, 0);
        layer.time_mix_g2 = create_tensor(tn(LLM_TENSOR_TIME_MIX_G2, "weight", i), {n_lora_gate, n_embd}, 0);

        layer.time_mix_lerp_fused = create_tensor(tn(LLM_TENSOR_TIME_MIX_LERP_FUSED, "weight", i), {n_embd, 1, 1, 6}, 0);

        layer.time_mix_k_k = create_tensor(tn(LLM_TENSOR_TIME_MIX_K_K, "weight", i), {attn_hidden_size}, 0);
        layer.time_mix_k_a = create_tensor(tn(LLM_TENSOR_TIME_MIX_K_A, "weight", i), {attn_hidden_size}, 0);
        layer.time_mix_r_k = create_tensor(tn(LLM_TENSOR_TIME_MIX_R_K, "weight", i), {attn_hidden_size}, 0);

        layer.time_mix_key = create_tensor(tn(LLM_TENSOR_TIME_MIX_KEY, "weight", i), {attn_hidden_size, n_embd}, 0);
        layer.time_mix_value = create_tensor(tn(LLM_TENSOR_TIME_MIX_VALUE, "weight", i), {attn_hidden_size, n_embd}, 0);
        layer.time_mix_receptance = create_tensor(tn(LLM_TENSOR_TIME_MIX_RECEPTANCE, "weight", i), {attn_hidden_size, n_embd}, 0);

        layer.time_mix_ln = create_tensor(tn(LLM_TENSOR_TIME_MIX_LN, "weight", i), {n_embd}, 0);
        layer.time_mix_ln_b = create_tensor(tn(LLM_TENSOR_TIME_MIX_LN, "bias", i), {n_embd}, 0);
        layer.time_mix_output = create_tensor(tn(LLM_TENSOR_TIME_MIX_OUTPUT, "weight", i), {n_embd, attn_hidden_size}, 0);

        // channel-mix: token-shift lerp is dense (shared by every expert); key/value are MoE'd.
        layer.channel_mix_lerp_k = create_tensor(tn(LLM_TENSOR_CHANNEL_MIX_LERP_K, "weight", i), {n_embd, 1, 1}, 0);

        layer.ffn_gate_inp    = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,    "weight", i), {n_embd, n_expert}, 0);
        layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias",   i), {n_expert}, 0);

        layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd,  ffn_size, n_expert}, 0);
        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {ffn_size, n_embd,  n_expert}, 0);

        layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd,  ffn_size}, 0);
        layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {ffn_size, n_embd}, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_rwkv7moe::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

ggml_tensor * llama_model_rwkv7moe::graph::build_rwkv7moe_channel_mix(const llama_layer * layer,
                                                                       ggml_tensor *       cur,
                                                                       ggml_tensor *       x_prev,
                                                                       int64_t             n_expert,
                                                                       int64_t             n_expert_used,
                                                                       int                 il) const {
    // xk = (x_prev - cur) * lerp_k + cur -- the one preprocessed input shared by the router,
    // the shared expert and every routed expert (mirrors ffn_moe.py's k_in exactly).
    ggml_tensor * sx = ggml_sub(ctx0, x_prev, cur);
    ggml_tensor * xk = ggml_add(ctx0, ggml_mul(ctx0, sx, layer->channel_mix_lerp_k), cur);

    ggml_tensor * moe_out = build_moe_ffn(xk,
            layer->ffn_gate_inp,
            layer->ffn_up_exps,
            /* gate_exps */ nullptr,
            layer->ffn_down_exps,
            layer->ffn_exp_probs_b,
            n_expert, n_expert_used,
            LLM_FFN_RELU_SQR,
            /* norm_w  */ false,
            /* w_scale */ 1.0f,
            // bias is added to the raw router logits for SELECTION only; the mixing weights
            // are softmax(raw logits) over just the selected top_k -- exactly ffn_moe.py's
            // select_logits = logits + routing_bias; gates = softmax(gather(logits, topk_idx)).
            LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT,
            il);
    cb(moe_out, "ffn_moe_out", il);

    ggml_tensor * shexp_out = build_ffn(xk,
            layer->ffn_up_shexp, NULL, NULL,
            NULL, NULL, NULL, // no gate -- shared expert is plain up -> ReLU^2 -> down
            layer->ffn_down_shexp, NULL, NULL,
            NULL, LLM_FFN_RELU_SQR, LLM_FFN_SEQ, il);
    cb(shexp_out, "ffn_shexp_out", il);

    return ggml_add(ctx0, moe_out, shexp_out);
}

llama_model_rwkv7moe::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_build_rwkv7_base(model, params) {
    GGML_ASSERT(hparams.token_shift_count == 2);

    const int64_t n_expert      = hparams.n_expert;
    const int64_t n_expert_used = hparams.n_expert_used();

    ggml_tensor * cur;
    ggml_tensor * inpL;
    ggml_tensor * v_first = nullptr;

    inpL = build_inp_embd(model.tok_embd);
    inpL = build_norm(inpL, model.tok_norm, model.tok_norm_b, LLM_NORM, 0);

    auto * rs_inp = build_rs_inp();

    const auto n_embd       = hparams.n_embd;
    const auto n_seq_tokens = ubatch.n_seq_tokens;
    const auto n_seqs       = ubatch.n_seqs;

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        const llama_layer * layer = &model.layers[il];
        inpL                      = ggml_reshape_3d(ctx0, inpL, n_embd, n_seq_tokens, n_seqs);

        ggml_tensor * token_shift = build_rwkv_token_shift_load(rs_inp, ubatch, il);

        ggml_tensor * att_shift =
            ggml_view_3d(ctx0, token_shift, n_embd, 1, n_seqs, token_shift->nb[1], token_shift->nb[2], 0);
        ggml_tensor * ffn_shift = ggml_view_3d(ctx0, token_shift, n_embd, 1, n_seqs, token_shift->nb[1],
                                               token_shift->nb[2], n_embd * ggml_element_size(token_shift));

        ggml_tensor * att_norm = build_norm(inpL, layer->attn_norm, layer->attn_norm_b, LLM_NORM, il);
        cb(att_norm, "attn_norm", il);

        ggml_tensor * x_prev = ggml_concat(
            ctx0, att_shift,
            ggml_view_3d(ctx0, att_norm, n_embd, n_seq_tokens - 1, n_seqs, att_norm->nb[1], att_norm->nb[2], 0), 1);

        cur = build_rwkv7_time_mix(rs_inp, att_norm, x_prev, v_first, ubatch, il);

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpL);
        cb(ffn_inp, "ffn_inp", il);

        ggml_tensor * ffn_norm = build_norm(ffn_inp, layer->attn_norm_2, layer->attn_norm_2_b, LLM_NORM, il);
        cb(ffn_norm, "ffn_norm", il);

        x_prev = ggml_concat(
            ctx0, ffn_shift,
            ggml_view_3d(ctx0, ffn_norm, n_embd, n_seq_tokens - 1, n_seqs, ffn_norm->nb[1], ffn_norm->nb[2], 0), 1);

        token_shift = ggml_concat(ctx0,
                                  ggml_view_3d(ctx0, att_norm, n_embd, 1, n_seqs, att_norm->nb[1], att_norm->nb[2],
                                               (n_seq_tokens - 1) * n_embd * ggml_element_size(att_norm)),
                                  ggml_view_3d(ctx0, ffn_norm, n_embd, 1, n_seqs, ffn_norm->nb[1], ffn_norm->nb[2],
                                               (n_seq_tokens - 1) * n_embd * ggml_element_size(ffn_norm)),
                                  1);
        ggml_build_forward_expand(gf, build_rwkv_token_shift_store(token_shift, ubatch, il));

        ffn_inp  = ggml_reshape_2d(ctx0, ffn_inp, n_embd, n_tokens);
        ffn_norm = ggml_reshape_2d(ctx0, ffn_norm, n_embd, n_tokens);
        x_prev   = ggml_reshape_2d(ctx0, x_prev, n_embd, n_tokens);

        if (il == n_layer - 1 && inp_out_ids) {
            ffn_inp  = ggml_get_rows(ctx0, ffn_inp, inp_out_ids);
            ffn_norm = ggml_get_rows(ctx0, ffn_norm, inp_out_ids);
            x_prev   = ggml_get_rows(ctx0, x_prev, inp_out_ids);
        }
        cur = build_rwkv7moe_channel_mix(layer, ffn_norm, x_prev, n_expert, n_expert_used, il);
        cur = ggml_add(ctx0, cur, ffn_inp);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }
    cur = inpL;
    cur = build_norm(cur, model.output_norm, model.output_norm_b, LLM_NORM, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur, model.output_s);

    cb(cur, "result_output", -1);
    res->t_logits = cur;
}
