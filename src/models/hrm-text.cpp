#include "models.h"
#include <cmath>
#include <vector>

llama_model_hrm_text::graph::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    GGML_ASSERT(model.tok_embd != nullptr);
    GGML_ASSERT(model.output != nullptr);
    GGML_ASSERT(model.hrm_z_l_init != nullptr);

    const auto & hparams = model.hparams;

    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());
    GGML_ASSERT(n_embd_head == n_rot);

    const int64_t n_stack = hparams.n_hrm_layer_per_stack;
    const int64_t h_cycles = hparams.n_hrm_h_cycles;
    const int64_t l_cycles = hparams.n_hrm_l_cycles;

    ggml_tensor * inp_pos = build_inp_pos();
    auto * inp_attn = build_attn_inp_kv();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    ggml_tensor * hidden_high = build_inp_embd(model.tok_embd);
    ggml_tensor * hidden_low = ggml_repeat(ctx0, model.hrm_z_l_init, hidden_high);
    cb(hidden_low, "hrm_z_l_init", -1);

    const float kq_scale = 1.0f / std::sqrt(float(n_embd_head));

    auto build_stack = [&](ggml_tensor * stack_inp, int slot_offset) -> ggml_tensor * {
        ggml_tensor * stack_cur = stack_inp;

        for (int layer_idx = 0; layer_idx < n_stack; ++layer_idx) {
            const int il = slot_offset + layer_idx;
            const auto & layer = model.layers[il];

            ggml_tensor * inpSA = stack_cur;
            ggml_tensor * cur = build_norm(stack_cur, nullptr, nullptr, LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);

            {
                ggml_tensor * attn_inp = cur;
                
                ggml_tensor * Qcur = build_lora_mm(layer.wq, cur, layer.wq_s);
                ggml_tensor * Kcur = build_lora_mm(layer.wk, cur, layer.wk_s);
                ggml_tensor * Vcur = build_lora_mm(layer.wv, cur, layer.wv_s);
				
				// ADD THESE THREE LINES: Unflatten the projection tracks into [head_dim, n_heads, n_tokens]
                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, hparams.n_head(il),    n_tokens);
                Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, hparams.n_head_kv(il), n_tokens);
                Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, hparams.n_head_kv(il), n_tokens);

                ggml_tensor * gate = build_lora_mm(layer.wqkv_gate, attn_inp, layer.wqkv_gate_s);
                cb(gate, "attn_gate_proj", il);

				Qcur = ggml_rope_ext(
                        ctx0, Qcur, inp_pos, nullptr,
                        (int)n_rot, 0, // mode 0: Plain RoPE
                        n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow);
                cb(Qcur, "Qcur_rope", il);

				Kcur = ggml_rope_ext(
                        ctx0, Kcur, inp_pos, nullptr,
                        (int)n_rot, 0, // mode 0: Plain RoPE
                        n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow);
                cb(Kcur, "Kcur_rope", il);

                // FIXED: Passed layer.wo and layer.bo tracking arrays perfectly
                cur = build_attn(inp_attn,
                        layer.wo, layer.bo, 
                        Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
                cb(cur, "attn_out", il);

                gate = ggml_sigmoid(ctx0, gate);
                cb(gate, "attn_gate_sig", il);

                cur = ggml_mul(ctx0, cur, gate);
                cb(cur, "attn_gated", il);

                cur = build_lora_mm(layer.wo, cur, layer.wo_s);
                cb(cur, "attn_o_proj", il);
            }

            ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            cur = build_norm(ffn_inp, nullptr, nullptr, LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);

            cur = build_ffn(cur,
                    layer.ffn_up,   nullptr, layer.ffn_up_s,
                    layer.ffn_gate, nullptr, layer.ffn_gate_s,
                    layer.ffn_down, nullptr, layer.ffn_down_s,
                    nullptr,
                    LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);

            cur = ggml_add(ctx0, cur, ffn_inp);
            cur = build_cvec(cur, il);
            cb(cur, "hrm_layer_out", il);

            stack_cur = cur;
        }

        stack_cur = build_norm(stack_cur, nullptr, nullptr, LLM_NORM_RMS, slot_offset);
        cb(stack_cur, "stack_final_norm", slot_offset);
        return stack_cur;
    };

    for (int h = 0; h < h_cycles; ++h) {
        for (int l = 0; l < l_cycles; ++l) {
            const int slot_offset = int((h * (l_cycles + 1) + l) * n_stack);
            hidden_low = build_stack(ggml_add(ctx0, hidden_low, hidden_high), slot_offset);
        }

        const int slot_offset = int((h * (l_cycles + 1) + l_cycles) * n_stack);
        hidden_high = build_stack(ggml_add(ctx0, hidden_high, hidden_low), slot_offset);
    }

    ggml_tensor * cur = hidden_high;

    if (inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur, model.output_s);
    cb(cur, "result_output", -1);

    res->t_logits = cur;
    ggml_build_forward_expand(gf, cur);
}