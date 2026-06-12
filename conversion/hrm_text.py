from __future__ import annotations

import re
import json

from typing import Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf, logger


@ModelBase.register("HrmTextForCausalLM")
class HrmTextModel(TextModel):
    model_arch = gguf.MODEL_ARCH.HRM_TEXT

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        with open(self.dir_model / "config.json", "r", encoding="utf-8") as f:
            self.raw_hparams = json.load(f)

        self.layers_per_stack = self.raw_hparams["num_hidden_layers"]
        self.h_cycles = self.raw_hparams["H_cycles"]
        self.l_cycles = self.raw_hparams["L_cycles"]
        self.physical_block_count = self.layers_per_stack * 2
        self.cache_block_count = self.layers_per_stack * self.h_cycles * (self.l_cycles + 1)

        # GGUF tensors store one physical L stack followed by one physical H stack.
        # The runtime expands these 32 physical layers across 128 KV-cache slots.
        self.block_count = self.physical_block_count
        self.tensor_map = gguf.get_tensor_name_map(self.model_arch, self.block_count)

    def set_vocab(self):
        # HRM-Text ships a Qwen2-style tokenizer.json. Keep it as a plain tokenizer;
        # do not add a chat template for validation GGUFs.
        self._set_vocab_gpt2()

    def get_vocab_base_pre(self, tokenizer) -> str:
        del tokenizer
        return "qwen2"

    def set_gguf_parameters(self):
        hp = self.raw_hparams
        head_dim = hp["head_dim"]

        self.gguf_writer.add_context_length(hp["max_position_embeddings"])
        self.gguf_writer.add_embedding_length(hp["hidden_size"])
        self.gguf_writer.add_block_count(self.cache_block_count)
        self.gguf_writer.add_feed_forward_length(hp["intermediate_size"])
        self.gguf_writer.add_head_count(hp["num_attention_heads"])
        self.gguf_writer.add_head_count_kv(hp["num_key_value_heads"])
        self.gguf_writer.add_key_length(head_dim)
        self.gguf_writer.add_value_length(head_dim)
        self.gguf_writer.add_rope_dimension_count(head_dim)
        self.gguf_writer.add_rope_freq_base(hp.get("rope_theta", 10000.0))
        self.gguf_writer.add_layer_norm_rms_eps(hp["rms_norm_eps"])
        self.gguf_writer.add_embedding_scale(hp["embedding_scale"])

        arch = self.gguf_writer.arch
        self.gguf_writer.add_uint32(gguf.Keys.LLM.HRM_LAYERS_PER_STACK.format(arch=arch), self.layers_per_stack)
        self.gguf_writer.add_uint32(gguf.Keys.LLM.HRM_H_CYCLES.format(arch=arch), self.h_cycles)
        self.gguf_writer.add_uint32(gguf.Keys.LLM.HRM_L_CYCLES.format(arch=arch), self.l_cycles)
        self.gguf_writer.add_bool(gguf.Keys.LLM.HRM_PREFIX_LM.format(arch=arch), bool(hp.get("prefix_lm", False)))

    def _format(self, key: gguf.MODEL_TENSOR, bid: int | None = None, suffix: str = ".weight") -> str:
        return self.format_tensor_name(key, bid=bid, suffix=suffix)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if name == "model.embed_tokens.weight":
            yield self._format(gguf.MODEL_TENSOR.TOKEN_EMBD), data_torch
            return

        if name == "lm_head.weight":
            yield self._format(gguf.MODEL_TENSOR.OUTPUT), data_torch
            return

        if name == "model.z_L_init":
            yield self._format(gguf.MODEL_TENSOR.HRM_Z_L_INIT, suffix=""), data_torch
            return

        match = re.fullmatch(r"model\.([LH])_module\.layers\.(\d+)\.(.+)", name)
        if match is None:
            raise ValueError(f"Can not map tensor {name!r}")

        stack, layer_s, tensor_name = match.groups()
        layer_idx = int(layer_s)
        if layer_idx >= self.layers_per_stack:
            raise ValueError(f"Layer index {layer_idx} outside HRM stack size {self.layers_per_stack}")

        physical_bid = layer_idx + (self.layers_per_stack if stack == "H" else 0)

        if tensor_name == "attn.gqkv_proj.weight":
            gate, q, k, v = torch.chunk(data_torch, 4, dim=0)
            logger.debug("Split %s as gate, q, k, v", name)
            yield self._format(gguf.MODEL_TENSOR.ATTN_GATE, physical_bid), gate.contiguous()
            yield self._format(gguf.MODEL_TENSOR.ATTN_Q, physical_bid), q.contiguous()
            yield self._format(gguf.MODEL_TENSOR.ATTN_K, physical_bid), k.contiguous()
            yield self._format(gguf.MODEL_TENSOR.ATTN_V, physical_bid), v.contiguous()
            return

        if tensor_name == "attn.o_proj.weight":
            yield self._format(gguf.MODEL_TENSOR.ATTN_OUT, physical_bid), data_torch
            return

        if tensor_name == "mlp.gate_up_proj.weight":
            gate, up = torch.chunk(data_torch, 2, dim=0)
            logger.debug("Split %s as gate, up", name)
            yield self._format(gguf.MODEL_TENSOR.FFN_GATE, physical_bid), gate.contiguous()
            yield self._format(gguf.MODEL_TENSOR.FFN_UP, physical_bid), up.contiguous()
            return

        if tensor_name == "mlp.down_proj.weight":
            yield self._format(gguf.MODEL_TENSOR.FFN_DOWN, physical_bid), data_torch
            return

        raise ValueError(f"Can not map tensor {name!r}")
