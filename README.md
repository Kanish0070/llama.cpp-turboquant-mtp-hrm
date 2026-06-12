```markdown

# Atomic llama.cpp (HRM Patch Edition)



![atomic llama](https://github.com/AtomicBot-ai/.github/raw/main/assets/atomic%20llama.png)



[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)



**This repository is a specialized fork that introduces native Hierarchical Reasoning Model (HRM) architecture support and structural GGUF tensor mapping, built on top of the AtomicChat TurboQuant/MTP inference engine.**



---



## 🚀 Fork Contributions: Native HRM Architecture Integration



While this repository is built on top of high-performance speculative and quantization forks, **the core contribution of this fork is the manual integration and architectural enablement of the HRM (Hierarchical Reasoning Model) pipeline inside the C++ runtime layer.** Instead of treating HRM like a standard causal transformer, this patch updates the GGUF script converters, header constants, and graph compilers to correctly map and execute macro-recurrent network states.



### 1. GGUF Parameter & Tensor Registration

* **Metadata Registration:** Updated `conversion/hrm_text.py` and `gguf-py` constants to natively register the `MODEL_ARCH.HRM_TEXT` enum type.

* **Hparams Handling:** The engine natively extracts HRM-specific variables (`layers_per_stack`, `H_cycles`, and `L_cycles`) right out of the GGUF header file to compute expanded runtime KV-cache slots.

* **Tensor Split Mapping:** Configured proper splitting and contiguous memory handling for custom model keys like `model.z_L_init`, `attn.gqkv_proj.weight` (chunked into gate, Q, K, and V), and gated MLP layers.



### 2. Workspace Node Optimization (`graph_max_nodes`)

* Because of the loop-unrolling and state dependencies inside an HRM graph, node memory demands scale much higher than normal models. 

* Patched `llama-context.cpp` to automatically allocate a worst-case threshold of **$n\_tokens \times 80$** (or $64 \times \text{total tensors}$) whenever an `LLM_ARCH_HRM_TEXT` model footprint is detected, preventing early scratchpad memory exhaustion and runtime segmentation faults.



### 3. Recurrent Graph Execution Block

* Implemented the core initialization mechanics inside `src/models/hrm-text.cpp` to structurally deploy the macro-recurrent looping framework across $H$ and $L$ cycles, routing states cleanly through the tensor layout:



```cpp

for (int h = 0; h < h_cycles; ++h) {

    for (int l = 0; l < l_cycles; ++l) {

        const int slot_offset = int((h * (l_cycles + 1) + l) * n_stack);

        hidden_low = build_stack(ggml_add(ctx0, hidden_low, hidden_high), slot_offset);

    }

    const int slot_offset = int((h * (l_cycles + 1) + l_cycles) * n_stack);

    hidden_high = build_stack(ggml_add(ctx0, hidden_high, hidden_low), slot_offset);

}



```



---



## 🔥 Inherited Base Engine Capabilities



This fork retains all top-tier performance features from the underlying `atomic-llama-cpp-turboquant` base engine:



* **Gemma 4 MTP Speculative Decoding:** Pairs a `gemma4` target with the official assistant prediction head for a **~+30-50% throughput acceleration** on short prompts.

* **Qwen 3.6 NextN Speculative Decoding:** Shared-model draft execution utilizing a secondary context over a single memory map (**+24-36% token-generation speedups** on MoE targets).

* **TurboQuant KV Compression:** Native support for rotated low-bit quantization matrices. Deploy `-ctk turbo3 -ctv turbo3` for a **~4.3× reduction in VRAM context allocation**, shifting the OOM ceiling on consumer GPUs.



---



## 🛠️ Quick Start



### Building from Source



Follow the standard `llama.cpp` compilation targets for your hardware layout. For example, to compile with CUDA support on Windows/Linux:



```bash

cmake -B build -DGGML_CUDA=ON

cmake --build build --config Release



```



### Basic Inference Example



```sh

llama-cli -m "path/to/HRM-Text-1B-BF16.gguf" -p "synth,cot\nDeep learning is" -n 50 --gpu-layers 99 --flash-attn on



```



*(Note: As the base HRM model is a pre-alignment checkpoint, prioritize structured token patterns or composite prefix wrapping inside your prompt strings to prevent unaligned sequence behavior).*



---



## 🏛️ Supported Layouts & Devices



* **CUDA:** Optimized execution profiles for standalone NVIDIA GPUs (tested on consumer Laptop RTX 4060 configurations).

* **Metal:** First-class silicon routing for Apple hardware structures.

* **Vulkan / SYCL / CPU:** Reference fallback capabilities for generic computing arrays.



---



*Maintained under the terms of the open-source MIT License.*



```



```
