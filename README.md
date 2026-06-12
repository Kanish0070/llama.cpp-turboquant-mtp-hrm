

# Atomic llama.cpp (HRM Patch Edition)



![atomic llama](https://github.com/AtomicBot-ai/.github/raw/main/assets/atomic%20llama.png)



[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)



**This repository is a specialized fork that introduces native Hierarchical Reasoning Model (HRM) architecture support and structural GGUF tensor mapping, built on top of the AtomicChat TurboQuant/MTP inference engine.**



---



## 🚀 Fork Contributions: Native HRM Architecture Integration



While this repository is built on top of high-performance speculative and quantization forks, **the core contribution of this fork is the manual integration and architectural enablement of the HRM (Hierarchical Reasoning Model) pipeline inside the C++ runtime layer.** Instead of treating HRM like a standard causal transformer, this patch updates the GGUF script converters, header constants, and graph compilers to correctly map and execute macro-recurrent network states.






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
