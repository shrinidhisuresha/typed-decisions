# Models and data used

Everything below is downloaded from its original source when needed. **No model weights and no
dataset rows are redistributed in this repository.** Licences were checked against each
Hugging Face repository's metadata on 1 October 2026.

## Models

| model | licence | role in this project | where |
|---|---|---|---|
| [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) | Apache-2.0 | zero-shot scorer (`cpu` preset); end-to-end tests | `presets`, `tests/test_end_to_end.py` |
| [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) | Apache-2.0 | model-size check (severity ordering) | development only |
| [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) | Apache-2.0 | zero-shot scorer (`cuda` preset); A/B and benchmarks | `presets`, `examples/` |
| [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) | Apache-2.0 | upstream of the MLX 4-bit conversion | — |
| [mlx-community/Qwen3.5-9B-4bit](https://huggingface.co/mlx-community/Qwen3.5-9B-4bit) | Apache-2.0 (inherits Qwen3.5-9B) | zero-shot scorer on Apple silicon (`apple` preset) | `presets`, `backend_impls/mlx.py` |
| [Qwen/Qwen3.5-0.8B-Base](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base), [Qwen/Qwen3.5-4B-Base](https://huggingface.co/Qwen/Qwen3.5-4B-Base) | Apache-2.0 | BaseCal experiment (not adopted) | `examples/basecal.py` |
| [Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) | Apache-2.0 | retrieval for Choice with more than 26 options; embedding probe | `retrieval.py`, `examples/` |
| [Qwen/Qwen3-Embedding-4B](https://huggingface.co/Qwen/Qwen3-Embedding-4B) | Apache-2.0 | larger retriever (benchmark) | `examples/benchmark_labels.py` |
| [Qwen/Qwen3-Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B), [4B](https://huggingface.co/Qwen/Qwen3-Reranker-4B) | Apache-2.0 | zero-shot reranker (benchmark) | `examples/benchmark_labels.py` |
| [answerdotai/ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base) | Apache-2.0 | default student for distilled heads | `distill.py` |
| [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | Apache-2.0 | fast embedding probe (2.4 ms CPU p50) | `examples/speed_sweep.py` |
| [ibm-granite/granite-embedding-30m-english](https://huggingface.co/ibm-granite/granite-embedding-30m-english) | Apache-2.0 | fast embedding probe (speed sweep) | `examples/speed_sweep.py` |
| [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5), [bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) | MIT | embedding probe (speed sweep) | `examples/speed_sweep.py` |
| [intfloat/e5-small-v2](https://huggingface.co/intfloat/e5-small-v2) | MIT | embedding probe (speed sweep) | `examples/speed_sweep.py` |
| [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) | Apache-2.0 | live vLLM backend verification | `examples/verify_backend.py` |
| [hf-internal-testing/tiny-random-LlamaForCausalLM](https://huggingface.co/hf-internal-testing/tiny-random-LlamaForCausalLM) | not declared | unit tests only (random weights) | `tests/` |

## Datasets

| dataset | licence | used for | redistributed? |
|---|---|---|---|
| [PolyAI/banking77](https://huggingface.co/datasets/PolyAI/banking77) (Casanueva et al., 2020) | CC BY 4.0 | the main benchmark (real user queries) | no, fetched at runtime; **attribution required** if you publish results or models trained on it |
| [bitext/Bitext-customer-support-llm-chatbot-training-dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset) | CDLA-Sharing-1.0 | intent and category benchmarks | no; the share-alike terms apply if you redistribute the data |
| [Tobi-Bueck/customer-support-tickets](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets) | CC BY-NC 4.0 | ticket queue, priority and incident benchmarks | no. **Non-commercial:** don't use models trained on it commercially, and don't publish such models as commercial-use |

## Software

PyTorch (BSD-3), Hugging Face Transformers (Apache-2.0), mlx and mlx-lm (MIT), httpx (BSD-3).
Optional and not bundled: vLLM (Apache-2.0), SGLang (Apache-2.0).

## Citation

If you use the BANKING77 results, cite: Casanueva, I., Temčinas, T., Gerz, D., Henderson, M.,
Vulić, I. (2020). *Efficient Intent Detection with Dual Sentence Encoders.* NLP4ConvAI.
