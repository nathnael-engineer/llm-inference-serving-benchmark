# High-Throughput LLM Inference & Serving Simulator

Benchmarking and capacity-planning platform evaluating vLLM PagedAttention, continuous batching, FP8 E4M3 / INT4 AWQ quantization, and speculative decoding acceptance rates under concurrent load.

## Quick Start

```bash
pip install -r requirements.txt
python scripts/run_benchmark.py --concurrency 64 --draft-acceptance 0.78
```
