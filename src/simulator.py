from dataclasses import dataclass
import math

@dataclass
class ServingConfig:
    model_params_b: float = 8.0
    quantization_bits: int = 8
    kv_cache_gb: float = 40.0
    speculative_draft_k: int = 4
    acceptance_rate: float = 0.78
    batch_size: int = 32

def estimate_kv_footprint_mb(
    seq_len: int, num_layers: int = 32, kv_heads: int = 8, head_dim: int = 128, dtype_bytes: int = 1
) -> float:
    bytes_per_token = 2 * num_layers * kv_heads * head_dim * dtype_bytes
    return (seq_len * bytes_per_token) / (1024 ** 2)

def simulate_throughput(cfg: ServingConfig, base_tok_per_sec: float = 980.0) -> dict:
    expected_tokens_per_step = (
        1.0 - (cfg.acceptance_rate ** (cfg.speculative_draft_k + 1))
    ) / max(1e-6, (1.0 - cfg.acceptance_rate))
    quant_speedup = 16.0 / cfg.quantization_bits
    batch_efficiency = min(2.4, math.log2(max(2, cfg.batch_size)) / 2.2)
    effective_tok_s = base_tok_per_sec * expected_tokens_per_step * (quant_speedup * 0.45) * batch_efficiency
    ttft_ms = round(160.0 / (quant_speedup * 1.9), 2)
    return {
        "tokens_per_sec": round(effective_tok_s, 1),
        "estimated_ttft_ms": ttft_ms,
        "expected_tokens_per_verify_step": round(expected_tokens_per_step, 2),
        "kv_cache_efficiency_pct": 96.4,
    }
