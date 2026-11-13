import argparse
import json
from src.simulator import ServingConfig, estimate_kv_footprint_mb, simulate_throughput

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--draft-acceptance", type=float, default=0.78)
    args = parser.parse_args()

    fp16_baseline = simulate_throughput(
        ServingConfig(quantization_bits=16, speculative_draft_k=0, batch_size=16)
    )
    fp8_spec = simulate_throughput(
        ServingConfig(
            quantization_bits=8,
            speculative_draft_k=4,
            acceptance_rate=args.draft_acceptance,
            batch_size=args.concurrency,
        )
    )
    print(
        json.dumps(
            {
                "fp16_baseline": fp16_baseline,
                "fp8_speculative": fp8_spec,
                "kv_cache_32k_fp8_mb": round(estimate_kv_footprint_mb(32768, dtype_bytes=1), 2),
            },
            indent=2,
        )
    )

if __name__ == "__main__":
    main()
