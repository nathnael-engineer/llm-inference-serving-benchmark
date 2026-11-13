import asyncio
import time
from typing import List
import httpx
import numpy as np

async def send_prompt(client: httpx.AsyncClient, url: str, prompt: str) -> dict:
    t0 = time.perf_counter()
    resp = await client.post(url, json={"prompt": prompt, "max_tokens": 128}, timeout=60.0)
    resp.raise_for_status()
    return {"latency_ms": (time.perf_counter() - t0) * 1000.0}

async def run_load_test(endpoint: str, concurrency: int = 32) -> dict:
    prompts = [f"Explain PagedAttention KV-cache block #{i}." for i in range(concurrency)]
    async with httpx.AsyncClient() as client:
        results: List[dict] = await asyncio.gather(
            *(send_prompt(client, endpoint, p) for p in prompts)
        )
    latencies = [r["latency_ms"] for r in results]
    return {
        "requests": len(results),
        "p50_ms": round(float(np.percentile(latencies, 50)), 2),
        "p99_ms": round(float(np.percentile(latencies, 99)), 2),
    }
