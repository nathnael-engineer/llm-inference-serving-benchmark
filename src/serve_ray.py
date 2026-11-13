from ray import serve
from fastapi import FastAPI
from pydantic import BaseModel

api = FastAPI(title="Ray Serve vLLM Router")

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.0

@serve.deployment(num_replicas=1, ray_actor_options={"num_cpus": 2, "num_gpus": 1})
@serve.ingress(api)
class VLLMDeployment:
    def __init__(self, model_id: str = "meta-llama/Meta-Llama-3.1-8B-Instruct") -> None:
        from vllm import AsyncEngineArgs, AsyncLLMEngine

        engine_args = AsyncEngineArgs(
            model=model_id,
            quantization="fp8",
            kv_cache_dtype="fp8",
            enable_prefix_caching=True,
            gpu_memory_utilization=0.92,
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

    @api.post("/v1/generate")
    async def generate(self, req: GenerateRequest) -> dict:
        from vllm import SamplingParams
        import uuid

        params = SamplingParams(temperature=req.temperature, max_tokens=req.max_tokens)
        request_id = str(uuid.uuid4())
        final_text = ""
        async for output in self.engine.generate(req.prompt, params, request_id):
            final_text = output.outputs[0].text
        return {"request_id": request_id, "text": final_text}

deployment = VLLMDeployment.bind()
