FROM vllm/vllm-openai:v0.6.2
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY scripts ./scripts
EXPOSE 8080
ENTRYPOINT ["python3", "scripts/run_benchmark.py"]
