# 🚀 SUPER NOVA RAG 🚀

## Overview

Hybrid Retrieval-Augmented Generation (RAG) pipeline.

## Pipeline

1. **Chunking by row (*.csv)**
    - add summary column: `combined_info`
    - column `_id` for benchmark only

```bash
python src/chunk_docs_neo4j.py --file src/file_paths.txt 
```

2. **Ensemble: BM25 + Embedding (Qwen3-0.6B / Qwen3-8B)**
    - Todo: BM25 for Vietnamese documents
1. **Reranker**
    - Qwen3-0.6B-Reranker: broken!
    - BAAI/bge-reranker-v2-m3

> 🔹 BM25 removed in benchmark to measure *pure embedding performance*.  

## Benchmark: top 100 embedding (Qwen3-8B) -> top (1,5,20) reranking (bge-reranker-v2-m3)

```bash
Final scores: {'hit@1': 0.88125, 'hit@5': 0.98125, 'hit@20': 1.0, 'missed': []}
```

## Benchmark: embedding only (Qwen3-0.6B) (no rerank)

```bash
Final scores:
{'hit@1': 0.653125, 'hit@5': 0.9, 'hit@10': 0.940625, 'missed': [{'id': '666baeb59793e149fe7393e5', 'query': 'Tôi muốn mua điện thoại ai, bên bạn còn hàng không?', 'expected': 'điện thoại ai'}, {'id': '666baeb59793e149fe7393e3', 'query': 'Tôi muốn mua điện thoại ai, bên bạn còn hàng không?', 'expected': 'điện thoại ai'}, {'id': '666baeb59793e149fe7393e4', 'query': 'Tôi muốn mua điện thoại ai, bên bạn còn hàng không?', 'expected': 'điện thoại ai'}, {'id': '666baeb59793e149fe7393e2', 'query': 'Tôi đang quan tâm đến điện thoại ai, bạn có thể tư vấn thêm cho tôi không?', 'expected': 'điện thoại ai'}, {'id': '666baeb59793e149fe7393ec', 'query': 'Tôi muốn mua điện thoại ai, bên bạn còn hàng không?', 'expected': 'điện thoại ai'}, {'id': '666baeb59793e149fe7393ed', 'query': 'Tôi đang quan tâm đến điện thoại ai, bạn có thể tư vấn thêm cho tôi không?', 'expected': 'điện thoại ai'}, {'id': '666baeb59793e149fe7393f2', 'query': 'điện thoại zte blade v50 design (8gb/256gb) có màu nào và dung lượng bao nhiêu GB vậy?', 'expected': 'điện thoại zte blade v50 design (8gb/256gb)'}, {'id': '666baeb89793e149fe7394a2', 'query': 'Tôi muốn đặt mua điện thoại điện thoại di động xor x2 prime gold, cần làm thế nào?', 'expected': 'điện thoại điện thoại di động xor x2 prime gold'}, {'id': '666baeb89793e149fe7394a5', 'query': 'Cho tôi hỏi điện thoại oppo a16k 3gb/32gb hiện tại giá bao nhiêu?', 'expected': 'điện thoại oppo a16k 3gb/32gb'}, {'id': '666baeb89793e149fe7394b4', 'query': 'điện thoại samsung galaxy a02s 4gb/64gb có màu nào và dung lượng bao nhiêu GB vậy?', 'expected': 'điện thoại samsung galaxy a02s 4gb/64gb'}, {'id': '666baeb89793e149fe7394b0', 'query': 'Cho tôi hỏi oppo a54 hiện tại giá bao nhiêu?', 'expected': 'oppo a54'}, {'id': '666baeb99793e149fe7394d3', 'query': 'Cho tôi hỏi oppo reno4 pro hiện tại giá bao nhiêu?', 'expected': 'oppo reno4 pro'}, {'id': '666baeb99793e149fe7394cf', 'query': 'Tôi muốn mua điện thoại samsung galaxy z fold2 5g, bên bạn còn hàng không?', 'expected': 'điện thoại samsung galaxy z fold2 5g'}, {'id': '666baeb99793e149fe7394d1', 'query': 'Cho tôi hỏi điện thoại energizer e241s hiện tại giá bao nhiêu?', 'expected': 'điện thoại energizer e241s'}, {'id': '666baeb99793e149fe7394d4', 'query': 'Tôi muốn mua oppo reno4, bên bạn còn hàng không?', 'expected': 'oppo reno4'}, {'id': '666baeb99793e149fe7394dd', 'query': 'Tôi muốn mua điện thoại samsung galaxy s20 ultra, bên bạn còn hàng không?', 'expected': 'điện thoại samsung galaxy s20 ultra'}, {'id': '666baeb99793e149fe7394de', 'query': 'Cho tôi hỏi điện thoại samsung galaxy s20 hiện tại giá bao nhiêu?', 'expected': 'điện thoại samsung galaxy s20'}, {'id': '666baeb99793e149fe7394d7', 'query': 'Cho tôi hỏi oppo a53 hiện tại giá bao nhiêu?', 'expected': 'oppo a53'}, {'id': '666baeb99793e149fe7394e0', 'query': 'Cho tôi hỏi đồng hồ thông minh samsung galaxy fit e (sm hiện tại giá bao nhiêu?', 'expected': 'đồng hồ thông minh samsung galaxy fit e (sm'}]}
```

## Benchmark: top 50 embedding (Qwen3-0.6B) -> top (1,5,10) reranking (bge-reranker-v2-m3)

- **Dataset**: 320 records (src\data\hoanghamobile_with_summary.csv)
- **Mode**: Embedding-only retrieval
- **Metric**: `hit@k` (1, 5, 10)

```bash
Final scores:
{'hit@1': 0.85625, 'hit@5': 0.978125, 'hit@10': 0.978125, 'missed': [{'id': '666baeb59793e149fe7393e5', 'query': 'Cho tôi hỏi điện thoại ai hiện tại giá bao nhiêu?', 'expected': 'điện thoại ai'}, {'id': '666baeb59793e149fe7393e4', 'query': 'Cho tôi hỏi điện thoại ai hiện tại giá bao nhiêu?', 'expected': 'điện thoại ai'}, {'id': '666baeb59793e149fe7393e2', 'query': 'Tôi muốn mua điện thoại ai, bên bạn còn hàng không?', 'expected': 'điện thoại ai'}, {'id': '666baeb59793e149fe7393ec', 'query': 'Cho tôi hỏi điện thoại ai hiện tại giá bao nhiêu?', 'expected': 'điện thoại ai'}, {'id': '666baeb79793e149fe739457', 'query': 'Tôi muốn đặt mua điện thoại samsung galaxy s23, cần làm thế nào?', 'expected': 'điện thoại samsung galaxy s23'}, {'id': '666baeb79793e149fe739462', 'query': 'Cho tôi hỏi vivo v25 pro 8gb/128gb hiện tại giá bao nhiêu?', 'expected': 'vivo v25 pro 8gb/128gb'}, {'id': '666baeb99793e149fe7394d8', 'query': 'Cho tôi hỏi oppo a12 hiện tại giá bao nhiêu?', 'expected': 'oppo a12'}]}
```

## Benchmark: top 50 embedding (Qwen3-0.6B) + 50 BM25 (dedup) -> top (1,5,20) reranking (bge-reranker-v2-m3)

```bash
Final scores: 
{'hit@1': 0.86875, 'hit@5': 0.96875, 'hit@20': 0.996875, 'missed': [{'id': '666baeb79793e149fe739450', 'query': 'Tôi đang quan tâm đến realme c55, bạn có thể tư vấn thêm cho tôi không?', 'expected': 'realme c55'}]}
```

## ⚙️ Setup & Run (with uv)

```bash
pip install uv
uv pip install -r requirements.txt --index-strategy unsafe-best-match
```

Run the MCP server

```bash
uv run uvicorn src.mcp_server:app --host 0.0.0.0 --port 8000 --reload
```

## Docker

```bash
bash run_docker.sh
```

neo4j web: localhost:7474

## 🔑 Environment Configuration (.env)

Create a `.env` file in the project root with the following variables:

```bash
# Embedding service (Qwen3-0.6B)
# openai compatible
OPENAI_BASE_URL_EMBED=http://localhost:8080/v1
OPENAI_API_KEY_EMBED=dummy_text
# for vllm: Qwen/Qwen3-Embedding-0.6B
OPENAI_API_MODEL_NAME_EMBED=Qwen/Qwen3-Embedding-0.6B
# 1024 for Qwen3-0.6B
EMBED_DIM=1024

# Reranker service (BGE-Reranker)
OPENAI_BASE_URL_RERANK=http://localhost:8081
OPENAI_API_KEY_RERANK=dummy_text
# for vllm: BAAI/bge-reranker-v2-m3
OPENAI_API_MODEL_NAME_RERANK=BAAI/bge-reranker-v2-m3

# Neo4j database
# docker network, docker compose config
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=12345678
NEO4J_DATABASE=neo4j
```

## VLLM localhost embedding & reranker

```yaml
services:
  embedding-service:
    image: vllm/vllm-openai:latest
    container_name: vllm-qwen3-embedding
    network_mode: host
    runtime: nvidia
    # environment:
    #   - HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN}  # Optional: Your HF token
    volumes:
      - ./models:/models  # Cache models
    ipc: host  # Shared memory for PyTorch/vLLM

    # --enforce-eager  Optional: For debugging; disable for better perf
    command: >
      --model Qwen/Qwen3-Embedding-0.6B
      --task embed
      --dtype float16
      --max-model-len 8192
      --host 0.0.0.0
      --port 8080
      --enforce-eager
      --gpu-memory-utilization 0.45
      --hf_overrides '{"matryoshka_dimensions":[1024]}'
    # restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
  bge-reranker:
    image: vllm/vllm-openai:latest
    container_name: vllm-bge-reranker
    network_mode: host
    runtime: nvidia
    # environment:
    #   - HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN}  # Optional: Your HF token
    volumes:
      - ./models:/models  # Cache models
    ipc: host  # Shared memory for PyTorch/vLLM
    command: >
      --model BAAI/bge-reranker-v2-m3
      --task score
      --dtype float16
      --max-model-len 8192
      --host 0.0.0.0
      --port 8081
      --enforce-eager
      --gpu-memory-utilization 0.3
    # restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
  ### Qwen3-0.6B-Reranker broken!
  # reranker-service:
  #   image: vllm/vllm-openai:latest
  #   container_name: vllm-qwen3-reranker
  #   network_mode: host
  #   runtime: nvidia
  #   # environment:
  #   #   - HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN}  # Optional: Your HF token
  #   volumes:
  #     - ./models:/models  # Cache models
  #   ipc: host  # Shared memory for PyTorch/vLLM
  #   command: >
  #     --model Qwen/Qwen3-Reranker-0.6B
  #     --task score
  #     --dtype float16
  #     --max-model-len 8192
  #     --port 8081
  #     --hf_overrides '{"architectures": ["Qwen3ForSequenceClassification"],"classifier_from_token": ["no", "yes"],"is_original_qwen3_reranker": true}'
  #     --enforce-eager
  #     --gpu-memory-utilization 0.45
  #   # restart: unless-stopped
  #   deploy:
  #     resources:
  #       reservations:
  #         devices:
  #           - driver: nvidia
  #             count: all
  #             capabilities: [gpu]
```