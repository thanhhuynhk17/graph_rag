# 🚀 SUPER NOVA RAG 🚀

## Overview
Hybrid Retrieval-Augmented Generation (RAG) pipeline.

## Pipeline
1. **Chunking by row (*.csv)**
2. **Ensemble: BM25 + Embedding (Qwen3-0.6B)**
3. **Reranker (Qwen3-0.6B)**

> 🔹 BM25 removed in benchmark to measure *pure embedding performance*.  

## Benchmark
- **Dataset**: 320 records
- **Mode**: Embedding-only retrieval
- **Metric**: `hit@k` (1, 5, 10)