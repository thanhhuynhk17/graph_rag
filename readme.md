# 🚀 SUPER NOVA RAG 🚀

## Overview

Hybrid Retrieval-Augmented Generation (RAG) pipeline.

## Pipeline

1. **Chunking by row (*.csv)**
2. **Ensemble: BM25 + Embedding (Qwen3-0.6B)**
3. **Reranker (Qwen3-0.6B / bge-reranker-v2-m3)**

> 🔹 BM25 removed in benchmark to measure *pure embedding performance*.  

## Benchmark: top 50 embedding -> top (1,5,10) reranking (bge-reranker-v2-m3)

- **Dataset**: 320 records (src\data\hoanghamobile_with_summary.csv)
- **Mode**: Embedding-only retrieval
- **Metric**: `hit@k` (1, 5, 10)

```bash
Final scores:
{'hit@1': 0.85625, 'hit@5': 0.978125, 'hit@10': 0.978125, 'missed': [{'id': '666baeb59793e149fe7393e5', 'query': 'Cho tôi hỏi điện thoại ai hiện tại giá bao nhiêu?', 'expected': 'điện thoại ai'}, {'id': '666baeb59793e149fe7393e4', 'query': 'Cho tôi hỏi điện thoại ai hiện tại giá bao nhiêu?', 'expected': 'điện thoại ai'}, {'id': '666baeb59793e149fe7393e2', 'query': 'Tôi muốn mua điện thoại ai, bên bạn còn hàng không?', 'expected': 'điện thoại ai'}, {'id': '666baeb59793e149fe7393ec', 'query': 'Cho tôi hỏi điện thoại ai hiện tại giá bao nhiêu?', 'expected': 'điện thoại ai'}, {'id': '666baeb79793e149fe739457', 'query': 'Tôi muốn đặt mua điện thoại samsung galaxy s23, cần làm thế nào?', 'expected': 'điện thoại samsung galaxy s23'}, {'id': '666baeb79793e149fe739462', 'query': 'Cho tôi hỏi vivo v25 pro 8gb/128gb hiện tại giá bao nhiêu?', 'expected': 'vivo v25 pro 8gb/128gb'}, {'id': '666baeb99793e149fe7394d8', 'query': 'Cho tôi hỏi oppo a12 hiện tại giá bao nhiêu?', 'expected': 'oppo a12'}]}
```

## Benchmark: top 100 embedding -> top (1,5,20) reranking

```bash
Final scores: 
{'hit@1': 0.88125, 'hit@5': 0.984375, 'hit@20': 0.984375, 'missed': [{'id': '666baeb79793e149fe739457', 'query': 'Tôi muốn mua điện thoại samsung galaxy s23, bên bạn còn hàng không?', 'expected': 'điện thoại samsung galaxy s23'}, {'id': '666baeb79793e149fe739462', 'query': 'Cho tôi hỏi vivo v25 pro 8gb/128gb hiện tại giá bao nhiêu?', 'expected': 'vivo v25 pro 8gb/128gb'}, {'id': '666baeb89793e149fe739481', 'query': 'Cho tôi hỏi samsung galaxy z fold4 hiện tại giá bao nhiêu?', 'expected': 'samsung galaxy z fold4'}, {'id': '666baeb89793e149fe73949e', 'query': 'Cho tôi hỏi điện thoại redmi note 11 pro (8gb/128gb) hiện tại giá bao nhiêu?', 'expected': 'điện thoại redmi note 11 pro (8gb/128gb)'}, {'id': '666baeb99793e149fe7394e7', 'query': 'Cho tôi hỏi điện thoại realme c3 hiện tại giá bao nhiêu?', 'expected': 'điện thoại realme c3'}]}
```