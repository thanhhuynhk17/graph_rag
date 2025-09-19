# 🚀 SUPER NOVA RAG 🚀

## Overview

Hybrid Retrieval-Augmented Generation (RAG) pipeline.

## Pipeline

1. **Chunking by row (*.csv)**
2. **Ensemble: BM25 + Embedding (Qwen3-0.6B)**
3. **Reranker**
    - Qwen3-0.6B-Reranker: broken
    - BAAI/bge-reranker-v2-m3

> 🔹 BM25 removed in benchmark to measure *pure embedding performance*.  

## Benchmark: embedding only (no rerank)

```bash
Final scores:
{'hit@1': 0.653125, 'hit@5': 0.9, 'hit@10': 0.940625, 'missed': [{'id': '666baeb59793e149fe7393e5', 'query': 'Tôi muốn mua điện thoại ai, bên bạn còn hàng không?', 'expected': 'điện thoại ai'}, {'id': '666baeb59793e149fe7393e3', 'query': 'Tôi muốn mua điện thoại ai, bên bạn còn hàng không?', 'expected': 'điện thoại ai'}, {'id': '666baeb59793e149fe7393e4', 'query': 'Tôi muốn mua điện thoại ai, bên bạn còn hàng không?', 'expected': 'điện thoại ai'}, {'id': '666baeb59793e149fe7393e2', 'query': 'Tôi đang quan tâm đến điện thoại ai, bạn có thể tư vấn thêm cho tôi không?', 'expected': 'điện thoại ai'}, {'id': '666baeb59793e149fe7393ec', 'query': 'Tôi muốn mua điện thoại ai, bên bạn còn hàng không?', 'expected': 'điện thoại ai'}, {'id': '666baeb59793e149fe7393ed', 'query': 'Tôi đang quan tâm đến điện thoại ai, bạn có thể tư vấn thêm cho tôi không?', 'expected': 'điện thoại ai'}, {'id': '666baeb59793e149fe7393f2', 'query': 'điện thoại zte blade v50 design (8gb/256gb) có màu nào và dung lượng bao nhiêu GB vậy?', 'expected': 'điện thoại zte blade v50 design (8gb/256gb)'}, {'id': '666baeb89793e149fe7394a2', 'query': 'Tôi muốn đặt mua điện thoại điện thoại di động xor x2 prime gold, cần làm thế nào?', 'expected': 'điện thoại điện thoại di động xor x2 prime gold'}, {'id': '666baeb89793e149fe7394a5', 'query': 'Cho tôi hỏi điện thoại oppo a16k 3gb/32gb hiện tại giá bao nhiêu?', 'expected': 'điện thoại oppo a16k 3gb/32gb'}, {'id': '666baeb89793e149fe7394b4', 'query': 'điện thoại samsung galaxy a02s 4gb/64gb có màu nào và dung lượng bao nhiêu GB vậy?', 'expected': 'điện thoại samsung galaxy a02s 4gb/64gb'}, {'id': '666baeb89793e149fe7394b0', 'query': 'Cho tôi hỏi oppo a54 hiện tại giá bao nhiêu?', 'expected': 'oppo a54'}, {'id': '666baeb99793e149fe7394d3', 'query': 'Cho tôi hỏi oppo reno4 pro hiện tại giá bao nhiêu?', 'expected': 'oppo reno4 pro'}, {'id': '666baeb99793e149fe7394cf', 'query': 'Tôi muốn mua điện thoại samsung galaxy z fold2 5g, bên bạn còn hàng không?', 'expected': 'điện thoại samsung galaxy z fold2 5g'}, {'id': '666baeb99793e149fe7394d1', 'query': 'Cho tôi hỏi điện thoại energizer e241s hiện tại giá bao nhiêu?', 'expected': 'điện thoại energizer e241s'}, {'id': '666baeb99793e149fe7394d4', 'query': 'Tôi muốn mua oppo reno4, bên bạn còn hàng không?', 'expected': 'oppo reno4'}, {'id': '666baeb99793e149fe7394dd', 'query': 'Tôi muốn mua điện thoại samsung galaxy s20 ultra, bên bạn còn hàng không?', 'expected': 'điện thoại samsung galaxy s20 ultra'}, {'id': '666baeb99793e149fe7394de', 'query': 'Cho tôi hỏi điện thoại samsung galaxy s20 hiện tại giá bao nhiêu?', 'expected': 'điện thoại samsung galaxy s20'}, {'id': '666baeb99793e149fe7394d7', 'query': 'Cho tôi hỏi oppo a53 hiện tại giá bao nhiêu?', 'expected': 'oppo a53'}, {'id': '666baeb99793e149fe7394e0', 'query': 'Cho tôi hỏi đồng hồ thông minh samsung galaxy fit e (sm hiện tại giá bao nhiêu?', 'expected': 'đồng hồ thông minh samsung galaxy fit e (sm'}]}
```

## Benchmark: top 50 embedding -> top (1,5,10) reranking (bge-reranker-v2-m3)

- **Dataset**: 320 records (src\data\hoanghamobile_with_summary.csv)
- **Mode**: Embedding-only retrieval
- **Metric**: `hit@k` (1, 5, 10)

```bash
Final scores:
{'hit@1': 0.85625, 'hit@5': 0.978125, 'hit@10': 0.978125, 'missed': [{'id': '666baeb59793e149fe7393e5', 'query': 'Cho tôi hỏi điện thoại ai hiện tại giá bao nhiêu?', 'expected': 'điện thoại ai'}, {'id': '666baeb59793e149fe7393e4', 'query': 'Cho tôi hỏi điện thoại ai hiện tại giá bao nhiêu?', 'expected': 'điện thoại ai'}, {'id': '666baeb59793e149fe7393e2', 'query': 'Tôi muốn mua điện thoại ai, bên bạn còn hàng không?', 'expected': 'điện thoại ai'}, {'id': '666baeb59793e149fe7393ec', 'query': 'Cho tôi hỏi điện thoại ai hiện tại giá bao nhiêu?', 'expected': 'điện thoại ai'}, {'id': '666baeb79793e149fe739457', 'query': 'Tôi muốn đặt mua điện thoại samsung galaxy s23, cần làm thế nào?', 'expected': 'điện thoại samsung galaxy s23'}, {'id': '666baeb79793e149fe739462', 'query': 'Cho tôi hỏi vivo v25 pro 8gb/128gb hiện tại giá bao nhiêu?', 'expected': 'vivo v25 pro 8gb/128gb'}, {'id': '666baeb99793e149fe7394d8', 'query': 'Cho tôi hỏi oppo a12 hiện tại giá bao nhiêu?', 'expected': 'oppo a12'}]}
```

## Benchmark: top 100 embedding -> top (1,5,20) reranking (bge-reranker-v2-m3)

```bash
Final scores: 
{'hit@1': 0.88125, 'hit@5': 0.984375, 'hit@20': 0.984375, 'missed': [{'id': '666baeb79793e149fe739457', 'query': 'Tôi muốn mua điện thoại samsung galaxy s23, bên bạn còn hàng không?', 'expected': 'điện thoại samsung galaxy s23'}, {'id': '666baeb79793e149fe739462', 'query': 'Cho tôi hỏi vivo v25 pro 8gb/128gb hiện tại giá bao nhiêu?', 'expected': 'vivo v25 pro 8gb/128gb'}, {'id': '666baeb89793e149fe739481', 'query': 'Cho tôi hỏi samsung galaxy z fold4 hiện tại giá bao nhiêu?', 'expected': 'samsung galaxy z fold4'}, {'id': '666baeb89793e149fe73949e', 'query': 'Cho tôi hỏi điện thoại redmi note 11 pro (8gb/128gb) hiện tại giá bao nhiêu?', 'expected': 'điện thoại redmi note 11 pro (8gb/128gb)'}, {'id': '666baeb99793e149fe7394e7', 'query': 'Cho tôi hỏi điện thoại realme c3 hiện tại giá bao nhiêu?', 'expected': 'điện thoại realme c3'}]}
```
