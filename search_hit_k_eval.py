import random
import pandas as pd

# --------------------------
# Prompt templates
# --------------------------
prompts = [
    "Tôi muốn mua {product_name}, bên bạn còn hàng không?",
    "Cho tôi hỏi {product_name} hiện tại giá bao nhiêu?",
    "Tôi đang quan tâm đến {product_name}, bạn có thể tư vấn thêm cho tôi không?",
    "{product_name} có màu nào và dung lượng bao nhiêu GB vậy?",
    "Tôi muốn đặt mua {product_name}, cần làm thế nào?"
]

def get_random_prompt(product_name: str) -> str:
    return random.choice(prompts).format(product_name=product_name)

# --------------------------
# Stub search (replace later)
# --------------------------
from src.utils.hybridsearch import run_hybrid_search

def hybrid_search(prompt: str, k: int):
    """
    Trả về list top-k kết quả (dict với _id).
    Bạn sẽ thay bằng search thực tế.
    """
    results = run_hybrid_search(prompt, k, is_bm25_enable=False)
    return [{"_id": r.metadata["_id"]} for r in results]

# --------------------------
# Benchmark for dataframe
# --------------------------
import time

def benchmark_df(df: pd.DataFrame, ks=(1, 5, 20)):
    results = {f"hit@{k}": 0 for k in ks}
    results["missed"] = []   # danh sách lưu id/query bị miss
    
    total = len(df)
    
    for _, row in df.iterrows():
        ground_truth_id = row["_id"]
        product_name = row.get("product_name") or row["title"].split("-")[0].strip()
        
        query = get_random_prompt(product_name)
        print(f"[TEST] id={ground_truth_id} | query='{query}'")
        # đo thời gian bắt đầu
        start = time.perf_counter()
        
        hit_any = False
        max_k = max(ks)
        # hybrid_search max_k, then select k_i for saving tokens
        # only use with reranker
        # run hybrid_search() on for-loop if benchmark with embedding only
        retrieved = hybrid_search(query, max_k)
        for k in ks:
            retrieved_k = retrieved[:k]
            if any(r["_id"] == ground_truth_id for r in retrieved_k):
                results[f"hit@{k}"] += 1
                hit_any = True
        
        elapsed = time.perf_counter() - start
        print(f"[TIME] query took {elapsed:.3f}s")

        if not hit_any:
            results["missed"].append({
                "id": ground_truth_id,
                "query": query,
                "expected": product_name
            })

    # normalize hit rates
    for k in ks:
        results[f"hit@{k}"] /= total

    return results



# --------------------------
# Demo
# --------------------------
if __name__ == "__main__":
    # CSV thực tế
    df = pd.read_csv("src/data/hoanghamobile_with_summary.csv")

    scores = benchmark_df(df)
    print("\nFinal scores:", scores)
