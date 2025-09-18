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
    results = run_hybrid_search(prompt, k)
    return [{"_id": r.metadata["_id"]} for r in results]

# --------------------------
# Benchmark for dataframe
# --------------------------
def benchmark_df(df: pd.DataFrame, ks=(1, 5, 10)):
    results = {f"hit@{k}": 0 for k in ks}
    results["missed"] = []   # danh sách lưu id/query bị miss
    
    total = len(df)
    
    for _, row in df.iterrows():
        ground_truth_id = row["_id"]
        product_name = row.get("product_name") or row["title"].split("-")[0].strip()
        
        query = get_random_prompt(product_name)
        print(f"[TEST] id={ground_truth_id} | query='{query}'")

        hit_any = False
        for k in ks:
            retrieved = hybrid_search(query, k)
            if any(r["_id"] == ground_truth_id for r in retrieved):
                results[f"hit@{k}"] += 1
                hit_any = True
        
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
