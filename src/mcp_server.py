from fastmcp import FastMCP
from langchain_mcp_adapters.tools import to_fastmcp
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel, Field
from typing import Annotated, Literal
from dotenv import load_dotenv
from underthesea import word_tokenize, pos_tag
from src.utils.hybridsearch import run_hybrid_search
from datetime import datetime
import os, uuid, json
from typing import List

# ---------------- CONFIG ----------------
load_dotenv()

def load_env(key: str, default: str = ""):
    return os.getenv(key, default)

NEO4J_URI  = load_env("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = load_env("NEO4J_USER", "neo4j")
NEO4J_PASS = load_env("NEO4J_PASSWORD", "12345678")
EMB_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
emb    = SentenceTransformer(EMB_MODEL)

def tao_embedding(text: str):
    return emb.encode(text, normalize_embeddings=True).tolist()

# ---------------- FastMCP server ----------------
mcp = FastMCP(name="RestaurantMCP")

# ---------------- Init Neo4j (com_que) ----------------
from src.utils.helpers import init_data_neo4j

# init_data_neo4j(driver=driver)

# ---------------- Models ----------------
class SearchReq(BaseModel):
    query: str = Field(..., min_length=3, description="Tên món ăn, tag hoặc ingredient để tìm kiếm")
    k: int = Field(10, description="Số lượng kết quả muốn lấy")

class DishReq(BaseModel):
    dish_id: str

class PriceReq(BaseModel):
    max_price: int
    
class FeedbackReq(BaseModel):
    customer_id: str
    bill_id: str
    dish_id: str
    text: str
# ---------------- Tools ----------------

@mcp.tool(name="init_database", description="...")
async def init_database():
    init_data_neo4j(driver=driver)
    return "✅ Đã khởi tạo dữ liệu mẫu vào Neo4j"

@mcp.tool(
    name="hybrid_search",
    description=(
    "Retrieve restaurant and dish info from the knowledge base.\n"
    "Supports:\n"
    "- Restaurant details: name, location, cuisine, opening hours\n"
    "- Dish details: name, ingredients, dietary info, cooking style\n"
    "Increasing 'k' allows fetching more candidate results for better coverage."
    )
)
async def hybrid_search(
    # Search query with minimum length
    query: Annotated[str, Field(min_length=3, max_length=300, description="Restaurant or dish name, location, or type")],

    # Number of top results to fetch
    k: Annotated[Literal[30, 50], Field(description="Number of top results to retrieve: 30=fast, 50=broad")] = 30,

    # Optional filter: cuisine type
    cuisine: Annotated[str, Field(max_length=50, description="Optional cuisine type to filter results")] = "",
    
):
    if not query:
        raise ValueError("Query must not be None")

    results = run_hybrid_search(
        query=query,
        k=k,
        is_bm25_enable=False
    )
    return results

@mcp.tool(name="search_database", description="Tìm món ăn theo tên, tag hoặc ingredient")
def search_database(req: SearchReq):
    # Build full-text query string in Python
    phrase = req.query.strip()
    words  = phrase.split()
    phrase_part = f'({phrase})^3'          # boost cao nhất cho cụm nguyên
    term_part   = f'({" OR ".join(words)})^1'  # mọi từ riêng
    query = f'{phrase_part} OR {term_part}'

    cypher = """
        CALL db.index.fulltext.queryNodes('keyword', $query) YIELD node, score
        RETURN node._id AS id,
            node.name_of_food AS name_food, score
        ORDER BY score DESC
        LIMIT $k
        """
    rec, _, _ = driver.execute_query(cypher, query=query, k=req.k)
    
    if len(rec) == 0:
        return "[EMPTY RESULT] This keyword/keyphrase is not found in neo4j"
    
    return [dict(r) for r in rec]

@mcp.tool(name="xem_schema", description="Xem schema graph hiện tại")
def xem_schema():
    rec, _, _ = driver.execute_query("CALL db.schema.visualization()")
    if not rec:
        return {}
    data = rec[0]["nodes"], rec[0]["relationships"]
    return {
        "nodes": [
            {"id": n.id, "labels": list(n.labels), **dict(n)} for n in rec[0]["nodes"]
        ],
        "relationships": [
            {"id": r.id, "type": r.type, "start": r.start_node.id, "end": r.end_node.id, **dict(r)} for r in rec[0]["relationships"]
        ]
    }
    
def normalize_price(p) -> int:
    try:
        import pandas as pd
        return int(pd.Series([p])
                     .astype(str)
                     .str.replace(r'[,. ]', '', regex=True)
                     .str.extract(r'(\d+)')[0]
                     .astype(float)
                     .fillna(0)
                     .iloc[0])
    except (ValueError, TypeError):
        raise ValueError("Giá nhập không hợp lệ")

@mcp.tool(name="tim_theo_gia", description="Tìm món ≤ giá tối đã")
def tim_theo_gia(req: PriceReq):

    max_int = normalize_price(req.max_price)
    cypher = """
    MATCH (c:Chunk)
    WHERE toInteger(replace(c.current_price, ',', '')) <= $max_int
    RETURN c._id           AS id,
           c.name_of_food   AS ten,
           c.current_price  AS current_price
    ORDER BY toInteger(replace(c.current_price, ',', '')) DESC
    """
    rec, _, _ = driver.execute_query(cypher, max_int=max_int)
    return [dict(r) for r in rec]

# @mcp.tool(name="goi_y_mua_kem", description="Gợi ý món thường được mua kèm (market-basket)")
# def goi_y_mua_kem(req: DishReq):
#     cypher = """
#     MATCH (d:Dish {dish_id:$id})-[:ALSO_BOUGHT]->(g:Dish)
#     RETURN g.dish_id AS id, g.name_of_food AS ten, g.current_price AS gia
#     ORDER BY gia LIMIT 5
#     """
#     rec, _, _ = driver.execute_query(cypher, id=req.dish_id)
#     return [r.data() for r in rec]
from underthesea import word_tokenize

def extract_phrases(text: str) -> List[str]:
    """
    "Món cơm chiên ăn khá ngon nhưng đợi lâu"
    → ['Món cơm chiên', 'ăn', 'khá ngon', 'nhưng', 'đợi lâu']
    """
    tokens = pos_tag(text)
    phrases, cur = [], []

    for w, p in tokens:
        if p in {'N', 'Np', 'Ny', 'V', 'A', 'R', 'C'}:
            cur.append(w)
        else:
            if cur:
                phrases.append(' '.join(cur))
                cur = []
    if cur:
        phrases.append(' '.join(cur))
    return phrases

@mcp.tool(name="feedback_bill", description="Khách phản hồi → chuỗi noun-phrase")
def feedback_bill(req: FeedbackReq):
    phrases = extract_phrases(req.text)
    if not phrases:                      # fallback
        phrases = [req.text[:50]]

    fid = str(uuid.uuid4())

    cypher = """
    MATCH (c:Customer {customer_id: $cust})
    MATCH (b:Bill  {bill_id: $bill})
    MATCH (d:Dish  {dish_id: $dish})

    CREATE (f:Feedback {feedback_id: $fid, created_at: datetime()})

    MERGE (c)-[:FEEDBACK_BILL]->(b)
    MERGE (f)-[:ABOUT]->(b)
    MERGE (f)-[:WRONG_DISH {value: 1}]->(d)

    // ---- tạo chuỗi phrase ----
    WITH f, $phrases AS phrases
    UNWIND range(0, size(phrases)-1) AS idx
        MERGE (p:Phrase {text: phrases[idx]})
        WITH f, phrases, p, idx
        ORDER BY idx
        WITH f, collect(p) AS nodes

    // cặp liền kề (n1,n2) trên mỗi row
    WITH f, nodes, range(0, size(nodes)-2) AS idxs
    UNWIND idxs AS i
        WITH f, nodes, nodes[i] AS n1, nodes[i+1] AS n2
        MERGE (n1)-[:NEXT]->(n2)
        WITH f, nodes   // 👈 giữ lại nodes để dùng tiếp

    // nối Feedback với phrase đầu
    WITH f, nodes[0] AS firstP, nodes
    MERGE (f)-[:STARTS_WITH]->(firstP)

    RETURN f.feedback_id AS feedback_id,
        firstP.text   AS first_phrase,
        nodes[-1].text AS last_phrase
    """

    rec, _, _ = driver.execute_query(
        cypher,
        cust=req.customer_id,
        bill=req.bill_id,
        dish=req.dish_id,
        fid=fid,
        phrases=phrases
    )
    return [r.data() for r in rec]

# ---------------- HTTP app ----------------
app = mcp.http_app()
