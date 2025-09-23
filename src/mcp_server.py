from fastmcp import FastMCP
from langchain_mcp_adapters.tools import to_fastmcp
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel, Field
from typing import Annotated, Literal
from dotenv import load_dotenv
from underthesea import word_tokenize, pos_tag
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

@mcp.tool(name="hybrid_search", description="Tìm món ăn theo tên, tag hoặc ingredient")
def hybrid_search(req: SearchReq):
    cypher = """
    MATCH (d:Dish)
    WHERE toLower(d.type_of_food) = toLower($q)
        OR toLower(d.name_of_food) = toLower($q)
        OR toLower(d.how_to_prepare) = toLower($q)
        OR toLower(d.main_ingredients) = toLower($q)
        OR toLower(d.taste) = toLower($q)
        OR toLower(d.outstanding_fragrance) = toLower($q)
        OR toLower(d.number_of_people_eating) = toLower($q)

        OR toLower(d.type_of_food) CONTAINS toLower($q)
        OR toLower(d.name_of_food) CONTAINS toLower($q)
        OR toLower(d.how_to_prepare) CONTAINS toLower($q)
        OR toLower(d.main_ingredients) CONTAINS toLower($q)
        OR toLower(d.taste) CONTAINS toLower($q)
        OR toLower(d.outstanding_fragrance) CONTAINS toLower($q)
        OR toLower(d.number_of_people_eating) CONTAINS toLower($q)

        OR EXISTS {
            MATCH (d)-[:HAS_TAG]->(t:Tag)
            WHERE toLower(t.tag) = toLower($q)
                OR toLower(t.tag) CONTAINS toLower($q)
        }
        OR EXISTS {
            MATCH (d)-[:HAS_INGREDIENT]->(i:Ingredient)
            WHERE toLower(i.name) = toLower($q)
                OR toLower(i.name) CONTAINS toLower($q)
        }
    RETURN apoc.text.join([
            toString(d.dish_id),
            toString(d.type_of_food),
            toString(d.name_of_food),
            toString(d.how_to_prepare),
            toString(d.main_ingredients),
            toString(d.taste),
            toString(d.outstanding_fragrance),
            toString(d.current_price),
            toString(d.number_of_people_eating)
        ], ', ') AS line
    LIMIT $k;
    """
    rec, _, _ = driver.execute_query(cypher, q=req.query, k=req.k)
    return [r.data() for r in rec]

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

@mcp.tool(name="tim_theo_gia", description="Tìm món ăn dưới mức giá tối đa")
def tim_theo_gia(req: PriceReq):
    cypher = """
    MATCH (d:Dish)
    WHERE d.current_price <= $max
    RETURN d.dish_id AS id, d.name_of_food AS ten, d.current_price
    ORDER BY d.current_price
    """
    rec, _, _ = driver.execute_query(cypher, max=req.max_price)
    return [r.data() for r in rec]

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
