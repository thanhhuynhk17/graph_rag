import os
from dotenv import load_dotenv
load_dotenv()

from fastmcp import FastMCP
from fastapi import FastAPI

from neo4j import GraphDatabase
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
# from underthesea import word_tokenize, pos_tag
from datetime import datetime
import uuid, json
from typing import ClassVar, Annotated, Literal, Dict, Optional, Any, List, Type
import re
import pandas as pd # test
from contextlib import asynccontextmanager

from src.models.order_manager import OrderManager
from src.utils.hybridsearch import run_hybrid_search
from src.utils.schemas import (
    SearchTypeCategory
)
# ---------------- CONFIG ----------------
from neomodel import config
# Configure neomodel to use neomodel driver
config.DATABASE_URL = f"bolt://{os.getenv('NEO4J_USER')}:{os.getenv('NEO4J_PASSWORD')}@{os.getenv('NEO4J_URI').replace('bolt://','')}/{os.getenv('NEO4J_DATABASE')}"
# Ensure all DateTimes are provided with a timezone before being serialised to UTC epoch
config.FORCE_TIMEZONE = True

# Configure logging based on environment variable
import logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()  # Default to INFO if not set
logging_levels = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL
}
logging.basicConfig(
    level=logging_levels.get(LOG_LEVEL, logging.INFO),  # Fallback to INFO if invalid
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logger.info("Logging configured with level: %s", LOG_LEVEL)

# ---------------- Lifespan ----------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up FastMCP server lifespan...")

    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI"),
        auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
    )

    # Initialize OrderManager instance for managing orders in Neo4j
    order_manager = OrderManager(driver)
    
    # Attach both driver and order_manager to the fast api state
    app.state.driver = driver
    app.state.order_manager = order_manager
    
    try:
        yield
    finally:
        # Shutdown: Close the driver connection
        logger.info("Shutting down FastMCP server lifespan...")
        await driver.close()
        logger.info("Driver connection closed.")

# ---------------- FastMCP server ----------------
mcp = FastMCP(
    name="RestaurantMCP",
    version="1.0.0"
)

# ---------------- Models ----------------
from src.utils.schemas import SearchReq, DishReq, PriceReq, WipeReq, FeedbackReq
# ---------------- Tools ----------------
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
        is_bm25_enable=True
    )
    return results

@mcp.tool(
    name="multi_filter_category",
    description=(
        "Chọn 1 hoặc nhiều nhóm món cùng lúc, mỗi nhóm có thể kèm 1 từ khóa trong tên món ăn. "
        "Cú pháp: <nhóm món> [<từ khóa>], ... "
        "Ví dụ: "
            "- món cá <>"
            "- món cá <basa>"
            "- món cá <>, món lẩu <lóc>, món rau <> "
            "- món cá <không cay>, món lẩu <không cay>, món rau <không cay> "
        )
)

def multi_filter_category(categories: List[str], keywords: List[str]) -> str:
   
    if len(categories) != len(keywords):
        return "Số lượng categories và keywords phải bằng nhau."

    VALID_TYPES = [
        "món cá", "món khai vị", "món ăn chơi", "món rau", "món gỏi",
        "món gà, vịt & trứng", "món tôm & mực", "món xào", "nước mát nhà làm",
        "lẩu", "món thịt", "món sườn & đậu hũ", "món canh", "các loại khô", "tráng miệng"
    ]
    
    if any(cat not in VALID_TYPES for cat in categories):
        return f"category must be one of: {', '.join(VALID_TYPES)}"
    
    final_blocks: List[str] = []

    for cat, kw in zip(categories, keywords):
        type_norm = cat.strip().lower()
        kw_norm   = kw.strip().lower()
        kw_regex  = f"(?i).*{re.escape(kw_norm)}.*" if kw_norm else None

        cypher = """
        MATCH (c:Chunk)
        WHERE toLower(c.type_of_food) = $type_norm
          AND ($kw_regex IS NULL OR
               any(f IN [c._name, c.main_ingredients, c.how_to_prepare,
                         c.taste, c.outstanding_fragrance] WHERE toLower(f) =~ $kw_regex))
        RETURN DISTINCT c._id   AS id,
                        c._name AS name,
                        c.text  AS description,
                        c.number_of_people_eating AS serves,
                        c.current_price AS price
        ORDER BY c._id
        """
        records, _, _ = app.state.driver.execute_query(
            cypher,
            type_norm=type_norm,
            kw_regex=kw_regex
        )

        if not records:
            final_blocks.append(
                f"[ EMPTY ] Không tìm thấy món nào với loại {cat} với yêu cầu '{kw}'"
            )
            continue

        lines = [f"Loại {cat} với yêu cầu '{kw}' - có {len(records)} món:"]
        for r in records:
            lines.append(
                f"  - ID: {r['id']}\n"
                f"    + tên: {r['name']}\n"
                f"    + giá: {r['price']} vnd\n"
                f"    + mô tả: {r['description']}\n"
                f"    + dành cho: {r['serves']}"
            )
        final_blocks.append("\n\n".join(lines))

    return "\n\n".join(final_blocks) if final_blocks else "Không tìm thấy món nào."

@mcp.tool(
    name="menu_value_count_and_price",
    description = (
        """Trả lời khách tất cả các món ăn hiện có trong menu và giá tiền, phân loại theo từng loại món"""
    )
)
def menu_value_count_and_price() -> str:
    cypher = """
    MATCH (c:Chunk)
    WHERE c.type_of_food IS NOT NULL
    WITH c.type_of_food AS type,
         collect(DISTINCT "\n+ [" + c._id + "] " + c._name + " - " + replace(c.current_price, ',000 vnd', 'k')) AS foods
    WITH type, size(foods) AS count, foods
    RETURN type, count, foods
    ORDER BY count DESC
    """
    
    records, _, _ = app.state.driver.execute_query(cypher)

    if not records:
        return "Hiện không có dữ liệu món ăn."

    lines = []
    for row in records:
        t = row["type"].lower().replace("món ", "")
        list_price_food = f' '.join(row['foods']).replace(',000 vnd', 'k')

        lines.append(f"\n- TYPE 'món {t}' ALL {row['count']} ITEMS: \n{list_price_food}")

    return "Menu hiện tại gồm:\n\n" + "\n".join(lines)

@mcp.tool(
    name="menu_value_count_just_name",
    description = (
        '''Trả lời khách tất cả các món ăn hiện có trong menu, phân loại theo từng loại món'''
    )
)
def menu_value_count_just_name() -> str:
    """
    Thống kê số lượng món theo type_of_food và liệt kê TẤT CẢ các món thuộc từng loại.
    """
    cypher = f"""
    MATCH (c:Chunk)
    WHERE c.type_of_food IS NOT NULL
    WITH c.type_of_food AS type, collect(DISTINCT "[" + c._id + "] " + c._name) AS foods
    WITH type, size(foods) AS count, foods
    RETURN type, count, foods
    ORDER BY count DESC
    """
    
    records, _, _ = app.state.driver.execute_query(cypher)

    if not records:
        return "Hiện không có dữ liệu món ăn."

    lines = []
    for row in records:
        t = row["type"].lower().replace("món ", "")
        list_price_food = f', '.join(row['foods'])

        lines.append(f"\n- TYPE 'món {t}' ALL {row['count']} ITEMS: \n{list_price_food}")

    return "Menu hiện tại gồm:\n\n" + "\n".join(lines)

@mcp.tool(
    name="hello_world",
    description = (
    "Dùng tool này để trả lời mọi câu hỏi của khách hàng. "
    "Nếu khách hàng chào hỏi, hỏi menu, hoặc hỏi bất kỳ điều gì không liên quan, "
    "hãy sử dụng tool này để phản hồi bằng cách chào hỏi và liệt kê toàn bộ các món ăn hiện có trong menu, "
    "liệt kê từng món theo từng loại."
)
)
def hello_and_show_menu() -> str:
    cypher = f"""
    MATCH (c:Chunk)
    WHERE c.type_of_food IS NOT NULL
    WITH c.type_of_food AS type, collect(DISTINCT c._name) AS foods
    WITH type, size(foods) AS count, foods
    RETURN type, count, foods
    ORDER BY count DESC
    """
    
    records, _, _ = app.state.driver.execute_query(cypher)

    if not records:
        return "Hiện không có dữ liệu món ăn."

    lines = []
    for row in records:
        t = row["type"].lower().replace("món ", "")
        list_price_food = f', '.join(row['foods'])

        lines.append(f"\n- TYPE 'món {t}' ALL {row['count']} ITEMS: \n{list_price_food}")

    return "Menu hiện tại gồm:\n\n" + "\n".join(lines)

@mcp.tool(
    name="search_food_price",
    description=(
        "Dùng tool này để tìm giá các món ăn theo ID. "
        "Có thể nhập một ID hoặc nhiều ID cùng lúc. "
        "Kết quả trả về gồm tên món và giá của từng ID."
    )
)
def search_food_price(food_ids: list[str]) -> str:
    cypher = f"""
    MATCH (c:Chunk)
    WHERE c._id IN $ids
    RETURN c._id AS id, c._name AS name, c.current_price AS price
    """
    records, _, _ = app.state.driver.execute_query(cypher, ids=food_ids)

    if not records:
        return "Không tìm thấy món ăn nào với ID đã nhập."

    lines = []
    for row in records:
        price_str = f"{row['price']} VND" if row.get("price") else "Chưa có giá"
        lines.append(f"- ID {row['id']}: {row['name']} — {price_str}")

    return "Kết quả tìm kiếm giá món ăn:\n" + "\n".join(lines)

# @mcp.tool(name="goi_y_mua_kem", description="Gợi ý món thường được mua kèm (market-basket)")
# @mcp.tool(name="feedback_bill", description="Khách phản hồi → chuỗi noun-phrase")

# ---------------- HTTP app ----------------
# Create ASGI app
mcp_app = mcp.http_app(
    path='/mcp'
)


@asynccontextmanager
async def combined_lifespan(app: FastAPI):
    # Nest FastMCP lifespan inside app lifespan for proper order
    async with lifespan(app):
        async with mcp_app.lifespan(app):  # Handles MCP-specific startup/shutdown
            yield

# Pass lifespan to FastAPI
app = FastAPI(lifespan=combined_lifespan)

# Mount the MCP server
app.mount("/", mcp_app)
