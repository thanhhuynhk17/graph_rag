from langchain_core.tools import tool
from langchain_mcp_adapters.tools import to_fastmcp
from fastmcp import FastMCP
from src.utils.hybridsearch import run_hybrid_search, HybridSearchQuery
from typing import Annotated, Literal
from pydantic import Field

# Convert to FastMCP
# fastmcp_hybrid_search = to_fastmcp(hybrid_search)

# Create FastMCP server
mcp = FastMCP(
    name="HybridSearchServer"
)

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

# Create ASGI app
app = mcp.http_app()