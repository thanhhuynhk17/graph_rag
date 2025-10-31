import os
from dotenv import load_dotenv
load_dotenv()

from fastmcp import FastMCP
from fastmcp.tools.tool import ToolResult, TextContent
from fastmcp.server.dependencies import get_context

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from neo4j import GraphDatabase
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from datetime import datetime
from typing import ClassVar, Annotated, Literal, Dict, Optional, Any, List, Type
from contextlib import asynccontextmanager
import re
import json
import uuid
import pendulum

from src.models.order_manager import OrderManager
from src.utils.hybridsearch import run_hybrid_search
from src.utils.order_service import OrderService
from src.utils.schemas import (
    SearchTypeCategory, SearchReq, DishReq, PriceReq, WipeReq, FeedbackReq,
    OrderRequest, OrderResponse, OrderedDish, OrderData, CustomerData,
    UserOrdersResponse
)
from src.models.order_graph import Dish, Customer, Order
from src.models.config import RestaurantConfig
from neomodel import db

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


def clear_database_schema() -> bool:
    """
    Clear all database constraints, indexes, and data.
    Returns True if successful, False if failed.
    """
    try:
        logger.info("Starting database schema cleanup...")

        # Step 1: Drop all constraints and indexes using APOC
        try:
            # Drop all property uniqueness constraints
            db.cypher_query("CALL apoc.schema.assert({}, {})")
            logger.info("✓ Dropped all constraints and indexes")
        except Exception as e:
            logger.warning(f"Could not drop constraints via APOC: {str(e)}")
            # Fallback: Try to drop specific constraints manually
            try:
                # Get all constraints
                constraints_result = db.cypher_query("SHOW CONSTRAINTS")
                if constraints_result and constraints_result[0]:
                    for constraint in constraints_result[0]:
                        if 'name' in constraint:
                            constraint_name = constraint['name']
                            try:
                                db.cypher_query(f"DROP CONSTRAINT {constraint_name}")
                                logger.info(f"Dropped constraint: {constraint_name}")
                            except Exception as ce:
                                logger.warning(f"Could not drop constraint {constraint_name}: {str(ce)}")

                logger.info("✓ Attempted to drop constraints manually")
            except Exception as e2:
                logger.warning(f"Could not drop constraints manually either: {str(e2)}")

        # Step 2: Delete all nodes and relationships
        try:
            # Delete all relationships first
            db.cypher_query("MATCH ()-[r]-() DELETE r")
            logger.info("✓ Deleted all relationships")

            # Delete all nodes
            db.cypher_query("MATCH (n) DELETE n")
            logger.info("✓ Deleted all nodes")

        except Exception as e:
            logger.error(f"Failed to delete data: {str(e)}")
            return False

        # Step 3: Verify database is clean
        try:
            node_count = db.cypher_query("MATCH (n) RETURN count(n) as count")[0][0]['count']
            rel_count = db.cypher_query("MATCH ()-[r]-() RETURN count(r) as count")[0][0]['count']

            if node_count == 0 and rel_count == 0:
                logger.info("✓ Database confirmed clean - 0 nodes, 0 relationships")
                return True
            else:
                logger.warning(f"⚠ Database not fully cleaned - {node_count} nodes, {rel_count} relationships remain")
                return False

        except Exception as e:
            logger.warning(f"Could not verify database cleanup: {str(e)}")
            return True  # Assume success if we can't verify

    except Exception as e:
        logger.error(f"Critical error during database cleanup: {str(e)}")
        return False

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

    # Auto-load dishes if enabled
    auto_load_dishes = os.getenv("AUTO_LOAD_DISHES", "false").lower() == "true"
    if auto_load_dishes:
        logger.info("AUTO_LOAD_DISHES enabled, checking database state...")

        # Step 1: Check if dishes already exist in database
        existing_dish_count = len(Dish.nodes)
        if existing_dish_count > 0:
            force_refresh = os.getenv("FORCE_REFRESH_DISHES", "false").lower() == "true"
            if not force_refresh:
                error_msg = (
                    f"⚠ DATABASE PROTECTION: Found {existing_dish_count} existing dishes in database.\n"
                    f"Auto-loading would clear all data and reload from CSV.\n"
                    f"\n"
                    f"To force refresh the database, use:\n"
                    f"  FORCE_REFRESH_DISHES=true uv run uvicorn src.mcp_server:app --port 8000 --host localhost\n"
                    f"\n"
                    f"Or to skip auto-loading entirely, use:\n"
                    f"  AUTO_LOAD_DISHES=false uv run uvicorn src.mcp_server:app --port 8000 --host localhost\n"
                    f"\n"
                    f"Server startup aborted for data safety."
                )
                logger.error(error_msg)
                raise RuntimeError("Database contains existing dishes. Set FORCE_REFRESH_DISHES=true to proceed with data refresh.")

            logger.warning(f"⚠ FORCE_REFRESH_DISHES=true detected - proceeding with database refresh")
            logger.warning(f"⚠ This will delete all {existing_dish_count} existing dishes and reload from CSV")
        else:
            logger.info("No existing dishes found, proceeding with fresh load")

        # Step 2: Clear database schema and data
        try:
            cleanup_success = clear_database_schema()
            if not cleanup_success:
                logger.warning("Database cleanup failed, but proceeding with dish loading...")
        except Exception as e:
            logger.warning(f"Database cleanup encountered an exception, but proceeding with dish loading: {str(e)}")

        # Step 3: Load dishes from CSV
        try:
            csv_path = "src/data/comque_new_enriched.csv"
            dishes = Dish.load_from_csv(csv_path)
            logger.info(f"✓ Successfully loaded {len(dishes)} dishes from {csv_path}")
        except Exception as e:
            logger.error(f"✗ Failed to load dishes from CSV: {str(e)}")
            logger.warning("Server will continue starting but without dish data")
    else:
        logger.info("AUTO_LOAD_DISHES disabled, skipping dish loading")

    try:
        yield
    finally:
        # Shutdown: Close the driver connection
        logger.info("Shutting down FastMCP server lifespan...")
        app.state.driver.close()
        logger.info("Driver connection closed.")

# ---------------- FastMCP server ----------------
mcp = FastMCP(
    name="RestaurantMCP",
    version="1.0.0"
)

# # ---------------- Tools ----------------
# @mcp.tool(
#     name="hybrid_search",
#     description=(
#     "Retrieve restaurant and dish info from the knowledge base.\n"
#     "Supports:\n"
#     "- Restaurant details: name, location, cuisine, opening hours\n"
#     "- Dish details: name, ingredients, dietary info, cooking style\n"
#     "Increasing 'k' allows fetching more candidate results for better coverage."
#     )
# )
# async def hybrid_search(
#     # Search query with minimum length
#     query: Annotated[str, Field(min_length=3, max_length=300, description="Restaurant or dish name, location, or type")],

#     # Number of top results to fetch
#     k: Annotated[Literal[30, 50], Field(description="Number of top results to retrieve: 30=fast, 50=broad")] = 30,

#     # Optional filter: cuisine type
#     cuisine: Annotated[str, Field(max_length=50, description="Optional cuisine type to filter results")] = "",
    
# ):
#     if not query:
#         raise ValueError("Query must not be None")

#     results = run_hybrid_search(
#         query=query,
#         k=k,
#         is_bm25_enable=True
#     )
#     return results

@mcp.tool(
    name="multi_dish_lookup",
    description="Get detailed information for specific dishes by their dish IDs (e.g., dish1, dish2, dish3)"
)
async def multi_dish_lookup(dish_ids: List[str]) -> ToolResult:
    if not dish_ids:
        return ToolResult(
            content=[TextContent(type="text", text="Không có dish IDs nào được cung cấp.")],
            structured_content={"error": "no_dish_ids_provided", "requested_ids": dish_ids}
        )

    found_dishes = []
    missing_ids = []
    errors = []

    for dish_id in dish_ids:
        dish_id = dish_id.strip()
        try:
            dish = Dish.nodes.get(dish_id=dish_id)
            dish_data = {
                "dish_id": dish.dish_id,
                "name_of_food": dish.name_of_food,
                "type_of_food": dish.type_of_food,
                "current_price": dish.current_price,
                "main_ingredients": dish.main_ingredients,
                "how_to_prepare": dish.how_to_prepare,
                "taste": dish.taste,
                "outstanding_fragrance": dish.outstanding_fragrance,
                "number_of_people_eating": dish.number_of_people_eating,
                "combine_info": dish.combine_info
            }
            found_dishes.append(dish_data)
        except Dish.DoesNotExist:
            missing_ids.append(dish_id)
        except Exception as e:
            errors.append({"dish_id": dish_id, "error": str(e)})

    # Create human-readable summary
    summary_parts = []
    if found_dishes:
        summary_parts.append(f"Tìm thấy {len(found_dishes)} món ăn:")
        for dish in found_dishes:
            summary_parts.append(f"- {dish['dish_id']}: {dish['name_of_food']} - {dish['current_price']:,.0f} vnđ")
    if missing_ids:
        summary_parts.append(f"\nKhông tìm thấy {len(missing_ids)} món: {', '.join(missing_ids)}")
    if errors:
        summary_parts.append(f"\nCó lỗi với {len(errors)} món ăn")

    return ToolResult(
        content=[TextContent(type="text", text="\n".join(summary_parts))],
        structured_content={
            "found_dishes": found_dishes,
            "missing_ids": missing_ids,
            "errors": errors,
            "total_requested": len(dish_ids),
            "total_found": len(found_dishes)
        }
    )

@mcp.tool(
    name="menu_value_count_and_price",
    description = (
        """Trả lời khách tất cả các món ăn hiện có trong menu và giá tiền, phân loại theo từng loại món"""
    )
)
async def menu_value_count_and_price() -> ToolResult:
    from itertools import groupby
    from operator import attrgetter

    dishes = Dish.nodes.all()
    if not dishes:
        return ToolResult(
            content=[TextContent(type="text", text="Hiện không có dữ liệu món ăn.")],
            structured_content={"error": "no_dishes_found", "total_dishes": 0}
        )

    # Sort and group dishes by type
    sorted_dishes = sorted(dishes, key=attrgetter('type_of_food'))
    grouped = {k: list(g) for k, g in groupby(sorted_dishes, key=attrgetter('type_of_food'))}

    # Create structured menu data
    menu_groups = {}
    summary_lines = ["Menu hiện tại gồm:"]

    total_dishes = 0
    for type_of_food, type_dishes in grouped.items():
        t = type_of_food.lower().replace("món ", "")

        # Create dishes list for this group
        dishes_list = []
        dish_lines = []
        for dish in type_dishes:
            dish_data = {
                "dish_id": dish.dish_id,
                "name_of_food": dish.name_of_food,
                "current_price": dish.current_price,
                "type_of_food": dish.type_of_food
            }
            dishes_list.append(dish_data)
            dish_lines.append(f"[{dish.dish_id}] {dish.name_of_food} - {dish.current_price:,.0f}k")

        menu_groups[t] = {
            "type_name": f"món {t}",
            "count": len(type_dishes),
            "dishes": dishes_list
        }

        summary_lines.append(f"- TYPE '{t}' ALL {len(type_dishes)} ITEMS:")
        summary_lines.extend([f"  + {line}" for line in dish_lines])

        total_dishes += len(type_dishes)

    return ToolResult(
        content=[TextContent(type="text", text="\n".join(summary_lines))],
        structured_content={
            "menu_groups": menu_groups,
            "total_dishes": total_dishes,
            "total_groups": len(menu_groups),
            "timestamp": datetime.now().isoformat()
        }
    )

@mcp.tool(
    name="take_order",
    description=(
        "Take a new order from a customer. Creates customer if they don't exist, "
        "validates dishes, calculates pricing, assigns tables, and creates order relationships."
    )
)
async def take_order(
    customer_id: Annotated[str, Field(description="Unique customer identifier")],
    full_name: Annotated[str, Field(description="Customer full name")],
    phone: Annotated[Optional[str], Field(description="Customer phone number")] = None,
    email: Annotated[Optional[str], Field(description="Customer email")] = None,
    dishes: Annotated[List[Dict[str, Any]], Field(description="List of dishes with 'dish_id' and 'quantity'")] = None,
    is_takeaway: Annotated[bool, Field(description="True for takeaway, False for dine-in")] = False,
    arrived_at: Annotated[Optional[str], Field(description="ISO datetime string when customer arrived")] = None,
    notes: Annotated[Optional[str], Field(description="Order notes")] = None
) -> ToolResult:
    """Refactored to use shared OrderService logic."""
    # Initialize order service and delegate
    order_service = OrderService()
    return await order_service.create_mcp_response(
        customer_id, full_name, phone, email, dishes, is_takeaway, arrived_at, notes
    )

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

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # Next.js dev server
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],  # Allow all headers including Content-Type
)

# ---------------- Traditional REST API endpoints ----------------

@app.post(
    "/api/orders",
    response_model=OrderResponse,
    summary="Create a new restaurant order",
    description="Creates customer if they don't exist, validates dishes, calculates pricing, assigns tables, and creates order relationships."
)
async def api_take_order(order_request: OrderRequest) -> OrderResponse:
    """
    Traditional REST API endpoint for taking orders.
    Refactored to use shared OrderService logic to ensure consistency with MCP tool.
    """
    try:
        # Initialize order service and delegate
        order_service = OrderService()
        return await order_service.create_api_response(order_request)

    except Exception as e:
        logger.error(f"API Error taking order: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.get(
    "/api/customers/{customer_id}/orders",
    response_model=UserOrdersResponse,
    summary="Get all orders for a customer",
    description="Retrieve all orders placed by a specific customer including order details and dish information."
)
async def get_user_orders(customer_id: str) -> UserOrdersResponse:
    """
    Get all orders for a customer by customer_id.
    Returns customer info and complete order history with dishes.
    """
    try:
        logger.info(f"Retrieving orders for customer: {customer_id}")

        # Get or create customer - auto-create if doesn't exist
        customer_data_dict = {
            'customer_id': customer_id,
            'full_name': customer_id,  # Use customer_id as default name
            'phone': None,
            'email': None
        }
        customer = Customer.get_or_create(customer_data_dict)

        # Check if this was a newly created customer
        if customer.full_name == customer_id and customer.phone is None:
            logger.info(f"Auto-created new customer: {customer_id}")

        customer_data = CustomerData(
            customer_id=customer.customer_id,
            full_name=customer.full_name,
            phone=customer.phone or None,
            email=customer.email
        )

        # Get all orders for this customer using neomodel ORM relationships
        # Customer.placed.all() returns (OrderNode, PlacedRelationship) tuples
        placed_relationships = customer.placed.all()

        # Process orders into OrderData objects
        orders = []
        for order, placed_rel in placed_relationships:
            # Get all dishes in this order using Order.items.all()
            # Order.items.all() returns (DishNode, ContainsRelationship) tuples
            items_relationships = order.items.all()

            # Build ordered dishes data from Contains relationships
            dishes_data = []
            for dish, contains_rel in items_relationships:
                dishes_data.append(OrderedDish(
                    dish_id=dish.dish_id,
                    name=dish.name_of_food,
                    quantity=contains_rel.quantity,
                    price=float(contains_rel.price),
                    subtotal=float(contains_rel.price) * contains_rel.quantity
                ))

            order_data = OrderData(
                order_id=order.order_id,
                customer_id=customer_id,
                customer_name=customer.full_name,
                total_bill=float(order.total_bill),
                is_takeaway=bool(order.is_takeaway),
                table_id=order.table_id,
                notes=order.notes,
                created_at=placed_rel.created_at.isoformat() if placed_rel.created_at else None,
                ordered_dishes=dishes_data
            )
            orders.append(order_data)

        logger.info(f"Found {len(orders)} orders for customer {customer_id}")

        return UserOrdersResponse(
            customer=customer_data,
            orders=orders,
            total_orders=len(orders),
            status="success",
            timestamp=datetime.now().isoformat(),
            error=None
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving orders for customer {customer_id}: {str(e)}")
        return UserOrdersResponse(
            customer=CustomerData(customer_id=customer_id, full_name="", phone=None, email=None),
            orders=[],
            total_orders=0,
            status="failed",
            timestamp=datetime.now().isoformat(),
            error=str(e)
        )


# Mount the MCP server
app.mount("/", mcp_app)
