import uuid
import pendulum
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from fastmcp.tools.tool import ToolResult
from fastmcp.server.dependencies import get_context
from fastmcp.tools.tool import TextContent

from src.models.order_graph import Customer, Order
from src.utils.schemas import OrderResponse, OrderData, CustomerData, OrderedDish

import logging
logger = logging.getLogger(__name__)


class OrderService:
    """Centralized service for order processing logic shared between MCP tools and REST API."""

    async def process_order(
        self,
        customer_id: str,
        full_name: str,
        phone: Optional[str] = None,
        email: Optional[str] = None,
        dishes: Optional[List[Dict[str, Any]]] = None,
        is_takeaway: bool = False,
        arrived_at: Optional[str] = None,
        notes: Optional[str] = None
    ) -> Tuple[bool, Order, Customer, str]:
        """
        Core order processing logic shared between API and MCP tool.

        Args:
            customer_id: Unique customer identifier
            full_name: Customer full name
            phone: Customer phone number
            email: Customer email
            dishes: List of dishes with 'dish_id' and 'quantity'
            is_takeaway: True for takeaway, False for dine-in
            arrived_at: ISO datetime string when customer arrived
            notes: Order notes

        Returns:
            Tuple of (success: bool, order: Order, customer: Customer, error_msg: str)
        """
        try:
            # Validate required parameters
            if not customer_id or not full_name:
                raise ValueError("customer_id and full_name are required")

            if dishes is None:
                dishes = []

            # Parse arrival time or default to now
            if arrived_at:
                try:
                    arrival_time = pendulum.parse(arrived_at)
                except Exception as e:
                    raise ValueError(f"Invalid arrived_at format: {e}")
            else:
                arrival_time = pendulum.now("UTC")

            # Get or create customer
            customer_data = {
                'customer_id': customer_id,
                'full_name': full_name,
                'phone': phone,
                'email': email
            }
            customer = Customer.get_or_create(customer_data)

            # Create new order
            order_id = f"order_{uuid.uuid4().hex[:12]}"
            order = Order(
                order_id=order_id,
                total_bill=0.0,  # Will be calculated and updated
                is_takeaway=is_takeaway,
                notes=notes
            ).save()

            logger.info(f"Created order {order_id} for customer {customer_id}")

            # Place order (this handles table assignment if not takeaway, dish validation, pricing, etc.)
            completed_order = customer.place_order(
                order=order,
                arrived_at=arrival_time,
                dishes=dishes,
                notes=notes
            )

            # Reload order to get updated total_bill
            completed_order.refresh()

            return True, completed_order, customer, ""

        except Exception as e:
            logger.error(f"Error processing order: {str(e)}")
            return False, None, None, str(e)

    async def create_mcp_response(
        self,
        customer_id: str,
        full_name: str,
        phone: Optional[str] = None,
        email: Optional[str] = None,
        dishes: Optional[List[Dict[str, Any]]] = None,
        is_takeaway: bool = False,
        arrived_at: Optional[str] = None,
        notes: Optional[str] = None
    ) -> ToolResult:
        """
        Create MCP ToolResult response for take_order tool.
        """
        success, order, customer, error_msg = await self.process_order(
            customer_id, full_name, phone, email, dishes, is_takeaway, arrived_at, notes
        )

        if not success:
            return ToolResult(
                content=[TextContent(type="text", text=f"Lỗi khi tạo đơn hàng: {error_msg}")],
                structured_content={
                    "error": error_msg,
                    "status": "failed",
                    "timestamp": datetime.now().isoformat()
                }
            )

        # Create human-readable summary
        summary_parts = [
            f"Đã tạo đơn hàng thành công!",
            f"Order ID: {order.order_id}",
            f"Customer: {customer.full_name} ({customer.customer_id})"
        ]

        if not is_takeaway and hasattr(order, 'table_id') and order.table_id:
            summary_parts.append(f"Table: {order.table_id}")
        else:
            summary_parts.append("Takeaway order")

        summary_parts.append(f"Total: {order.total_bill:,.0f} vnđ")

        if order.notes:
            summary_parts.append(f"Notes: {order.notes}")

        # Get ordered dishes for display
        ordered_dishes = []
        if hasattr(order, 'items') and order.items:
            summary_parts.append("Ordered dishes:")
            for dish in order.items.all():
                rel_obj = order.items.relationship(dish)
                dish_info = {
                    "dish_id": dish.dish_id,
                    "name": dish.name_of_food,
                    "quantity": rel_obj.quantity,
                    "price": rel_obj.price,
                    "subtotal": rel_obj.quantity * rel_obj.price
                }
                ordered_dishes.append(dish_info)
                summary_parts.append(f"  - {dish_info['name']} x{dish_info['quantity']} = {dish_info['subtotal']:,.0f} vnđ")

        # Prepare structured response
        order_data = {
            "order_id": order.order_id,
            "customer_id": customer.customer_id,
            "customer_name": customer.full_name,
            "total_bill": order.total_bill,
            "is_takeaway": order.is_takeaway,
            "table_id": getattr(order, 'table_id', None),
            "notes": order.notes,
            "created_at": order.created_at.isoformat() if hasattr(order, 'created_at') else None,
            "ordered_dishes": ordered_dishes
        }

        return ToolResult(
            content=[TextContent(type="text", text="\n".join(summary_parts))],
            structured_content={
                "order": order_data,
                "customer": {
                    "customer_id": customer.customer_id,
                    "full_name": customer.full_name,
                    "phone": customer.phone,
                    "email": customer.email
                },
                "status": "success",
                "timestamp": datetime.now().isoformat()
            }
        )

    async def create_api_response(
        self,
        order_request
    ) -> OrderResponse:
        """
        Create REST API OrderResponse from an OrderRequest.
        """
        # Convert OrderRequest.dishes (List[OrderDish]) to List[Dict[str, Any]]
        dishes_param = []
        if order_request.dishes:
            for dish in order_request.dishes:
                dishes_param.append({
                    "dish_id": dish.dish_id,
                    "quantity": dish.quantity
                })

        success, order, customer, error_msg = await self.process_order(
            customer_id=order_request.customer_id,
            full_name=order_request.full_name,
            phone=order_request.phone,
            email=order_request.email,
            dishes=dishes_param,
            is_takeaway=order_request.is_takeaway,
            arrived_at=order_request.arrived_at,
            notes=order_request.notes
        )

        if not success:
            # Handle error case
            return OrderResponse(
                order=OrderData(
                    order_id="",
                    customer_id=order_request.customer_id,
                    customer_name="",
                    total_bill=0.0,
                    is_takeaway=order_request.is_takeaway,
                    ordered_dishes=[]
                ),
                customer=CustomerData(
                    customer_id=order_request.customer_id,
                    full_name=order_request.full_name
                ),
                status="failed",
                timestamp=datetime.now().isoformat(),
                error=error_msg
            )

        # Success case - build OrderData from structured_content
        ordered_dishes = []
        if hasattr(order, 'items') and order.items:
            for dish in order.items.all():
                rel_obj = order.items.relationship(dish)
                dish_info = OrderedDish(
                    dish_id=dish.dish_id,
                    name=dish.name_of_food,
                    quantity=rel_obj.quantity,
                    price=rel_obj.price,
                    subtotal=rel_obj.quantity * rel_obj.price
                )
                ordered_dishes.append(dish_info)

        order_data = OrderData(
            order_id=order.order_id,
            customer_id=customer.customer_id,
            customer_name=customer.full_name,
            total_bill=order.total_bill,
            is_takeaway=order.is_takeaway,
            table_id=getattr(order, 'table_id', None),
            notes=order.notes,
            created_at=order.created_at.isoformat() if hasattr(order, 'created_at') else None,
            ordered_dishes=ordered_dishes
        )

        customer_data = CustomerData(
            customer_id=customer.customer_id,
            full_name=customer.full_name,
            phone=customer.phone,
            email=customer.email
        )

        return OrderResponse(
            order=order_data,
            customer=customer_data,
            status="success",
            timestamp=datetime.now().isoformat()
        )
