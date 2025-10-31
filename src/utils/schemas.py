from pydantic import BaseModel, Field, model_validator
from typing import Literal, Optional, Dict, Any, List

class SearchReq(BaseModel):
    query: str = Field(..., min_length=3, description="Tên món ăn, tag hoặc ingredient để tìm kiếm")
    k: int = Field(10, description="Số lượng kết quả muốn lấy")

class DishReq(BaseModel):
    dish_id: str

class PriceReq(BaseModel):
    max_price: int

class WipeReq(BaseModel):
    drop_schema: bool = Field(False, description="Nếu True, xóa luôn tất cả index/constraint trong DB")
    database: str | None = Field(None, description="Tên database cần xóa, để None sẽ dùng DB mặc định")

class FeedbackReq(BaseModel):
    customer_id: str
    bill_id: str
    dish_id: str
    text: str

# Order Management Schemas (sync with MCP take_order tool)
class OrderDish(BaseModel):
    dish_id: str = Field(..., description="Dish identifier")
    quantity: int = Field(..., gt=0, description="Quantity (must be positive)")

class OrderRequest(BaseModel):
    """Request model for take_order API endpoint - matches MCP take_order parameters exactly"""
    customer_id: str = Field(..., description="Unique customer identifier")
    full_name: str = Field(..., description="Customer full name")
    phone: Optional[str] = Field(None, description="Customer phone number")
    email: Optional[str] = Field(None, description="Customer email")
    dishes: Optional[List[OrderDish]] = Field(None, description="List of dishes with dish_id and quantity")
    is_takeaway: bool = Field(False, description="True for takeaway, False for dine-in")
    arrived_at: Optional[str] = Field(None, description="ISO datetime string when customer arrived")
    notes: Optional[str] = Field(None, description="Order notes")


    @model_validator(mode="after")
    def validate_business_logic(self) -> "OrderRequest":
        """Validate business logic constraints"""
        # Check at least one contact method provided
        if not self.phone and not self.email:
            raise ValueError("At least one contact method (phone or email) must be provided")

        # Validate customer required fields
        if not self.full_name or not self.full_name.strip():
            raise ValueError("full_name is required and cannot be empty")
        if not self.customer_id or not self.customer_id.strip():
            raise ValueError("customer_id is required and cannot be empty")

        # For dine-in orders, dishes should typically be provided
        if not self.is_takeaway and self.dishes is not None and len(self.dishes) > 0:
            # Validate dish quantities are positive
            for dish in self.dishes:
                if dish.quantity <= 0:
                    raise ValueError(f"Dish quantity must be positive: {dish.dish_id} has quantity {dish.quantity}")

        return self

class OrderedDish(BaseModel):
    """Dish information in order response"""
    dish_id: str
    name: str
    quantity: int
    price: float
    subtotal: float

class OrderData(BaseModel):
    """Order information in response"""
    order_id: str
    customer_id: str
    customer_name: str
    total_bill: float
    is_takeaway: bool
    table_id: Optional[int] = None
    notes: Optional[str] = None
    created_at: Optional[str] = None
    ordered_dishes: List[OrderedDish]

class CustomerData(BaseModel):
    """Customer information in response"""
    customer_id: str
    full_name: str
    phone: Optional[List[str]] = None
    email: Optional[str] = None

class OrderResponse(BaseModel):
    """Response model for take_order API endpoint - matches MCP ToolResult.structured_content"""
    order: OrderData
    customer: CustomerData
    status: Literal["success", "failed"]
    timestamp: str
    error: Optional[str] = None
    
class SearchTypeCategory(BaseModel):
    category: Literal[
        "món cá", "món khai vị", "món ăn chơi", "món rau", "món gỏi",
        "món gà, vịt & trứng", "món tôm & mực", "món xào", "nước mát nhà làm",
        "lẩu", "món thịt", "món sườn & đậu hũ", "món canh", "các loại khô", "tráng miệng"
    ] = Field(..., description="Loại món ăn")
    keyword: str = Field("", description="Từ khóa thêm (VD: 'cay', 'mặn')")

# User Orders Schemas
class UserOrdersResponse(BaseModel):
    """Response model for get_user_orders API endpoint"""
    customer: CustomerData
    orders: List[OrderData]
    total_orders: int
    status: Literal["success", "failed"]
    timestamp: str
    error: Optional[str] = None
