from pydantic import BaseModel, Field
from typing import Literal

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
    
class SearchTypeCategory(BaseModel):
    category: Literal[
        "món cá", "món khai vị", "món ăn chơi", "món rau", "món gỏi",
        "món gà, vịt & trứng", "món tôm & mực", "món xào", "nước mát nhà làm",
        "lẩu", "món thịt", "món sườn & đậu hũ", "món canh", "các loại khô", "tráng miệng"
    ] = Field(..., description="Loại món ăn")
    keyword: str = Field("", description="Từ khóa thêm (VD: 'cay', 'mặn')")