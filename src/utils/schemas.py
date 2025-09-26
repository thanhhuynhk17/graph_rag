from pydantic import BaseModel, Field

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