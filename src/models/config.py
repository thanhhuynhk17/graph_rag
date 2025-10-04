from typing import NamedTuple

class RestaurantConfig(NamedTuple):
    """Configuration for restaurant order management."""
    NUM_OF_TABLES: int = 2
    TABLE_WINDOW_MINUTES: int = 45
    TIMEZONE: str = "Asia/Ho_Chi_Minh"
