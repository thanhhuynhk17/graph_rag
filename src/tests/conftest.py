"""
Pytest configuration and fixtures for the restaurant order management system tests.

This module provides:
- Database connection fixtures
- Test data fixtures
- Cleanup utilities
- Common test utilities
"""

import os
import pytest
import pendulum
from typing import Generator, Dict, List
from neo4j import AsyncGraphDatabase
from neomodel import config, db

# Test database configuration
TEST_DB_URL = os.getenv("TEST_NEO4J_BOLT_URL", "bolt://neo4j:test@localhost:7687/test")


@pytest.fixture(autouse=True, scope="function")
def setup_test_database():
    """Set up test database connection for each test."""
    # Use main database for testing
    config.DATABASE_URL = "bolt://neo4j:12345678@localhost:7687"

    # Clean database before each test - delete ALL nodes and relationships
    try:
        # First, delete all relationships
        db.cypher_query("MATCH ()-[r]-() DELETE r")
        # Then delete all nodes
        db.cypher_query("MATCH (n) DELETE n")
        print("Database cleaned successfully")
    except Exception as e:
        print(f"Warning: Could not clear test database: {e}")

    yield

    # Cleanup after each test - ensure clean state
    try:
        # Delete all relationships first
        db.cypher_query("MATCH ()-[r]-() DELETE r")
        # Then delete all nodes
        db.cypher_query("MATCH (n) DELETE n")
        print("Database cleanup completed")
    except Exception as e:
        print(f"Warning: Could not cleanup after test: {e}")


@pytest.fixture
def sample_customer_data() -> Dict:
    """Sample customer data for testing."""
    return {
        "customer_id": "test_customer_001",
        "full_name": "Nguyễn Văn Test",
        "phone": ["0901234567", "0987654321"],
        "email": "test@example.com"
    }


@pytest.fixture
def sample_customer_data_clean() -> Dict:
    """Clean customer data for testing (alternative fixture)."""
    return {
        "customer_id": "test_customer_clean",
        "full_name": "Test Customer Clean",
        "phone": ["0901111111"],
        "email": "clean@test.com"
    }


@pytest.fixture
def sample_dish_data() -> List[Dict]:
    """Sample dish data for testing."""
    return [
        {
            "dish_id": "dish_001",
            "type_of_food": "MÓN KHAI VỊ",
            "name_of_food": "Bánh xèo",
            "how_to_prepare": "Đổ bột gạo pha nước cốt dừa, chiên vàng giòn cùng topping",
            "main_ingredients": "Bột gạo, tôm, thịt ba chỉ, giá đỗ",
            "taste": "Đồ béo",
            "outstanding_fragrance": "Thơm và béo của nước cốt dừa",
            "current_price": 145000.0,
            "number_of_people_eating": "2-3 người",
            "combine_info": "MÓN KHAI VỊ, Bánh xèo, Đổ bột gạo pha nước cốt dừa, chiên vàng giòn cùng topping, Bột gạo, tôm, thịt ba chỉ, giá đỗ, Đồ béo, Thơm và béo của nước cốt dừa hoà quyện cùng topping (tôm, thịt,), 2-3 người"
        },
        {
            "dish_id": "dish_002",
            "type_of_food": "MÓN THỊT",
            "name_of_food": "Thịt kho tiêu",
            "how_to_prepare": "Thịt heo thái miếng vừa, ướp nước mắm, đường, tiêu rồi kho đến khi thấm gia vị",
            "main_ingredients": "Thịt heo, nước mắm, đường, tiêu",
            "taste": "Đồ mặn",
            "outstanding_fragrance": "Mùi tiêu xay dậy lên trong nước kho",
            "current_price": 150000.0,
            "number_of_people_eating": "2-3 người",
            "combine_info": "MÓN THỊT, Thịt kho tiêu, Thịt heo thái miếng vừa, ướp nước mắm, đường, tiêu rồi kho đến khi thấm gia vị, Thịt heo, nước mắm, đường, tiêu, Đồ mặn, Mùi tiêu xay dậy lên trong nước kho, thơm nồng ấm., 2-3 người"
        }
    ]


@pytest.fixture
def sample_order_data() -> Dict:
    """Sample order data for testing."""
    return {
        "order_id": "order_001",
        "total_bill": 295000.0,
        "is_takeaway": False,
        "is_pre_paid": False,
        "table_id": 1,
        "notes": "Không hành, không ớt"
    }


@pytest.fixture
def sample_datetime():
    """Sample datetime for testing."""
    return pendulum.datetime(2025, 9, 30, 14, 30, tz="Asia/Ho_Chi_Minh")


@pytest.fixture
def mock_embedding():
    """Mock embedding vector for testing."""
    return [0.1] * 1024  # 1024-dimensional vector


# Utility functions for tests
def create_test_customer(data: Dict = None) -> 'Customer':
    """Create a test customer instance."""
    from ..models.order_graph import Customer

    if data is None:
        data = {
            "customer_id": "test_customer_util",
            "full_name": "Test Customer",
            "phone": ["0900000000"],
            "email": "util@test.com"
        }

    customer = Customer(**data)
    customer.save()
    return customer


def create_test_dish(data: Dict = None) -> 'Dish':
    """Create a test dish instance."""
    from ..models.order_graph import Dish

    if data is None:
        data = {
            "dish_id": "test_dish_util",
            "type_of_food": "MÓN KHAI VỊ",
            "name_of_food": "Test Dish",
            "current_price": 100000.0,
            "combine_info": "Test dish for unit testing"
        }

    dish = Dish(**data)
    dish.save()
    return dish


def create_test_order(data: Dict = None) -> 'Order':
    """Create a test order instance."""
    from ..models.order_graph import Order

    if data is None:
        data = {
            "order_id": "test_order_util",
            "total_bill": 100000.0,
            "is_takeaway": False,
            "table_id": 1
        }

    order = Order(**data)
    order.save()
    return order


def cleanup_test_data():
    """Clean up test data after tests."""
    try:
        # Delete all test nodes
        query = """
        MATCH (n)
        WHERE n._id STARTS WITH 'test_'
        DETACH DELETE n
        """
        db.cypher_query(query)
    except Exception as e:
        print(f"Warning: Could not cleanup test data: {e}")
