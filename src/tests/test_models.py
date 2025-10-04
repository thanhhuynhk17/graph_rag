"""
Unit tests for Neomodel models in the restaurant order management system.

Tests cover:
- Model creation and validation
- Relationship management
- Query operations
- Full-text and vector search
- Edge cases and error handling
"""

import pytest
import pendulum
from typing import List
from neomodel import DoesNotExist, UniqueProperty

from ..models.order_graph import Customer, Order, Dish, Placed, Contains


class TestCustomerModel:
    """Test cases for Customer model."""

    def test_customer_creation(self, sample_customer_data_clean):
        """Test creating a customer node."""
        customer = Customer(**sample_customer_data_clean)
        customer.save()

        # Verify the customer was saved
        assert customer.customer_id == sample_customer_data_clean["customer_id"]
        assert customer.full_name == sample_customer_data_clean["full_name"]
        assert customer.phone == sample_customer_data_clean["phone"]
        assert customer.email == sample_customer_data_clean["email"]

        # Verify it can be retrieved
        # Debug: Check if customer exists in database
        all_customers = Customer.nodes.all()
        print(f"Total customers in DB: {len(all_customers)}")
        for cust in all_customers:
            print(f"Customer in DB: {cust.customer_id} - {cust.full_name}")

        # Try to find by customer_id
        retrieved = None
        for cust in all_customers:
            if cust.customer_id == sample_customer_data_clean["customer_id"]:
                retrieved = cust
                break

        if retrieved is None:
            # Try filter as last resort
            try:
                retrieved = Customer.nodes.filter(customer_id=sample_customer_data_clean["customer_id"]).first()
            except:
                pass

        assert retrieved is not None, f"Could not find customer with customer_id: {sample_customer_data_clean['customer_id']}"
        assert retrieved.full_name == sample_customer_data_clean["full_name"]

    def test_customer_unique_id_constraint(self, sample_customer_data):
        """Test that customer _id must be unique."""
        # Create first customer
        customer1 = Customer(**sample_customer_data)
        customer1.save()

        # Try to create second customer with same ID
        customer2 = Customer(**sample_customer_data)
        with pytest.raises(UniqueProperty):
            customer2.save()

    def test_customer_phone_array(self):
        """Test customer phone array property."""
        customer = Customer(
            customer_id="test_phone_customer",
            full_name="Phone Test Customer",
            phone=["0901111111", "0902222222"]
        )
        customer.save()

        retrieved = Customer.nodes.filter(customer_id="test_phone_customer").first()
        assert len(retrieved.phone) == 2
        assert "0901111111" in retrieved.phone

    def test_customer_without_optional_fields(self):
        """Test creating customer with only required fields."""
        customer = Customer(
            customer_id="minimal_customer",
            full_name="Minimal Customer"
        )
        customer.save()

        retrieved = Customer.nodes.filter(customer_id="minimal_customer").first()
        assert retrieved.full_name == "Minimal Customer"
        assert retrieved.phone == []
        assert retrieved.email is None


class TestDishModel:
    """Test cases for Dish model."""

    def test_dish_creation(self, sample_dish_data):
        """Test creating a dish node."""
        dish_data = sample_dish_data[0]
        dish = Dish(**dish_data)
        dish.save()

        # Verify the dish was saved
        assert dish.dish_id == dish_data["dish_id"]
        assert dish.name_of_food == dish_data["name_of_food"]
        assert dish.current_price == dish_data["current_price"]
        assert dish.combine_info == dish_data["combine_info"]

    def test_dish_fulltext_search(self, sample_dish_data):
        """Test full-text search on combine_info field."""
        # Create test dishes
        for dish_data in sample_dish_data:
            dish = Dish(**dish_data)
            dish.save()

        # Search for dishes containing "Bánh xèo"
        results = Dish.nodes.filter(combine_info__icontains="Bánh xèo")
        assert len(results) == 1
        assert results[0].name_of_food == "Bánh xèo"

    def test_dish_vector_index(self, sample_dish_data, mock_embedding):
        """Test vector embedding property."""
        dish_data = sample_dish_data[0]
        dish_data["embedding"] = mock_embedding

        dish = Dish(**dish_data)
        dish.save()

        # Verify embedding was saved
        retrieved = Dish.nodes.filter(dish_id=dish_data["dish_id"]).first()
        assert retrieved.embedding == mock_embedding

    def test_dish_price_validation(self):
        """Test dish price handling."""
        dish = Dish(
            dish_id="price_test_dish",
            name_of_food="Price Test",
            current_price=100000.0,
            combine_info="Test dish for price validation"
        )
        dish.save()

        retrieved = Dish.nodes.filter(dish_id="price_test_dish").first()
        assert retrieved.current_price == 100000.0


class TestOrderModel:
    """Test cases for Order model."""

    def test_order_creation(self, sample_order_data):
        """Test creating an order node."""
        order = Order(**sample_order_data)
        order.save()

        # Verify the order was saved
        assert order.order_id == sample_order_data["order_id"]
        assert order.total_bill == sample_order_data["total_bill"]
        assert order.is_takeaway == sample_order_data["is_takeaway"]
        assert order.table_id == sample_order_data["table_id"]

    def test_order_without_optional_fields(self):
        """Test creating order with minimal fields."""
        order = Order(
            order_id="minimal_order",
            total_bill=50000.0,
            is_takeaway=False
        )
        order.save()

        retrieved = Order.nodes.filter(order_id="minimal_order").first()
        assert retrieved.total_bill == 50000.0
        assert retrieved.is_takeaway is False
        assert retrieved.table_id is None
        assert retrieved.notes is None


class TestRelationships:
    """Test cases for model relationships."""

    def test_customer_order_relationship(self, sample_customer_data, sample_order_data, sample_datetime):
        """Test PLACED relationship between Customer and Order."""
        # Create customer and order
        customer = Customer(**sample_customer_data)
        customer.save()

        order = Order(**sample_order_data)
        order.save()

        # Create relationship with arrived_at time
        arrived_at = sample_datetime
        customer.placed.connect(order, {"arrived_at": arrived_at})

        # Verify relationship exists
        assert len(customer.placed) == 1
        assert customer.placed[0].order_id == order.order_id

        # Verify relationship properties
        placed_rel = customer.placed.relationship(order)
        assert placed_rel.arrived_at == arrived_at

    def test_order_dish_relationship(self, sample_order_data, sample_dish_data):
        """Test CONTAINS relationship between Order and Dish."""
        # Create order and dishes
        order = Order(**sample_order_data)
        order.save()

        dish1 = Dish(**sample_dish_data[0])
        dish1.save()

        dish2 = Dish(**sample_dish_data[1])
        dish2.save()

        # Create relationships with quantities and prices
        order.items.connect(dish1, {"quantity": 1, "price": 145000.0})
        order.items.connect(dish2, {"quantity": 2, "price": 150000.0})

        # Verify relationships exist
        assert len(order.items) == 2

        # Verify relationship properties
        contains_rel1 = order.items.relationship(dish1)
        assert contains_rel1.quantity == 1
        assert contains_rel1.price == 145000.0

        contains_rel2 = order.items.relationship(dish2)
        assert contains_rel2.quantity == 2
        assert contains_rel2.price == 150000.0

    def test_relationship_deletion(self, sample_customer_data, sample_order_data):
        """Test deleting relationships."""
        # Create customer and order with relationship
        customer = Customer(**sample_customer_data)
        customer.save()

        order = Order(**sample_order_data)
        order.save()

        customer.placed.connect(order)

        # Verify relationship exists
        assert len(customer.placed) == 1

        # Delete relationship
        customer.placed.disconnect(order)

        # Verify relationship is gone
        assert len(customer.placed) == 0


class TestQueryOperations:
    """Test cases for query operations."""

    def test_customer_filtering(self, sample_customer_data):
        """Test filtering customers."""
        # Create test customers
        customer1 = Customer(customer_id="filter_test_1", full_name="Nguyễn A", phone=["0901111111"])
        customer1.save()

        customer2 = Customer(customer_id="filter_test_2", full_name="Nguyễn B", phone=["0902222222"])
        customer2.save()

        # Filter by phone (using 'in' operator for array properties)
        results = Customer.nodes.filter(phone="0901111111")
        assert len(results) == 1
        assert results[0].full_name == "Nguyễn A"

    def test_order_by_table_id(self, sample_order_data):
        """Test ordering orders by table_id."""
        # Create test orders
        order1 = Order(order_id="order_by_test_1", table_id=2, total_bill=100000.0)
        order1.save()

        order2 = Order(order_id="order_by_test_2", table_id=1, total_bill=150000.0)
        order2.save()

        # Order by table_id
        results = Order.nodes.order_by("table_id")
        assert len(results) == 2
        assert results[0].table_id == 1
        assert results[1].table_id == 2

    def test_dish_search_by_type(self, sample_dish_data):
        """Test searching dishes by type_of_food."""
        # Create test dishes
        for dish_data in sample_dish_data:
            dish = Dish(**dish_data)
            dish.save()

        # Search by food type
        khai_vi_dishes = Dish.nodes.filter(type_of_food="MÓN KHAI VỊ")
        assert len(khai_vi_dishes) == 1
        assert khai_vi_dishes[0].name_of_food == "Bánh xèo"

        thit_dishes = Dish.nodes.filter(type_of_food="MÓN THỊT")
        assert len(thit_dishes) == 1
        assert thit_dishes[0].name_of_food == "Thịt kho tiêu"


class TestEdgeCases:
    """Test cases for edge cases and error handling."""

    def test_delete_nonexistent_node(self):
        """Test deleting a node that doesn't exist."""
        with pytest.raises(DoesNotExist):
            Customer.nodes.get(customer_id="nonexistent_id")

    def test_relationship_with_missing_nodes(self, sample_customer_data, sample_order_data):
        """Test relationship operations with missing nodes."""
        customer = Customer(**sample_customer_data)
        customer.save()

        order = Order(**sample_order_data)
        order.save()

        # Delete order
        order.delete()

        # Try to access relationship (should not raise error, just be empty)
        assert len(customer.placed) == 0

    def test_large_phone_array(self):
        """Test customer with many phone numbers."""
        many_phones = [f"090{i:06d}" for i in range(10)]  # 0900000000 to 0900000009

        customer = Customer(
            customer_id="many_phones_customer",
            full_name="Many Phones Customer",
            phone=many_phones
        )
        customer.save()

        retrieved = Customer.nodes.filter(customer_id="many_phones_customer").first()
        assert len(retrieved.phone) == 10
        assert retrieved.phone == many_phones

    def test_special_characters_in_names(self):
        """Test handling of special characters in names."""
        special_name = "Nguyễn Thị Ái-É Ò Ó Ỏ Ọ"

        customer = Customer(
            customer_id="special_chars_customer",
            full_name=special_name
        )
        customer.save()

        retrieved = Customer.nodes.filter(customer_id="special_chars_customer").first()
        assert retrieved.full_name == special_name


class TestDataIntegrity:
    """Test cases for data integrity and constraints."""

    def test_cascade_delete_behavior(self, sample_customer_data, sample_order_data, sample_dish_data):
        """Test that deleting customer doesn't cascade to order."""
        # Create customer, order, and dishes
        customer = Customer(**sample_customer_data)
        customer.save()

        order = Order(**sample_order_data)
        order.save()

        dish = Dish(**sample_dish_data[0])
        dish.save()

        # Create relationships
        customer.placed.connect(order)
        order.items.connect(dish)

        # Delete customer
        customer.delete()

        # Order should still exist (no cascade)
        retrieved_order = Order.nodes.get(order_id=sample_order_data["order_id"])
        assert retrieved_order is not None

        # But relationship should be gone
        assert len(retrieved_order.items) == 0

    def test_relationship_property_validation(self, sample_order_data, sample_dish_data):
        """Test validation of relationship properties."""
        order = Order(**sample_order_data)
        order.save()

        dish = Dish(**sample_dish_data[0])
        dish.save()

        # Test with valid relationship properties
        order.items.connect(dish, {"quantity": 1, "price": 145000.0})

        # Verify properties
        contains_rel = order.items.relationship(dish)
        assert contains_rel.quantity == 1
        assert contains_rel.price == 145000.0


# Cleanup after all tests
@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Clean up test data after each test."""
    yield
    # This will run after each test
    try:
        # Delete all test nodes created in this test session
        query = """
        MATCH (n)
        WHERE n.customer_id STARTS WITH 'test_' OR
              n.customer_id STARTS WITH 'filter_test_' OR
              n.customer_id STARTS WITH 'minimal_' OR
              n.customer_id STARTS WITH 'price_test_' OR
              n.customer_id STARTS WITH 'order_by_test_' OR
              n.customer_id STARTS WITH 'special_chars_' OR
              n.customer_id STARTS WITH 'many_phones_' OR
              n.order_id STARTS WITH 'test_' OR
              n.order_id STARTS WITH 'filter_test_' OR
              n.order_id STARTS WITH 'minimal_' OR
              n.order_id STARTS WITH 'price_test_' OR
              n.order_id STARTS WITH 'order_by_test_' OR
              n.order_id STARTS WITH 'special_chars_' OR
              n.order_id STARTS WITH 'many_phones_' OR
              n.dish_id STARTS WITH 'test_' OR
              n.dish_id STARTS WITH 'filter_test_' OR
              n.dish_id STARTS WITH 'minimal_' OR
              n.dish_id STARTS WITH 'price_test_' OR
              n.dish_id STARTS WITH 'order_by_test_' OR
              n.dish_id STARTS WITH 'special_chars_' OR
              n.dish_id STARTS WITH 'many_phones_'
        DETACH DELETE n
        """
        from neomodel import db
        db.cypher_query(query)
    except Exception:
        pass  # Ignore cleanup errors
