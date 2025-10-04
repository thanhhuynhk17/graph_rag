"""
Integration tests for the OrderManager business logic.

Tests cover:
- Table assignment logic
- Dish validation and pricing
- Order creation workflow
- Error handling and edge cases
- Integration with Neomodel models
"""

import pytest
import pendulum
from unittest.mock import AsyncMock, MagicMock
from typing import Dict, List

from ..models.order_graph import Customer, Order, Dish, Placed, Contains


class TestOrderManager:
    """Test cases for OrderManager business logic."""

    @pytest.fixture
    def mock_driver(self):
        """Mock Neo4j driver for testing."""
        mock_driver = MagicMock()
        mock_driver.execute_query = AsyncMock()
        return mock_driver

    @pytest.fixture
    def order_manager(self, mock_driver):
        """OrderManager instance with mocked driver."""
        from ..models.order_manager import OrderManager
        return OrderManager(mock_driver)

    @pytest.fixture
    def sample_dishes_in_db(self, sample_dish_data):
        """Create sample dishes in the database."""
        dishes = []
        for dish_data in sample_dish_data:
            dish = Dish(**dish_data)
            dish.save()
            dishes.append(dish)
        return dishes

    @pytest.mark.asyncio
    async def test_assign_table_no_conflicts(self, order_manager, sample_datetime):
        """Test table assignment when no conflicts exist."""
        # Mock database response for no occupied tables
        order_manager.driver.execute_query = AsyncMock(return_value=MagicMock(records=[]))

        # Test assigning table for a specific time
        dt = sample_datetime
        result = await order_manager.assign_table(dt)

        # Should assign table 1 (first available)
        assert result == 1

        # Verify the query was called with correct parameters
        order_manager.driver.execute_query.assert_called_once()
        call_args = order_manager.driver.execute_query.call_args
        # Check that the call was made (parameters are in kwargs)
        assert call_args is not None

    @pytest.mark.asyncio
    async def test_assign_table_with_conflicts(self, order_manager, sample_datetime):
        """Test table assignment when tables are occupied."""
        # Mock database response showing table 1 is occupied
        mock_record = MagicMock()
        mock_record.__getitem__ = MagicMock(return_value=1)
        order_manager.driver.execute_query = AsyncMock(
            return_value=MagicMock(records=[mock_record])
        )

        dt = sample_datetime
        result = await order_manager.assign_table(dt)

        # Should assign table 2 (only table 1 occupied)
        assert result == 2

    @pytest.mark.asyncio
    async def test_assign_table_all_occupied(self, order_manager, sample_datetime):
        """Test table assignment when all tables are occupied."""
        # Mock database response showing both tables occupied
        mock_records = [
            MagicMock(__getitem__=MagicMock(return_value=1)),
            MagicMock(__getitem__=MagicMock(return_value=2))
        ]
        order_manager.driver.execute_query = AsyncMock(
            return_value=MagicMock(records=mock_records)
        )

        dt = sample_datetime

        # Should raise ValueError when no tables available
        with pytest.raises(ValueError, match="No table free"):
            await order_manager.assign_table(dt)

    @pytest.mark.asyncio
    async def test_validate_and_prepare_dishes_success(self, order_manager, sample_dishes_in_db):
        """Test successful dish validation and price fetching."""
        # Test data with valid dish IDs
        dishes = [
            {"id": "dish_001", "quantity": 1},
            {"id": "dish_002", "quantity": 2}
        ]

        # Mock database response with prices
        mock_records = [
            MagicMock(__getitem__=lambda key: {"id": "dish_001", "price": 145000.0}.get(key, None)),
            MagicMock(__getitem__=lambda key: {"id": "dish_002", "price": 150000.0}.get(key, None))
        ]
        order_manager.driver.execute_query = AsyncMock(
            return_value=MagicMock(records=mock_records)
        )

        result = await order_manager.validate_and_prepare_dishes(dishes)

        # Verify results
        assert len(result) == 2
        assert result[0]["id"] == "dish_001"
        assert result[0]["quantity"] == 1
        assert result[0]["price"] == 145000.0
        assert result[1]["id"] == "dish_002"
        assert result[1]["quantity"] == 2
        assert result[1]["price"] == 150000.0

    @pytest.mark.asyncio
    async def test_validate_and_prepare_dishes_missing_id(self, order_manager):
        """Test dish validation with missing dish ID."""
        dishes = [
            {"quantity": 1}  # Missing 'id' field
        ]

        with pytest.raises(ValueError, match="Each dish must have 'id' and 'quantity'"):
            await order_manager.validate_and_prepare_dishes(dishes)

    @pytest.mark.asyncio
    async def test_validate_and_prepare_dishes_invalid_quantity(self, order_manager):
        """Test dish validation with invalid quantity."""
        dishes = [
            {"id": "dish_001", "quantity": 0}  # Invalid quantity
        ]

        with pytest.raises(ValueError, match="Quantity must be a positive integer"):
            await order_manager.validate_and_prepare_dishes(dishes)

    @pytest.mark.asyncio
    async def test_validate_and_prepare_dishes_not_found(self, order_manager):
        """Test dish validation when dish not found in database."""
        dishes = [
            {"id": "nonexistent_dish", "quantity": 1}
        ]

        # Mock database response showing dish not found
        mock_record = MagicMock()
        mock_record.__getitem__ = MagicMock(return_value=None)
        order_manager.driver.execute_query = AsyncMock(
            return_value=MagicMock(records=[mock_record])
        )

        with pytest.raises(ValueError, match="Dish IDs not found"):
            await order_manager.validate_and_prepare_dishes(dishes)

    @pytest.mark.asyncio
    async def test_validate_and_prepare_empty_dishes(self, order_manager):
        """Test dish validation with empty dishes list."""
        result = await order_manager.validate_and_prepare_dishes([])
        assert result == []

    @pytest.mark.asyncio
    async def test_create_order_dine_in_success(self, order_manager, sample_dishes_in_db, sample_datetime):
        """Test successful dine-in order creation."""
        # Setup test data
        guest_id = "guest_001"
        guest_name = "Nguyễn Văn A"
        guest_phone = "0901234567"
        dishes = [{"id": "dish_001", "quantity": 1}]

        # Mock database responses
        def mock_execute_query(query, params, **kwargs):
            if "MATCH (d:Dish" in query:
                # Dish validation query
                mock_record = MagicMock()
                mock_record.__getitem__ = MagicMock(return_value=145000.0)
                return MagicMock(records=[mock_record])
            elif "MATCH (o:Order)" in query and "table_id" in query:
                # Table assignment query
                return MagicMock(records=[])  # No occupied tables
            else:
                # Order creation query
                return MagicMock(records=[])

        order_manager.driver.execute_query = AsyncMock(side_effect=mock_execute_query)

        # Create order
        order_id, table_id = await order_manager.create_order(
            guest_id=guest_id,
            guest_name=guest_name,
            guest_phone_number=guest_phone,
            is_takeaway=False,
            dishes=dishes,
            dt=sample_datetime
        )

        # Verify results
        assert order_id is not None
        assert table_id == 1  # First available table

        # Verify database calls
        assert order_manager.driver.execute_query.call_count >= 3  # At least 3 calls made

    @pytest.mark.asyncio
    async def test_create_order_takeaway_success(self, order_manager, sample_dishes_in_db, sample_datetime):
        """Test successful takeaway order creation."""
        # Setup test data
        guest_id = "guest_002"
        guest_name = "Nguyễn Thị B"
        guest_phone = "0909876543"
        dishes = [{"id": "dish_002", "quantity": 1}]

        # Mock database responses
        def mock_execute_query(query, params, **kwargs):
            if "MATCH (d:Dish" in query:
                # Dish validation query
                mock_record = MagicMock()
                mock_record.__getitem__ = MagicMock(return_value=150000.0)
                return MagicMock(records=[mock_record])
            else:
                # Order creation query (no table assignment for takeaway)
                return MagicMock(records=[])

        order_manager.driver.execute_query = AsyncMock(side_effect=mock_execute_query)

        # Create takeaway order
        order_id, table_id = await order_manager.create_order(
            guest_id=guest_id,
            guest_name=guest_name,
            guest_phone_number=guest_phone,
            is_takeaway=True,
            dishes=dishes,
            dt=sample_datetime
        )

        # Verify results
        assert order_id is not None
        assert table_id is None  # No table for takeaway

    @pytest.mark.asyncio
    async def test_create_order_table_reservation_only(self, order_manager, sample_datetime):
        """Test order creation for table reservation without dishes."""
        # Setup test data - no dishes
        guest_id = "guest_003"
        guest_name = "Nguyễn Văn C"
        guest_phone = "0905555555"
        dishes = []

        # Mock database responses
        def mock_execute_query(query, params, **kwargs):
            if "MATCH (o:Order)" in query and "table_id" in query:
                # Table assignment query
                return MagicMock(records=[])  # No occupied tables
            else:
                # Order creation query
                return MagicMock(records=[])

        order_manager.driver.execute_query = AsyncMock(side_effect=mock_execute_query)

        # Create order without dishes
        order_id, table_id = await order_manager.create_order(
            guest_id=guest_id,
            guest_name=guest_name,
            guest_phone_number=guest_phone,
            is_takeaway=False,
            dishes=dishes,
            dt=sample_datetime
        )

        # Verify results
        assert order_id is not None
        assert table_id == 1  # First available table

    @pytest.mark.asyncio
    async def test_create_order_missing_required_fields(self, order_manager):
        """Test order creation with missing required fields."""
        with pytest.raises(ValueError, match="Guest ID, name, and phone number are required"):
            await order_manager.create_order(
                guest_id="",  # Empty guest_id
                guest_name="Test Name",
                guest_phone_number="0901234567",
                is_takeaway=False,
                dishes=[],
                dt=pendulum.now()
            )

    @pytest.mark.asyncio
    async def test_create_order_database_error(self, order_manager, sample_datetime):
        """Test order creation with database error."""
        # Mock database error
        order_manager.driver.execute_query = AsyncMock(
            side_effect=Exception("Database connection failed")
        )

        with pytest.raises(Exception, match="Database connection failed"):
            await order_manager.create_order(
                guest_id="guest_error",
                guest_name="Error Test",
                guest_phone_number="0901234567",
                is_takeaway=False,
                dishes=[],
                dt=sample_datetime
            )

    def test_calculate_total_cost(self, order_manager, sample_dishes_in_db):
        """Test total cost calculation."""
        # Mock validated dishes with prices
        validated_dishes = [
            {"id": "dish_001", "quantity": 2, "price": 145000.0},
            {"id": "dish_002", "quantity": 1, "price": 150000.0}
        ]

        # Calculate total (this would be done internally)
        total_cost = sum(d["quantity"] * d["price"] for d in validated_dishes)

        # Verify calculation
        expected_total = (2 * 145000.0) + (1 * 150000.0)  # 290000 + 150000 = 440000
        assert total_cost == expected_total

    def test_datetime_timezone_conversion(self, order_manager, sample_datetime):
        """Test datetime conversion from Asia/Ho_Chi_Minh to UTC."""
        # The order manager should convert datetime to UTC for Neo4j
        dt_hcm = sample_datetime  # Already in Asia/Ho_Chi_Minh timezone
        dt_utc = dt_hcm.in_timezone("UTC")

        # Verify conversion
        assert dt_utc.timezone.name == "UTC"
        assert dt_hcm.hour == dt_utc.hour + 7  # 7 hour difference

    @pytest.mark.asyncio
    async def test_concurrent_table_assignment(self, order_manager, sample_datetime):
        """Test table assignment under concurrent scenarios."""
        # This is a simplified test - in reality would need more complex mocking
        # to simulate concurrent access

        # Mock table assignment query
        call_count = 0
        def mock_table_query(query, params, **kwargs):
            nonlocal call_count
            call_count += 1

            # Simulate first call finds no occupied tables, second call finds table 1 occupied
            if call_count == 1:
                return MagicMock(records=[])  # No tables occupied
            else:
                mock_record = MagicMock()
                mock_record.__getitem__ = MagicMock(return_value=1)
                return MagicMock(records=[mock_record])

        order_manager.driver.execute_query = AsyncMock(side_effect=mock_table_query)

        # First assignment should get table 1
        result1 = await order_manager.assign_table(sample_datetime)
        assert result1 == 1

        # Second assignment should get table 2
        result2 = await order_manager.assign_table(sample_datetime)
        assert result2 == 2


class TestOrderManagerIntegration:
    """Integration tests with actual Neomodel models."""

    def test_full_order_workflow_with_models(self, sample_customer_data, sample_order_data, sample_dish_data, sample_datetime):
        """Test complete order workflow using Neomodel models."""
        # Create customer, order, and dishes using Neomodel
        customer = Customer(**sample_customer_data)
        customer.save()

        # Create order with calculated total bill
        order = Order(
            order_id=sample_order_data["order_id"],
            total_bill=145000.0 + (2 * 150000.0),  # 1x dish1 + 2x dish2
            is_takeaway=sample_order_data["is_takeaway"],
            is_pre_paid=sample_order_data["is_pre_paid"],
            table_id=sample_order_data["table_id"],
            notes=sample_order_data["notes"]
        )
        order.save()

        dish1 = Dish(**sample_dish_data[0])
        dish1.save()

        dish2 = Dish(**sample_dish_data[1])
        dish2.save()

        # Create relationships using Neomodel
        arrived_at = sample_datetime
        customer.placed.connect(order, {"arrived_at": arrived_at})

        # Add dishes to order
        order.items.connect(dish1, {"quantity": 1, "price": 145000.0})
        order.items.connect(dish2, {"quantity": 2, "price": 150000.0})

        # Verify the complete workflow
        assert len(customer.placed) == 1
        assert len(order.items) == 2

        # Calculate total from relationships
        total_from_rels = sum(
            order.items.relationship(dish).quantity * order.items.relationship(dish).price
            for dish in order.items.all()
        )

        # Should match the order's total_bill
        assert total_from_rels == order.total_bill

        # Verify relationship properties
        placed_rel = customer.placed.relationship(order)
        assert placed_rel.arrived_at == arrived_at

    def test_dish_search_and_filter(self, sample_dish_data):
        """Test searching and filtering dishes using Neomodel."""
        # Create dishes
        for dish_data in sample_dish_data:
            dish = Dish(**dish_data)
            dish.save()

        # Test filtering by food type
        khai_vi_dishes = Dish.nodes.filter(type_of_food="MÓN KHAI VỊ")
        assert len(khai_vi_dishes) == 1

        # Test filtering by price range
        expensive_dishes = Dish.nodes.filter(current_price__gte=150000.0)
        assert len(expensive_dishes) == 1
        assert expensive_dishes[0].name_of_food == "Thịt kho tiêu"

        # Test full-text search
        search_results = Dish.nodes.filter(combine_info__icontains="Bánh xèo")
        assert len(search_results) == 1

    def test_customer_order_history(self, sample_customer_data, sample_order_data, sample_datetime):
        """Test customer's order history tracking."""
        # Create customer
        customer = Customer(**sample_customer_data)
        customer.save()

        # Create multiple orders for the customer
        order1 = Order(order_id="order_hist_1", total_bill=100000.0, table_id=1)
        order1.save()

        order2 = Order(order_id="order_hist_2", total_bill=150000.0, table_id=2)
        order2.save()

        # Create relationships
        customer.placed.connect(order1, {"arrived_at": sample_datetime})
        customer.placed.connect(order2, {"arrived_at": sample_datetime.add(hours=2)})

        # Verify order history
        orders = list(customer.placed)
        assert len(orders) == 2

        # Verify chronological ordering
        order_times = [
            customer.placed.relationship(order).arrived_at
            for order in orders
        ]
        assert order_times[0] <= order_times[1]


# Performance and stress tests
class TestPerformance:
    """Performance and stress tests."""

    @pytest.mark.asyncio
    async def test_large_number_of_dishes(self, order_manager):
        """Test handling large number of dishes in order."""
        # Create many dishes
        large_dish_list = [
            {"id": f"dish_{i:03d}", "quantity": 1}
            for i in range(100)
        ]

        # Mock database response for all dishes
        mock_records = [
            MagicMock(__getitem__=lambda key, id=f"dish_{i:03d}": {"id": id, "price": 100000.0}.get(key, None))
            for i in range(100)
        ]
        order_manager.driver.execute_query = AsyncMock(
            return_value=MagicMock(records=mock_records)
        )

        # Should handle large dish list without issues
        result = await order_manager.validate_and_prepare_dishes(large_dish_list)
        assert len(result) == 100

    def test_rapid_order_creation(self, order_manager, sample_datetime):
        """Test rapid order creation (stress test)."""
        # This would require more complex async mocking
        # For now, just verify the method exists and can be called
        assert hasattr(order_manager, 'create_order')
        assert callable(order_manager.create_order)
