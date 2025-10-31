"""
Tests for the MCP server functionality.
Tests the refactored tools that use the Dish model.
"""

import pytest
from typing import List
import pytest_asyncio
import os
import tempfile
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi import FastAPI
from ..mcp_server import mcp, multi_dish_lookup, menu_value_count_and_price, lifespan, clear_database_schema
from ..models.order_graph import Dish
from neomodel import db


@pytest_asyncio.fixture
async def setup_test_database_async():
    """Async fixture for database cleanup."""
    # Clean database before test
    try:
        db.cypher_query("MATCH ()-[r]-() DELETE r")
        db.cypher_query("MATCH (n) DELETE n")
    except Exception as e:
        pass
    yield
    # Cleanup after test
    try:
        db.cypher_query("MATCH ()-[r]-() DELETE r")
        db.cypher_query("MATCH (n) DELETE n")
    except Exception as e:
        pass

@pytest.fixture(autouse=True)
def setup_test_dishes(sample_dish_data: List[dict]):
    """Create test dishes before each test."""
    dishes = []
    for dish_data in sample_dish_data:
        dish = Dish(**dish_data)
        dish.save()
        dishes.append(dish)
    yield dishes
    # Cleanup happens automatically via setup_test_database fixture

def test_multi_dish_lookup_valid_types():
    """Test multi_dish_lookup with valid dish IDs."""
    # Test single dish ID
    result_obj = multi_dish_lookup.fn(["dish1"])
    result = result_obj.content[0].text  # Extract text from ToolResult
    structured = result_obj.structured_content  # Extract structured data

    # Check human-readable content
    assert "Bánh xèo" in result
    assert "145,000 vnđ" in result
    assert "Tìm thấy 1 món ăn" in result

    # Check structured content for detailed data
    assert len(structured["found_dishes"]) == 1
    dish_data = structured["found_dishes"][0]
    assert dish_data["dish_id"] == "dish1"
    assert dish_data["name_of_food"] == "Bánh xèo"
    assert dish_data["type_of_food"] == "món khai vị"
    assert "number_of_people_eating" in dish_data
    assert dish_data["number_of_people_eating"] == "2-3 người"

    # Test multiple dish IDs
    result_obj = multi_dish_lookup.fn(["dish1", "dish2"])
    result = result_obj.content[0].text  # Extract text from ToolResult
    structured = result_obj.structured_content  # Extract structured data

    assert "Bánh xèo" in result
    assert "Thịt kho tiêu" in result
    assert "Tìm thấy 2 món ăn" in result

    # Check structured content
    assert len(structured["found_dishes"]) == 2
    assert structured["total_found"] == 2
    assert structured["total_requested"] == 2

def test_multi_dish_lookup_invalid_id():
    """Test multi_dish_lookup with invalid dish ID."""
    result_obj = multi_dish_lookup.fn(["INVALID_ID"])
    result = result_obj.content[0].text
    structured = result_obj.structured_content

    assert "Không tìm thấy 1 món" in result
    assert "INVALID_ID" in result
    assert len(structured["missing_ids"]) == 1
    assert structured["missing_ids"][0] == "INVALID_ID"

def test_multi_dish_lookup_empty_list():
    """Test multi_dish_lookup with empty list."""
    result_obj = multi_dish_lookup.fn([])
    result = result_obj.content[0].text
    assert "Không có dish IDs nào được cung cấp" in result

def test_menu_value_count_and_price():
    """Test menu_value_count_and_price functionality."""
    result_obj = menu_value_count_and_price.fn()
    result = result_obj.content[0].text
    structured = result_obj.structured_content

    # Verify the menu includes all dishes with correct format
    assert "Menu hiện tại gồm:" in result
    assert "type 'khai vị'" in result.lower()
    assert "type 'thịt'" in result.lower()
    assert "Bánh xèo" in result
    assert "Thịt kho tiêu" in result

    # Check structured content
    assert "menu_groups" in structured
    assert len(structured["menu_groups"]) >= 2  # Should have at least khai vị and thịt
    assert structured["total_dishes"] >= 2
    assert structured["total_groups"] >= 2

    # Verify price formatting
    assert "145k" in result or "145,000" in result
    assert "150k" in result or "150,000" in result

# Removed test_hello_and_show_menu since that function doesn't exist

def test_empty_database():
    """Test all functions with empty database."""
    # Clear all dishes
    for dish in Dish.nodes.all():
        dish.delete()

    # Test each function
    result_obj = menu_value_count_and_price.fn()
    result = result_obj.content[0].text
    assert "Hiện không có dữ liệu món ăn" in result

    # Test multi_dish_lookup for empty database
    result_obj = multi_dish_lookup.fn([])
    result = result_obj.content[0].text
    assert "Không có dish IDs nào được cung cấp" in result


# ------------------------ Database Reset and Auto-loading Tests ------------------------

class TestDatabaseReset:
    """Test the clear_database_schema function functionality."""

    def test_clear_database_schema_success(self, setup_test_database):
        """Test successful database schema clearing."""
        # Create some test data with constraints/indexes
        Dish(dish_id="test_clear", type_of_food="TEST", name_of_food="Test Dish",
             current_price=100.0, combine_info="Test").save()

        # Verify data exists
        node_count_before = len(Dish.nodes.all())
        assert node_count_before > 0

        # Clear database schema
        result = clear_database_schema()
        assert result is True

        # Verify data is cleared
        # Use Dish.nodes.all() since direct cypher_query indexing is failing
        all_dishes_after = Dish.nodes.all()
        assert len(all_dishes_after) == 0

    def test_clear_database_schema_constraint_drop_failure(self, setup_test_database):
        """Test database clearing when constraint drop fails."""
        # Create test data
        Dish(dish_id="test_clear", type_of_food="TEST", name_of_food="Test Dish",
             current_price=100.0, combine_info="Test").save()

        # Mock APOC schema assert to fail
        with patch('src.mcp_server.db.cypher_query') as mock_query:
            # First call (drop constraints) fails
            mock_query.side_effect = [
                Exception("APOC not available"),  # First call fails
                [],  # Second call (get constraints) returns empty
                [],  # Third call (delete relationships) succeeds
                [],  # Fourth call (delete nodes) succeeds
                ([{'count': 0}], None),  # Fifth call (node count) succeeds
                ([{'count': 0}], None),  # Sixth call (rel count) succeeds
            ]

            result = clear_database_schema()
            assert result is True  # Should still succeed through fallback

    def test_clear_database_schema_data_deletion_failure(self, setup_test_database):
        """Test database clearing when data deletion fails."""
        # Create test data
        Dish(dish_id="test_clear", type_of_food="TEST", name_of_food="Test Dish",
             current_price=100.0, combine_info="Test").save()

        # Mock to fail at relationship deletion
        with patch('src.mcp_server.db.cypher_query') as mock_query:
            mock_query.side_effect = [
                ([], None),  # Drop constraints succeeds
                Exception("Deletion failed")  # Relationship deletion fails
            ]

            result = clear_database_schema()
            assert result is False

    def test_clear_database_schema_verification_failure(self, setup_test_database):
        """Test database clearing when verification fails."""
        # Create test data
        Dish(dish_id="test_clear", type_of_food="TEST", name_of_food="Test Dish",
             current_price=100.0, combine_info="Test").save()

        # Mock verification to return non-zero counts
        with patch('src.mcp_server.db.cypher_query') as mock_query:
            mock_query.side_effect = [
                ([], None),  # Drop constraints
                ([], None),  # Delete relationships
                ([], None),  # Delete nodes
                ([{'count': 1}], None),  # Node count shows 1 remaining
                ([{'count': 0}], None),  # Relationship count
            ]

            result = clear_database_schema()
            assert result is False


class TestAutoLoading:
    """Test the auto-loading functionality in lifespan."""

    @pytest.fixture
    def test_csv_file(self):
        """Create a temporary CSV file with test dish data."""
        csv_content = """_id,type_of_food,name_of_food,how_to_prepare,main_ingredients,taste,outstanding_fragrance,current_price,number_of_people_eating,combine_info
test_001,MÓN TEST,Test Dish 1,Test prep,Test ingredients,Test taste,Test fragrance,100,000,2-3 people,Test combined info
test_002,MÓN TEST,Test Dish 2,Test prep 2,Test ingredients 2,Test taste 2,Test fragrance 2,200,000,4-5 people,Test combined info 2
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
            f.write(csv_content)
            return f.name

    @pytest.mark.asyncio
    async def test_lifespan_auto_loading_enabled(self, test_csv_file):
        """Test lifespan with AUTO_LOAD_DISHES=true."""
        app = FastAPI()

        # Ensure no dishes exist initially
        for dish in Dish.nodes.all():
            dish.delete()

        with patch.dict(os.environ, {
            'AUTO_LOAD_DISHES': 'true',
            'FORCE_REFRESH_DISHES': 'true',
            'NEO4J_URI': 'bolt://localhost:7687',
            'NEO4J_USER': 'neo4j',
            'NEO4J_PASSWORD': '12345678'
        }):
            with patch('src.mcp_server.clear_database_schema', return_value=True) as mock_clear:
                with patch('src.models.order_graph.Dish.load_from_csv') as mock_load:
                    with patch('neo4j.GraphDatabase.driver') as mock_driver:
                        mock_driver_instance = MagicMock()
                        mock_driver_instance.close = AsyncMock(return_value=None)
                        mock_driver.return_value = mock_driver_instance
                        dishes_data = [
                            {'dish_id': 'test_001', 'name_of_food': 'Test Dish 1', 'current_price': 100000.0},
                            {'dish_id': 'test_002', 'name_of_food': 'Test Dish 2', 'current_price': 200000.0}
                        ]
                        mock_load.return_value = dishes_data

                        # Run lifespan
                        async with lifespan(app):
                            pass

                        # Verify database was cleared
                        mock_clear.assert_called_once()

                        # Verify dishes were loaded
                        mock_load.assert_called_once_with("src/data/comque_new_enriched.csv")

    @pytest.mark.asyncio
    async def test_lifespan_auto_loading_disabled(self):
        """Test lifespan with AUTO_LOAD_DISHES=false."""
        app = FastAPI()

        with patch.dict(os.environ, {
            'AUTO_LOAD_DISHES': 'false',
            'NEO4J_URI': 'bolt://localhost:7687',
            'NEO4J_USER': 'neo4j',
            'NEO4J_PASSWORD': '12345678'
        }):
            with patch('src.mcp_server.clear_database_schema') as mock_clear:
                with patch('src.models.order_graph.Dish.load_from_csv') as mock_load:
                    with patch('neo4j.GraphDatabase.driver') as mock_driver:
                        mock_driver_instance = MagicMock()
                        mock_driver_instance.close = AsyncMock(return_value=None)
                        mock_driver.return_value = mock_driver_instance
                        # Run lifespan
                        async with lifespan(app):
                            pass

                        # Verify database was NOT cleared
                        mock_clear.assert_not_called()

                        # Verify dishes were NOT loaded
                        mock_load.assert_not_called()

    @pytest.mark.asyncio
    async def test_lifespan_database_clear_failure(self):
        """Test lifespan when database clearing fails."""
        app = FastAPI()

        with patch.dict(os.environ, {
            'AUTO_LOAD_DISHES': 'true',
            'FORCE_REFRESH_DISHES': 'true',
            'NEO4J_URI': 'bolt://localhost:7687',
            'NEO4J_USER': 'neo4j',
            'NEO4J_PASSWORD': '12345678'
        }):
            with patch('src.mcp_server.clear_database_schema', return_value=False) as mock_clear:
                with patch('src.models.order_graph.Dish.load_from_csv') as mock_load:
                    with patch('neo4j.GraphDatabase.driver') as mock_driver:
                        mock_driver_instance = MagicMock()
                        mock_driver_instance.close = AsyncMock(return_value=None)
                        mock_driver.return_value = mock_driver_instance
                        # Run lifespan
                        async with lifespan(app):
                            pass

                        # Verify database clearing was attempted
                        mock_clear.assert_called_once()

                        # Verify dishes were still loaded despite cleanup failure
                        # (According to the implementation, it proceeds with loading even if cleanup fails)
                        mock_load.assert_called_once_with("src/data/comque_new_enriched.csv")

    @pytest.mark.asyncio
    async def test_lifespan_dish_loading_failure(self):
        """Test lifespan when dish loading fails."""
        app = FastAPI()

        with patch.dict(os.environ, {
            'AUTO_LOAD_DISHES': 'true',
            'FORCE_REFRESH_DISHES': 'true',
            'NEO4J_URI': 'bolt://localhost:7687',
            'NEO4J_USER': 'neo4j',
            'NEO4J_PASSWORD': '12345678'
        }):
            with patch('src.mcp_server.clear_database_schema', return_value=True):
                with patch('src.models.order_graph.Dish.load_from_csv', side_effect=Exception("CSV load failed")):
                    with patch('neo4j.GraphDatabase.driver') as mock_driver:
                        mock_driver_instance = MagicMock()
                        mock_driver_instance.close = AsyncMock(return_value=None)
                        mock_driver.return_value = mock_driver_instance
                        # Run lifespan - should not raise exception but log warning
                        async with lifespan(app):
                            pass

                        # Server should continue starting despite dish loading failure

    @pytest.mark.asyncio
    async def test_lifespan_neo4j_unavailable(self):
        """Test lifespan when Neo4j is unavailable during auto-loading."""
        app = FastAPI()

        with patch.dict(os.environ, {
            'AUTO_LOAD_DISHES': 'true',
            'NEO4J_URI': 'bolt://localhost:7687',
            'NEO4J_USER': 'neo4j',
            'NEO4J_PASSWORD': '12345678'
        }), patch.dict(os.environ, {'FORCE_REFRESH_DISHES': 'true'}):
            with patch('src.mcp_server.clear_database_schema', side_effect=Exception("Neo4j unavailable")):
                with patch('src.models.order_graph.Dish.load_from_csv', side_effect=Exception("Neo4j unavailable")):
                    with patch('neo4j.GraphDatabase.driver') as mock_driver:
                        mock_driver_instance = MagicMock()
                        mock_driver_instance.close = AsyncMock(return_value=None)
                        mock_driver.return_value = mock_driver_instance
                        # Run lifespan - should handle failures gracefully
                        async with lifespan(app):
                            pass

                        # Server should continue starting despite Neo4j connection issues

    @pytest.mark.asyncio
    async def test_environment_variable_default(self):
        """Test default AUTO_LOAD_DISHES behavior."""
        app = FastAPI()

        # Test with missing environment variable (should default to disabled)
        with patch.dict(os.environ, {}, clear=True):
            with patch('src.mcp_server.clear_database_schema') as mock_clear:
                with patch('src.models.order_graph.Dish.load_from_csv') as mock_load:
                    with patch('neo4j.GraphDatabase.driver') as mock_driver:
                        mock_driver_instance = MagicMock()
                        mock_driver_instance.close = AsyncMock(return_value=None)
                        mock_driver.return_value = mock_driver_instance
                        # Run lifespan
                        async with lifespan(app):
                            pass

                        # Verify auto-loading is disabled by default
                        mock_clear.assert_not_called()
                        mock_load.assert_not_called()


class TestIntegrationScenarios:
    """Test complete integration scenarios."""

    @pytest.mark.asyncio
    async def test_full_auto_loading_workflow(self):
        """Test the complete auto-loading workflow from start to finish."""
        # This would require full Neo4j setup, but we'll mock for unit testing
        with patch.dict(os.environ, {'AUTO_LOAD_DISHES': 'true', 'FORCE_REFRESH_DISHES': 'true'}):
            with patch('src.mcp_server.clear_database_schema', return_value=True) as mock_clear:
                with patch('src.models.order_graph.Dish.load_from_csv', return_value=[
                    {'dish_id': 'D001', 'name_of_food': 'Loaded Dish', 'current_price': 100000.0}
                ]) as mock_load:
                    with patch('neo4j.GraphDatabase.driver') as mock_driver:
                        mock_driver_instance = MagicMock()
                        mock_driver_instance.close = AsyncMock(return_value=None)
                        mock_driver.return_value = mock_driver_instance
                        app = FastAPI()

                        async with lifespan(app):
                            pass

                        mock_clear.assert_called_once()
                        mock_load.assert_called_once()

                        # Verify the OrderManager and Driver are attached
                        assert hasattr(app.state, 'driver')
                        assert hasattr(app.state, 'order_manager')
