import uuid
from typing import Optional, List, Dict
from neo4j import Driver
import pendulum
from datetime import timedelta
import logging
from neo4j.exceptions import Neo4jError
from neomodel import install_labels, db

# Import models
from .order_graph import Customer, Order, Dish, Placed, Contains
from .config import RestaurantConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('order_manager.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
config = RestaurantConfig()

class OrderManager:
    """Manages restaurant orders using neomodel ORM."""
    
    def __init__(self, driver: Driver):
        self.driver = driver
        # Set the database connection for neomodel
        from neomodel import config as neomodel_config
        # Note: In a real application, you'd configure this properly
        # For now, we'll use the driver directly with neomodel queries
        
    async def initialize(self):
        """Initialize database constraints and indexes."""
        try:
            # Install labels and constraints for all models
            install_labels(Customer)
            install_labels(Order)
            install_labels(Dish)
            install_labels(Placed)
            install_labels(Contains)
            logger.info("Database constraints and indexes initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize database constraints: {str(e)}")
            raise Exception(f"Database initialization error: {str(e)}")

    async def assign_table(self, dt: pendulum.DateTime) -> int:
        """
        Assign a free table (1–2) with no orders overlapping ±45 min based on created_at.
        Uses trigger-generated created_at timestamp. Datetime input is in Asia/Ho_Chi_Minh timezone,
        converted to UTC for Neo4j.
        
        Args:
            dt (pendulum.DateTime): The datetime in Asia/Ho_Chi_Minh timezone.
        
        Returns:
            int: The assigned table number.
        
        Raises:
            ValueError: If no table is free in the 90-minute window.
            Exception: For database or unexpected errors.
        """
        namespace = "OrderManager|assign_table"
        try:
            dt_utc = dt.in_timezone("UTC")
            start = dt_utc - timedelta(minutes=config.TABLE_WINDOW_MINUTES)
            end = dt_utc + timedelta(minutes=config.TABLE_WINDOW_MINUTES)
            logger.info(f"{namespace}: Assigning table for time {dt_utc} (UTC) with window {start} to {end}")

            # Query orders with table_id and filter by created_at in PLACED relationship
            # Note: Neomodel doesn't directly support filtering on relationship properties
            # So we'll use a hybrid approach with a single optimized query
            
            cypher = """
            MATCH (o:Order)<-[p:PLACED]-()
            WHERE o.table_id IS NOT NULL
            AND $start <= p.created_at <= $end
            RETURN o.table_id AS table_id
            """
            
            result = self.driver.execute_query(
                cypher,
                {"start": start.isoformat(), "end": end.isoformat()},
                routing_control="READ"
            )
            
            occupied = {int(record["table_id"]) for record in result.records if record["table_id"] is not None}
            logger.info(f"{namespace}: Occupied tables: {occupied}")
            
            for t in range(1, config.NUM_OF_TABLES + 1):
                if t not in occupied:
                    logger.info(f"{namespace}: Assigned table {t} for time {dt_utc}")
                    return t
            
            logger.warning(f"{namespace}: No free tables found for time {dt_utc}")
            raise ValueError("No table free in the 90-minute window around the provided time.")
        
        except ValueError:
            raise
        except Neo4jError as e:
            logger.error(f"{namespace}: Database error during table assignment for time {dt_utc}: {str(e)}")
            raise Exception(f"Database error while assigning table: {str(e)}")
        except Exception as e:
            logger.error(f"{namespace}: Unexpected error during table assignment for time {dt_utc}: {str(e)}")
            raise Exception(f"Unexpected error during table assignment: {str(e)}")

    async def validate_and_prepare_dishes(self, dishes: List[Dict]) -> List[Dict]:
        """
        Validate dishes by checking if each dish ID exists in the database and fetch their prices.
        Allows empty dishes list (e.g., for table reservation without ordering).
        
        Args:
            dishes (List[Dict]): List of dishes with 'id' and 'quantity', or empty list.
        
        Returns:
            List[Dict]: List of dishes with 'id', 'quantity', and 'price', or empty list.
        
        Raises:
            ValueError: If validation fails (e.g., invalid quantity, dish not found).
            Exception: For database or unexpected errors.
        """
        namespace = "OrderManager|validate_and_prepare_dishes"
        try:
            if not dishes:  # Allow empty dishes list
                logger.info(f"{namespace}: No dishes provided; proceeding with empty dish list")
                return []

            # Validate dish format
            for d in dishes:
                if "id" not in d or "quantity" not in d:
                    logger.error(f"{namespace}: Invalid dish format: {d}")
                    raise ValueError("Each dish must have 'id' and 'quantity'.")
                if not isinstance(d["quantity"], int) or d["quantity"] <= 0:
                    logger.error(f"{namespace}: Invalid quantity for dish {d['id']}: {d['quantity']}")
                    raise ValueError(f"Quantity must be a positive integer for dish {d['id']}.")

            # Fetch all dishes at once for efficiency
            dish_ids = [d["id"] for d in dishes]
            logger.info(f"{namespace}: Validating dishes: {dish_ids}")

            # Use neomodel to fetch dishes
            dish_nodes = Dish.nodes.filter(dish_id__in=dish_ids)
            
            # Create lookup dictionary
            dish_dict = {dish.dish_id: dish for dish in dish_nodes}
            
            # Validate all dishes exist and prepare data
            prepared_dishes = []
            missing_ids = []
            
            for d in dishes:
                dish_id = d["id"]
                if dish_id in dish_dict:
                    prepared_dishes.append({
                        "id": dish_id,
                        "quantity": d["quantity"],
                        "price": float(dish_dict[dish_id].current_price or 0.0)
                    })
                else:
                    missing_ids.append(dish_id)
            
            if missing_ids:
                logger.error(f"{namespace}: Dish IDs not found: {missing_ids}")
                raise ValueError(f"Dish IDs not found: {', '.join(missing_ids)}")

            logger.info(f"{namespace}: Dishes validated and prepared successfully: {len(prepared_dishes)} dishes")
            return prepared_dishes

        except ValueError:
            raise
        except Neo4jError as e:
            logger.error(f"{namespace}: Database error during dish validation: {str(e)}")
            raise Exception(f"Database error while validating dishes: {str(e)}")
        except Exception as e:
            logger.error(f"{namespace}: Unexpected error during dish validation: {str(e)}")
            raise Exception(f"Unexpected error during dish validation: {str(e)}")

    async def create_order(
        self,
        guest_id: str,
        guest_name: str,
        guest_phone_number: str,
        is_takeaway: bool,
        dishes: List[Dict],  # [{"id": str, "quantity": int}] or empty list
        dt: pendulum.DateTime,
        notes: Optional[str] = None,
        email: Optional[str] = None
    ) -> tuple[str, Optional[int]]:
        """
        Create a Customer, Order, and relationships with PLACED and CONTAINS.
        Customer is PLACED Order. Order CONTAINS Dish with quantity and price.
        Uses provided datetime in Asia/Ho_Chi_Minh timezone, converted to UTC for Neo4j.
        Allows empty dishes list for table reservation without ordering.
        
        Args:
            guest_id (str): Unique identifier for the guest.
            guest_name (str): Name of the guest.
            guest_phone_number (str): Guest's phone number.
            is_takeaway (bool): Whether the order is for takeaway.
            dishes (List[Dict]): List of dishes with id and quantity, or empty list.
            dt (pendulum.DateTime): Order creation time in Asia/Ho_Chi_Minh timezone.
            notes (Optional[str]): Optional notes for the order.
            email (Optional[str]): Optional email for the customer.
        
        Returns:
            tuple[str, Optional[int]]: Order ID and assigned table ID (None for takeaway).
        
        Raises:
            ValueError: If input validation fails or no table is available.
            Exception: For database or unexpected errors.
        """
        namespace = "OrderManager|create_order"
        try:
            # Validate inputs
            if not guest_id or not guest_name or not guest_phone_number:
                logger.error(f"{namespace}: Invalid input: guest_id, guest_name, or guest_phone_number is empty")
                raise ValueError("Guest ID, name, and phone number are required.")
            
            # Handle datetime
            dt_utc = dt.in_timezone("UTC")
            logger.info(f"{namespace}: Creating order for guest {guest_id} at {dt_utc} (UTC)")

            # Validate and prepare dishes (fetch prices if dishes provided)
            prepared_dishes = await self.validate_and_prepare_dishes(dishes)
            
            # Calculate total cost
            total_cost = 0.0
            if prepared_dishes:
                total_cost = sum(d["quantity"] * d["price"] for d in prepared_dishes)
                logger.info(f"{namespace}: Calculated total_cost for guest {guest_id}: {total_cost}")
            else:
                logger.info(f"{namespace}: No dishes provided for guest {guest_id}; setting total_cost to 0")

            # Assign table if not takeaway
            table_id = None
            if not is_takeaway:
                try:
                    table_id = await self.assign_table(dt)
                    logger.info(f"{namespace}: Assigned table {table_id} for order")
                except ValueError as e:
                    logger.error(f"{namespace}: Failed to assign table for guest {guest_id}: {str(e)}")
                    raise ValueError(f"Failed to assign table: {str(e)}")
                except Exception as e:
                    logger.error(f"{namespace}: Unexpected error assigning table for guest {guest_id}: {str(e)}")
                    raise Exception(f"Failed to assign table: {str(e)}")

            # Get or create customer using neomodel
            try:
                customer = Customer.nodes.get_or_none(customer_id=guest_id)
                if customer is None:
                    customer = Customer(
                        customer_id=guest_id,
                        full_name=guest_name,
                        phone=[guest_phone_number] if guest_phone_number else [],
                        email=email
                    ).save()
                    logger.info(f"{namespace}: Created new customer {guest_id}")
                else:
                    # Update existing customer
                    customer.full_name = guest_name
                    if email:
                        customer.email = email
                    # Merge phone numbers
                    if guest_phone_number and guest_phone_number not in customer.phone:
                        customer.phone = list(set(customer.phone + [guest_phone_number]))
                    customer.save()
                    logger.info(f"{namespace}: Updated existing customer {guest_id}")
            except Neo4jError as e:
                logger.error(f"{namespace}: Database error creating/updating customer {guest_id}: {str(e)}")
                raise Exception(f"Failed to create/update customer: {str(e)}")

            # Create order using neomodel
            order_id = str(uuid.uuid4())
            try:
                order = Order(
                    order_id=order_id,
                    total_bill=total_cost,
                    is_takeaway=is_takeaway,
                    is_pre_paid=False,
                    table_id=table_id,
                    notes=notes
                ).save()
                logger.info(f"{namespace}: Created order {order_id}")
            except Neo4jError as e:
                logger.error(f"{namespace}: Database error creating order {order_id}: {str(e)}")
                raise Exception(f"Failed to create order: {str(e)}")

            # Create PLACED relationship with timestamps
            try:
                rel = Placed(arrived_at=dt, created_at=dt_utc)
                customer.placed.connect(order, rel)
                logger.info(f"{namespace}: Created PLACED relationship between customer {guest_id} and order {order_id}")
            except Exception as e:
                logger.error(f"{namespace}: Error creating PLACED relationship: {str(e)}")
                raise Exception(f"Failed to create customer-order relationship: {str(e)}")

            # Create CONTAINS relationships for dishes
            if prepared_dishes:
                try:
                    for dish_data in prepared_dishes:
                        dish = Dish.nodes.get(dish_id=dish_data["id"])
                        contains_rel = Contains(
                            quantity=dish_data["quantity"],
                            price=dish_data["price"]
                        )
                        order.items.connect(dish, contains_rel)
                    logger.info(f"{namespace}: Created {len(prepared_dishes)} CONTAINS relationships")
                except Exception as e:
                    logger.error(f"{namespace}: Error creating CONTAINS relationships: {str(e)}")
                    raise Exception(f"Failed to create dish relationships: {str(e)}")

            logger.info(f"{namespace}: Order {order_id} created successfully for guest {guest_id}")
            return order_id, table_id
        
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"{namespace}: Unexpected error creating order for guest {guest_id}: {str(e)}")
            raise Exception(f"Unexpected error creating order: {str(e)}")

# ------------------------------------------------------------------
# Quick demo
# ------------------------------------------------------------------
from neo4j import AsyncGraphDatabase
import pendulum

async def create_test_dishes(driver):
    """Create test Dish nodes for the demo."""
    namespace = "Demo|create_test_dishes"
    try:
        # Use neomodel to create test dishes
        dishes_data = [
            {"dish_id": "dish1", "current_price": 100000.0, "name_of_food": "Test Dish 1"},
            {"dish_id": "dish2", "current_price": 50000.0, "name_of_food": "Test Dish 2"}
        ]
        
        for dish_data in dishes_data:
            dish = Dish.nodes.get_or_none(dish_id=dish_data["dish_id"])
            if dish is None:
                Dish(**dish_data).save()
        
        logger.info(f"{namespace}: Test dishes created successfully")
    except Exception as e:
        logger.error(f"{namespace}: Failed to create test dishes: {str(e)}")
        raise

async def demo():
    namespace = "Demo|demo"
    driver = AsyncGraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "12345678"))  # Replace with your credentials
    manager = OrderManager(driver)
    await manager.initialize()  # Initialize constraints
    
    try:
        # Create test dishes
        await create_test_dishes(driver)
        
        # Set datetime for 02:09 AM +07, September 30, 2025
        dt = pendulum.datetime(2025, 9, 30, 2, 9, tz="Asia/Ho_Chi_Minh")
        
        # Order 1: Dine-in with dishes
        order_id1, table_id1 = await manager.create_order(
            guest_id="guest1",
            guest_name="Early Bird",
            guest_phone_number="0900000001",
            is_takeaway=False,
            dishes=[{"id": "dish1", "quantity": 1}, {"id": "dish2", "quantity": 2}],
            dt=dt,
            email="early@bird.com",
            notes="No spicy food"
        )
        print(f"First order ID: {order_id1}, Table: {table_id1}")  # Expected table: 1
        
        # Order 2: Dine-in with dishes, close in time (should get different table)
        order_id2, table_id2 = await manager.create_order(
            guest_id="guest2",
            guest_name="Overlap",
            guest_phone_number="0900000002",
            is_takeaway=False,
            dishes=[{"id": "dish1", "quantity": 1}],
            dt=dt,
            email="overlap@bird.com",
            notes="Extra napkins"
        )
        print(f"Second order ID: {order_id2}, Table: {table_id2}")  # Expected table: 2
        
        # Order 3: Takeaway, no table
        order_id3, table_id3 = await manager.create_order(
            guest_id="guest3",
            guest_name="Takeaway",
            guest_phone_number="0900000003",
            is_takeaway=True,
            dishes=[{"id": "dish2", "quantity": 1}],
            dt=dt,
            email="takeaway@bird.com",
            notes="Pack quickly"
        )
        print(f"Takeaway order ID: {order_id3}, Table: {table_id3}")  # Expected table: None
        
        # Order 4: Table reservation without dishes
        order_id4, table_id4 = await manager.create_order(
            guest_id="guest4",
            guest_name="Table Only",
            guest_phone_number="0900000004",
            is_takeaway=False,
            dishes=[],
            dt=dt,
            email="table@bird.com",
            notes="Reserve for meeting"
        )
        print(f"Table-only order ID: {order_id4}, Table: {table_id4}")  # Expected table: 1 or 2
    
    except Exception as e:
        logger.error(f"{namespace}: Demo failed: {str(e)}")
        print(f"Demo failed: {str(e)}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(demo())
