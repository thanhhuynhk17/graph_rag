import uuid
from typing import Optional, List, Dict
from neo4j import AsyncDriver, RoutingControl
import pendulum
from datetime import timedelta
import logging
from neo4j.exceptions import Neo4jError

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

class OrderManager:
    """Manages restaurant orders in a Neo4j knowledge graph using inlined Cypher queries."""
    
    def __init__(self,
                driver: AsyncDriver
        ):
        self.driver = driver
        
    async def initialize(self):
        """Initialize constraints and indexes for Customer, Dish, and Order nodes.
        
        Args:
            refresh_database (bool): If True, deletes all nodes and relationships before creating constraints and indexes. Use with caution.
        """
        raise NotImplementedError("The 'initialize' method is not implemented in this snippet.")
    
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
            start = (dt_utc - timedelta(minutes=45)).isoformat()
            end = (dt_utc + timedelta(minutes=45)).isoformat()
            logger.info(f"{namespace}: Assigning table for time {dt_utc} (UTC) with window {start} to {end}")

            cypher = (
                f"MATCH (o:Order)\n"
                "WHERE o.table_id IS NOT NULL\n"
                "AND o.created_at IS NOT NULL\n"
                "AND $start <= datetime(o.created_at) <= $end\n"
                "RETURN o.table_id AS table_id"
            )
            result = await self.driver.execute_query(
                cypher,
                {"start": start, "end": end},
                routing_control=RoutingControl.READ
            )
            
            occupied = {int(record["table_id"]) for record in result.records if record["table_id"] is not None}
            logger.info(f"{namespace}: Occupied tables: {occupied}")
            NUM_OF_TABLES = 2
            for t in range(1, NUM_OF_TABLES + 1):
                if t not in occupied:
                    logger.info(f"{namespace}: Assigned table {t} for time {dt_utc}")
                    return t
            
            logger.warning(f"{namespace}: No free tables found for time {dt_utc}")
            raise ValueError("No table free in the 90-minute window around the provided time.")
        
        except ValueError as e:
            logger.error(f"{namespace}: Table assignment failed: {str(e)}")
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

            dish_data = {}
            for d in dishes:
                if "id" not in d or "quantity" not in d:
                    logger.error(f"{namespace}: Invalid dish format: {d}")
                    raise ValueError("Each dish must have 'id' and 'quantity'.")
                if not isinstance(d["quantity"], int) or d["quantity"] <= 0:
                    logger.error(f"{namespace}: Invalid quantity for dish {d['id']}: {d['quantity']}")
                    raise ValueError(f"Quantity must be a positive integer for dish {d['id']}.")
                dish_data[d["id"]] = d["quantity"]

            dish_ids = list(dish_data.keys())
            logger.info(f"{namespace}: Validating dishes: {dish_ids}")

            cypher = """
            UNWIND $dish_ids AS id
            OPTIONAL MATCH (d:Dish {_id: id})
            RETURN id, d.price AS price
            """
            result = await self.driver.execute_query(
                cypher,
                {"dish_ids": dish_ids},
                routing_control=RoutingControl.READ
            )

            fetched_prices = {record["id"]: record["price"] for record in result.records}
            missing_ids = [id for id in dish_ids if fetched_prices.get(id) is None]
            if missing_ids:
                logger.error(f"{namespace}: Dish IDs not found: {missing_ids}")
                raise ValueError(f"Dish IDs not found: {', '.join(missing_ids)}")

            prepared_dishes = [
                {"id": id, "quantity": int(dish_data[id]), "price": float(fetched_prices[id])}
                for id in dish_ids
            ]
            logger.info(f"{namespace}: Dishes validated and prepared successfully: {len(prepared_dishes)} dishes")
            return prepared_dishes

        except ValueError as e:
            logger.error(f"{namespace}: Validation error for dishes: {str(e)}")
            raise ValueError(f"Failed to validate and prepare dishes: {str(e)}")
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
            dt_utc = dt.in_timezone("UTC").isoformat()
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

            # Create or update Customer node
            customer_cypher = """
            MERGE (c:Customer {_id: $guest_id})
            ON CREATE SET 
                c.full_name = $full_name,
                c.phone = CASE WHEN $phone IS NULL THEN [] ELSE [$phone] END,
                c.email = $email,
                c.notes = $notes
            ON MATCH SET 
                c.full_name = $full_name,
                c.phone = CASE 
                            WHEN $phone IS NULL THEN c.phone 
                            ELSE apoc.coll.toSet(c.phone + [$phone]) 
                        END,
                c.email = CASE WHEN $email IS NULL THEN c.email ELSE $email END,
            """
            try:
                await self.driver.execute_query(
                    customer_cypher,
                    {
                        "guest_id": guest_id,
                        "full_name": guest_name,
                        "phone": guest_phone_number,
                        "email": email
                    },
                    routing_control=RoutingControl.WRITE
                )
                logger.info(f"{namespace}: Customer {guest_id} created/updated successfully")
            except Neo4jError as e:
                logger.error(f"{namespace}: Database error creating/updating customer {guest_id}: {str(e)}")
                raise Exception(f"Failed to create/update customer: {str(e)}")

            # Create Order node and relationships (skip dish relationships if no dishes)
            order_id = str(uuid.uuid4())
            order_cypher = """
            MATCH (c:Customer {_id: $customer_id})
            CREATE (o:Order {
                _id: $order_id,
                total_bill: $total_bill,
                is_takeaway: $is_takeaway,
                is_pre_paid: $is_pre_paid,
                table_id: $table_id,
                created_at: $created_at
            })
            MERGE (c)-[:PLACED]->(o)
            """
            params = {
                "customer_id": guest_id,
                "order_id": order_id,
                "total_bill": total_cost,
                "is_takeaway": is_takeaway,
                "is_pre_paid": False,
                "table_id": table_id,
                "created_at": dt_utc
            }
            
            if prepared_dishes:
                order_cypher += """
                WITH o
                UNWIND $dishes AS d
                MATCH (dish:Dish {_id: d.id})
                MERGE (o)-[r:CONTAINS]->(dish)
                SET r.quantity = d.quantity, r.price = d.price
                """
                params["dishes"] = prepared_dishes
                logger.info(f"{namespace}: Adding {len(prepared_dishes)} CONTAINS relationships with quantities: {[d['quantity'] for d in prepared_dishes]}")
            
            order_cypher += """
            RETURN elementId(o) AS order_neo4j_id, o._id AS order_id
            """
            
            try:
                result = await self.driver.execute_query(
                    order_cypher,
                    params,
                    routing_control=RoutingControl.WRITE
                )
                logger.info(f"{namespace}: Order {order_id} created successfully for guest {guest_id}")
                return order_id, table_id
            except Neo4jError as e:
                logger.error(f"{namespace}: Database error creating order {order_id} for guest {guest_id}: {str(e)}")
                raise Exception(f"Failed to create order: {str(e)}")
        
        except ValueError as e:
            logger.error(f"{namespace}: Validation error for guest {guest_id}: {str(e)}")
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
        cypher = """
        MERGE (d:Dish {_id: $id})
        SET d.price = $price
        """
        dishes = [
            {"id": "dish1", "price": 100_000},
            {"id": "dish2", "price": 50_000}
        ]
        for dish in dishes:
            await driver.execute_query(
                cypher,
                dish,
                routing_control=RoutingControl.WRITE
            )
        logger.info(f"{namespace}: Test dishes created successfully")
    except Neo4jError as e:
        logger.error(f"{namespace}: Failed to create test dishes: {str(e)}")
        raise

async def demo():
    namespace = "Demo|demo"
    driver = AsyncGraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "12345678"))  # Replace with your credentials
    manager = OrderManager(driver)
    await manager.initialize(refresh_database=True)  # Use with caution: Refreshes the database
    
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
    # finally:
    #     await manager.close()  # Commented out as close method is not defined

if __name__ == "__main__":
    import asyncio
    asyncio.run(demo())
