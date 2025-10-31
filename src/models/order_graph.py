from neomodel import (ArrayProperty, BooleanProperty, DateTimeProperty,
                    FloatProperty, IntegerProperty, RelationshipTo,
                    StringProperty, StructuredNode, StructuredRel,
                    UniqueIdProperty, VectorIndex, db)
from typing import ClassVar, Dict, List, Optional, Union
import csv
import logging
import os
import uuid
import decimal
from decimal import Decimal

import pendulum
from dotenv import load_dotenv
from .config import RestaurantConfig

load_dotenv()


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Custom Exceptions


class OrderGraphError(Exception):
    """Base exception for all order graph errors."""
    pass


class ValidationError(OrderGraphError):
    """Raised when data validation fails."""
    pass


class DatabaseError(OrderGraphError):
    """Raised when database operations fail."""
    pass


class ResourceNotFoundError(OrderGraphError):
    """Raised when a requested resource is not found."""
    pass


class Placed(StructuredRel):
    """PLACED relationship between Customer and Order"""
    arrived_at = DateTimeProperty()  # When customer arrived at restaurant
    created_at = DateTimeProperty(default_now=True)  # When order was created

    def validate_datetime(self):
        """
        Validate datetime properties.
        - arrived_at must be provided
        - created_at defaults to current time

        Raises:
            ValidationError: If datetime validation fails
        """
        if not self.arrived_at:
            raise ValidationError("arrived_at datetime must be provided")

        if not self.created_at:
            self.created_at = pendulum.now("UTC")


class Contains(StructuredRel):
    """CONTAINS relationship between Order and Dish"""
    quantity = IntegerProperty(required=True)
    price = FloatProperty(required=True)  # Price at the time of order

    def validate(self):
        """
        Validate relationship properties.

        Raises:
            ValidationError: If validation fails
        """
        if self.quantity <= 0:
            raise ValidationError("Quantity must be positive")
        if self.price <= 0:
            raise ValidationError("Price must be positive")
        if not isinstance(self.quantity, int):
            raise ValidationError("Quantity must be an integer")


class Customer(StructuredNode):
    """Customer node representing restaurant patrons"""
    customer_id = StringProperty(unique_index=True)
    full_name = StringProperty(required=True)
    phone = ArrayProperty(StringProperty(), default=[])
    email = StringProperty()

    # Relationships
    placed = RelationshipTo('Order', 'PLACED', model=Placed)

    @classmethod
    def get_or_create(cls, customer_data: Dict) -> 'Customer':
        """
        Get existing customer or create new one.

        Args:
            customer_data (Dict): Customer data including:
                - customer_id (required)
                - full_name (required)
                - phone (optional)
                - email (optional)

        Returns:
            Customer: New or existing customer node

        Raises:
            ValidationError: If required data is missing
            DatabaseError: If database operation fails
        """
        try:
            if not customer_data.get('customer_id') or not customer_data.get('full_name'):
                raise ValidationError("customer_id and full_name are required")

            customer = cls.nodes.get_or_none(
                customer_id=customer_data['customer_id'])

            if customer is None:
                # Create new customer
                customer = cls(
                    customer_id=customer_data['customer_id'],
                    full_name=customer_data['full_name'],
                    phone=[customer_data['phone']] if customer_data.get(
                        'phone') else [],
                    email=customer_data.get('email')
                ).save()
                logger.info(
                    f"Created new customer: {customer_data['full_name']}")
            else:
                # Update existing customer
                customer.full_name = customer_data['full_name']
                if customer_data.get('email'):
                    customer.email = customer_data['email']
                if customer_data.get('phone') and customer_data['phone'] not in customer.phone:
                    customer.phone = list(
                        set(customer.phone + [customer_data['phone']]))
                customer.save()
                logger.info(
                    f"Updated existing customer: {customer_data['full_name']}")

            return customer

        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Error creating/updating customer: {str(e)}")
            raise DatabaseError(f"Failed to create/update customer: {str(e)}")

    def place_order(
        self,
        order: 'Order',
        arrived_at: 'pendulum.DateTime',
        dishes: Optional[List[Dict]] = None,
        table_id: Optional[int] = None,
        notes: Optional[str] = None
    ) -> 'Order':
        """
        Create an order and establish the PLACED relationship.

        Args:
            order (Order): The order to place
            arrived_at (pendulum.DateTime): When customer arrived
            dishes (Optional[List[Dict]]): List of dishes with id and quantity
            table_id (Optional[int]): Assigned table number
            notes (Optional[str]): Order notes

        Returns:
            Order: The created order

        Raises:
            ValidationError: If validation fails
            DatabaseError: If database operation fails
        """
        try:
            # Create PLACED relationship with timestamps
            # Fix: Pass relationship properties directly to connect() instead of passing the rel object
            self.placed.connect(order, {'arrived_at': arrived_at, 'created_at': pendulum.now("UTC")})

            # Add dishes if provided
            if dishes:
                prepared_dishes = order.validate_and_prepare_dishes(dishes)
                order.total_bill = order.calculate_total_bill(prepared_dishes)

                # Create CONTAINS relationships
                for dish_data in prepared_dishes:
                    dish = Dish.nodes.get(dish_id=dish_data['dish_id'])
                    order.items.connect(dish, {'quantity': dish_data['quantity'], 'price': dish_data['price']})

            logger.info(
                f"Created order {order.order_id} for customer {self.customer_id}")
            return order

        except (ValidationError, ResourceNotFoundError):
            raise
        except Exception as e:
            logger.error(f"Error placing order: {str(e)}")
            raise DatabaseError(f"Failed to place order: {str(e)}")


class Order(StructuredNode):
    """Order node representing customer orders"""
    order_id = StringProperty(unique_index=True, required=True)
    total_bill = FloatProperty(required=True)
    is_takeaway = BooleanProperty(default=False)
    is_pre_paid = BooleanProperty(default=False)
    table_id = IntegerProperty()  # None for takeaway
    notes = StringProperty()

    # Relationships
    items = RelationshipTo('Dish', 'CONTAINS', model=Contains)

    @classmethod
    def assign_table(cls, dt: 'pendulum.DateTime', config: 'RestaurantConfig') -> int:
        """
        Assign a free table based on the configuration settings.
        Checks for tables with no orders overlapping ±45 min based on created_at.

        Args:
            dt (pendulum.DateTime): The datetime to check for table availability
            config (RestaurantConfig): Restaurant configuration settings

        Returns:
            int: The assigned table number

        Raises:
            ValidationError: If no table is available in the time window
            DatabaseError: If database operation fails
        """
        try:
            dt_utc = dt.in_timezone("UTC")
            window_start = dt_utc - \
                pendulum.duration(minutes=config.TABLE_WINDOW_MINUTES)
            window_end = dt_utc + \
                pendulum.duration(minutes=config.TABLE_WINDOW_MINUTES)

            logger.info(
                f"Assigning table for time {dt_utc} (UTC) with window {window_start} to {window_end}")

            # Query orders with table_id and filter by created_at in PLACED relationship
            cypher = """
            MATCH (o:Order)<-[p:PLACED]-()
            WHERE o.table_id IS NOT NULL
            AND $start <= p.created_at <= $end
            RETURN o.table_id AS table_id
            """

            params = {
                "start": window_start.isoformat(),
                "end": window_end.isoformat()
            }

            results, _ = db.cypher_query(cypher, params)
            occupied_tables = {
                int(record[0]) for record in results if record[0] is not None}

            logger.info(f"Occupied tables: {occupied_tables}")

            # Find first available table
            for table_num in range(1, config.NUM_OF_TABLES + 1):
                if table_num not in occupied_tables:
                    logger.info(
                        f"Assigned table {table_num} for time {dt_utc}")
                    return table_num

            logger.warning(f"No free tables found for time {dt_utc}")
            raise ValidationError(
                f"No table free in the {config.TABLE_WINDOW_MINUTES * 2} minute window around {dt}")

        except ValidationError:
            raise
        except Exception as e:
            logger.error(
                f"Database error during table assignment for time {dt_utc}: {str(e)}")
            raise DatabaseError(f"Failed to assign table: {str(e)}")

    def validate_and_prepare_dishes(self, dishes: List[Dict]) -> List[Dict]:
        """
        Validate dishes and fetch their current prices.
        Allows empty dishes list for table reservations.

        Args:
            dishes (List[Dict]): List of dishes with 'dish_id' and 'quantity'

        Returns:
            List[Dict]: List of dishes with 'dish_id', 'quantity', and 'price'

        Raises:
            ValidationError: If validation fails
            ResourceNotFoundError: If any dish is not found
        """
        if not dishes:
            logger.info("No dishes provided; proceeding with empty dish list")
            return []

        try:
            # Validate format
            for dish in dishes:
                if not isinstance(dish, dict) or 'dish_id' not in dish or 'quantity' not in dish:
                    raise ValidationError(
                        "Each dish must have 'dish_id' and 'quantity'")
                if not isinstance(dish['quantity'], int) or dish['quantity'] <= 0:
                    raise ValidationError(
                        f"Invalid quantity for dish {dish['dish_id']}: {dish['quantity']}")

            # Fetch all dishes at once
            dish_ids = [d['dish_id'] for d in dishes]
            dish_nodes = Dish.nodes.filter(dish_id__in=dish_ids)

            # Create lookup dictionary
            dish_dict = {d.dish_id: d for d in dish_nodes}

            # Validate and prepare
            prepared_dishes = []
            missing_ids = []

            for dish in dishes:
                dish_id = dish['dish_id']
                if dish_id in dish_dict:
                    prepared_dishes.append({
                        'dish_id': dish_id,
                        'quantity': dish['quantity'],
                        'price': float(dish_dict[dish_id].current_price)
                    })
                else:
                    missing_ids.append(dish_id)

            if missing_ids:
                raise ResourceNotFoundError(
                    f"Dish IDs not found: {', '.join(missing_ids)}")

            return prepared_dishes

        except (ValidationError, ResourceNotFoundError):
            raise
        except Exception as e:
            logger.error(f"Error validating dishes: {str(e)}")
            raise DatabaseError(f"Failed to validate dishes: {str(e)}")

    def calculate_total_bill(self, prepared_dishes: List[Dict]) -> float:
        """
        Calculate total bill from prepared dishes.

        Args:
            prepared_dishes (List[Dict]): List of dishes with quantity and price

        Returns:
            float: Total bill amount
        """
        return sum(d['quantity'] * d['price'] for d in prepared_dishes)


class Dish(StructuredNode):
    """Dish node representing menu items loaded from CSV"""
    # Class level constants for CSV processing
    # For converting prices like "145,000" to float
    PRICE_MULTIPLIER: ClassVar[float] = 1000.0

    # Basic identifiers
    dish_id = StringProperty(unique_index=True, required=True)

    # Menu details
    type_of_food = StringProperty(required=True)
    name_of_food = StringProperty(required=True)
    how_to_prepare = StringProperty()
    main_ingredients = StringProperty()
    taste = StringProperty()
    outstanding_fragrance = StringProperty()

    # Pricing and serving
    current_price = FloatProperty(required=True)
    number_of_people_eating = StringProperty()

    # Combined information for search and embedding
    combine_info = StringProperty(fulltext_index=True)

    # Vector embedding for semantic search
    embedding = ArrayProperty(
        base_property=FloatProperty(),
        vector_index=VectorIndex(
            dimensions=int(os.getenv("EMBED_DIM", "1024")),
            similarity_function="cosine"
        )
    )

    @classmethod
    def convert_price(cls, price_str: str) -> float:
        """Convert price string (e.g. "145,000") to float."""
        try:
            # Remove commas and convert to Decimal for precision
            price_str = price_str.replace(",", "")
            return float(Decimal(price_str))
        except (ValueError, TypeError, decimal.InvalidOperation) as e:
            logger.error(f"Error converting price {price_str}: {str(e)}")
            raise ValidationError(f"Invalid price format: {price_str}")

    @classmethod
    def load_from_csv(cls, csv_path: str) -> List[Dict]:
        """
        Load dishes from CSV file and create/update Dish nodes.
        Returns list of created/updated dishes.

        Args:
            csv_path: Path to CSV file containing dish data

        Returns:
            List[Dict]: List of processed dishes with their data

        Raises:
            ValidationError: If CSV data validation fails
            DatabaseError: If database operations fail
        """
        try:
            results = []
            with open(csv_path, 'r', encoding='utf-8-sig') as f:  # utf-8-sig handles BOM automatically
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        # Convert and validate price
                        price = cls.convert_price(row['current_price'])

                        # Prepare dish data (handle BOM in column name if present)
                        dish_id = row.get('_id') or row.get('\ufeff_id')
                        if not dish_id:
                            logger.error("Could not find dish ID in row - available columns: {list(row.keys())}")
                            continue

                        dish_data = {
                            'dish_id': f"dish{dish_id}",
                            'type_of_food': row['type_of_food'].lower(),
                            'name_of_food': row['name_of_food'],
                            'how_to_prepare': row['how_to_prepare'],
                            'main_ingredients': row['main_ingredients'],
                            'taste': row['taste'],
                            'outstanding_fragrance': row['outstanding_fragrance'],
                            'current_price': price,
                            'number_of_people_eating': row['number_of_people_eating'],
                            'combine_info': row['combine_info']
                        }

                        # Create or update dish
                        dish = cls.nodes.get_or_none(
                            dish_id=dish_data['dish_id'])
                        if dish is None:
                            dish = cls(**dish_data).save()
                            logger.info(
                                f"Created new dish: {dish_data['name_of_food']}")
                        else:
                            for key, value in dish_data.items():
                                setattr(dish, key, value)
                            dish.save()
                            logger.info(
                                f"Updated existing dish: {dish_data['name_of_food']}")

                        results.append(dish_data)

                    except ValidationError as e:
                        logger.error(
                            f"Validation error processing row {row.get('_id', 'unknown')}: {str(e)}")
                        continue
                    except Exception as e:
                        logger.error(
                            f"Error processing row {row.get('_id', 'unknown')}: {str(e)}")
                        continue

            logger.info(
                f"Successfully processed {len(results)} dishes from CSV")
            return results

        except FileNotFoundError:
            logger.error(f"CSV file not found: {csv_path}")
            raise ResourceNotFoundError(f"CSV file not found: {csv_path}")
        except Exception as e:
            logger.error(f"Error loading dishes from CSV: {str(e)}")
            raise DatabaseError(f"Failed to load dishes from CSV: {str(e)}")
