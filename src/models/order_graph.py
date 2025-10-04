import os
from dotenv import load_dotenv
load_dotenv()

from neomodel import (
    StructuredNode, StructuredRel,
    StringProperty, IntegerProperty, FloatProperty, BooleanProperty,
    ArrayProperty, DateTimeProperty, UniqueIdProperty, RelationshipTo,
    VectorIndex
)
from typing import List, Optional


class Placed(StructuredRel):
    """PLACED relationship between Customer and Order"""
    arrived_at = DateTimeProperty()  # When customer arrived at restaurant
    created_at = DateTimeProperty(default_now=True)  # When order was created


class Contains(StructuredRel):
    """CONTAINS relationship between Order and Dish"""
    quantity = IntegerProperty(required=True)
    price = FloatProperty(required=True)  # Price at the time of order


class Customer(StructuredNode):
    """Customer node representing restaurant patrons"""
    customer_id = StringProperty(unique_index=True)
    full_name = StringProperty(required=True)
    phone = ArrayProperty(StringProperty(), default=[])
    email = StringProperty()

    # Relationships
    placed = RelationshipTo('Order', 'PLACED', model=Placed)

    def place_order(self, order: 'Order', arrived_at=None):
        """Create PLACED relationship with an order"""
        rel = Placed(arrived_at=arrived_at)
        self.placed.connect(order, rel)


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


class Dish(StructuredNode):
    """Dish node representing menu items loaded from CSV"""
    # Basic identifiers
    dish_id = StringProperty(unique_index=True, required=True)

    # Menu details
    type_of_food = StringProperty()
    name_of_food = StringProperty()
    how_to_prepare = StringProperty()
    main_ingredients = StringProperty()
    taste = StringProperty()
    outstanding_fragrance = StringProperty()

    # Pricing and serving
    current_price = FloatProperty()
    number_of_people_eating = StringProperty()

    # Combined information for search and embedding
    combine_info = StringProperty(fulltext_index=True)

    # Vector embedding for semantic search (similar to Task model)
    embedding = ArrayProperty(
        base_property=FloatProperty(),
        vector_index=VectorIndex(
            dimensions=int(os.getenv("EMBED_DIM", "1024")),
            similarity_function="cosine"
        )
    )
