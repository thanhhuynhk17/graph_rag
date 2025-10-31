# Neomodel Documentation - Complete Reference

## Overview
Neomodel is an Object Graph Mapper (OGM) for the Neo4j graph database, built on the neo4j_driver. It provides familiar Django-style model definitions with powerful query APIs, schema enforcement through cardinality restrictions, full transaction support, and thread safety.

**Key Features:**
- Django model style definitions
- Powerful query API
- Schema enforcement via cardinality restrictions
- Full transaction support
- Thread safe
- Async support
- Pre/post save/delete hooks
- Django integration via django_neomodel

## Requirements & Installation

### Requirements
**For releases 5.x:**
- Python 3.7+
- Neo4j 5.x, 4.4 (LTS)

**For releases 4.x:**
- Python 3.7 → 3.10
- Neo4j 4.x (including 4.4 LTS for neomodel version 4.0.10)

### Installation
```bash
# From PyPI (recommended)
pip install neomodel

# From GitHub
pip install git+git://github.com/neo4j-contrib/neomodel.git@HEAD#egg=neomodel-dev
```

### Breaking Changes in 5.3
- `config.AUTO_INSTALL_LABELS` removed - use `neomodel_install_labels` command instead
- Database class moved to `neomodel.sync_.core` - new AsyncDatabase in `neomodel.async_.core`

### Deprecations in 5.3
Standalone methods moved to Database() class (to be removed in future release):
- `change_neo4j_password`
- `clear_neo4j_database`
- `drop_constraints`
- `drop_indexes`
- `remove_all_labels`
- `install_labels`
- `install_all_labels`

For async calls, use methods in AsyncDatabase() singleton.

## Getting Started

### Connecting to Neo4j
```python
from neomodel import config
config.DATABASE_URL = 'bolt://neo4j_username:neo4j_password@localhost:7687'
```

### Direct Cypher Queries
```python
from neomodel import db
results, meta = db.cypher_query("RETURN 'Hello World' as message")
```

### Database Inspection (Requires APOC)
```bash
# Generate neomodel definitions from existing database
neomodel_inspect_database --db bolt://neo4j:password@localhost:7687 --write-to yourapp/models.py
```

### Installing Labels and Constraints
```bash
# Apply schema constraints and indexes
neomodel_install_labels yourapp.py someapp.models --db bolt://neo4j:password@localhost:7687

# Remove all constraints and indexes
neomodel_remove_labels --db bolt://neo4j:password@localhost:7687
```

### Generate Class Diagrams
```bash
neomodel_generate_diagram models/my_models.py --file-type arrows --write-to-dir img
```

## Core Concepts

### Node Entities
```python
from neomodel import StructuredNode, StringProperty, IntegerProperty, UniqueIdProperty

class Country(StructuredNode):
    code = StringProperty(unique_index=True, required=True)

class City(StructuredNode):
    name = StringProperty(required=True)
    country = RelationshipTo(Country, 'FROM_COUNTRY')

class Person(StructuredNode):
    uid = UniqueIdProperty()
    name = StringProperty(unique_index=True)
    age = IntegerProperty(index=True, default=0)

    # traverse outgoing IS_FROM relations
    country = RelationshipTo(Country, 'IS_FROM')
    # traverse outgoing LIVES_IN relations
    city = RelationshipTo(City, 'LIVES_IN')
```

### Create, Update, Delete Operations
```python
# Create
jim = Person(name='Jim', age=3).save()

# Update
jim.age = 4
jim.save()  # with validation

# Delete
jim.delete()

# Refresh from database
jim.refresh()

# Get Neo4j internal element ID
element_id = jim.element_id
```

### Retrieving Nodes
```python
# Get all nodes
all_nodes = Person.nodes.all()

# Get specific node (raises DoesNotExist if no match)
jim = Person.nodes.get(name='Jim')

# Get or None
someone = Person.nodes.get_or_none(name='bob')

# First match
someone = Person.nodes.first(name='bob')
someone = Person.nodes.first_or_none(name='bob')

# Filter nodes
people = Person.nodes.filter(age__gte=3)

# Lazy loading (returns IDs only)
all_nodes_lazy = Person.nodes.all(lazy=True)
```

### Iteration and Slicing
```python
# Iterable
for person in Person.nodes:
    print(person.name)

# Sliceable
people_slice = Person.nodes.filter(age__gte=2)[2:]

# Length
count = len(Person.nodes.filter(age__gte=2))

# Boolean check
if Person.nodes:
    print("We have person nodes!")
```

## Relationships

### Basic Relationships
```python
from neomodel import Relationship, RelationshipTo, RelationshipFrom

class Person(StructuredNode):
    name = StringProperty()
    # Undirected relationship
    friends = Relationship('Person', 'FRIEND')

    # Directed relationships
    spouse = RelationshipTo('Person', 'MARRIED_TO')
    children = RelationshipTo('Person', 'PARENT_OF')
    parents = RelationshipFrom('Person', 'PARENT_OF')
```

### Working with Relationships
```python
# Create nodes
germany = Country(code='DE').save()
jim = Person(name='Jim').save()
berlin = City(name='Berlin').save()

# Connect relationships
jim.country.connect(germany)
berlin.country.connect(germany)
jim.city.connect(berlin)

# Check connections
if jim.country.is_connected(germany):
    print("Jim's from Germany")

# Traverse relationships
for person in germany.inhabitant.all():
    print(person.name)

# Count relationships
len(germany.inhabitant)  # 1

# Filter relationships
germany.inhabitant.filter(name='Jim')

# Exclude relationships
germany.inhabitant.exclude(name='Jim')

# Disconnect
jim.country.disconnect(germany)

# Connect multiple
usa = Country(code='US').save()
jim.country.connect(usa)
jim.country.connect(germany)

# Disconnect all
jim.country.disconnect_all()

# Replace relationship
jim.country.replace(germany)
```

### Cardinality Constraints
```python
from neomodel import One, ZeroOrOne, ZeroOrMore, OneOrMore

class Person(StructuredNode):
    # One-to-one
    car = RelationshipTo('Car', 'OWNS', cardinality=One)

class Car(StructuredNode):
    # One-to-one (inverse)
    owner = RelationshipFrom('Person', 'OWNS', cardinality=One)

# Available cardinalities:
# ZeroOrOne  - 0 or 1
# One        - exactly 1
# ZeroOrMore - 0 or more (default)
# OneOrMore  - 1 or more
```

### Relationship Properties
```python
from neomodel import StructuredRel, DateTimeProperty, StringProperty

class FriendRel(StructuredRel):
    since = DateTimeProperty(default=lambda: datetime.now(ZoneInfo("UTC")), index=True)
    met = StringProperty()
    meeting_id = StringProperty(unique_index=True)  # Neo4j 5.7+

class Person(StructuredNode):
    name = StringProperty()
    friends = RelationshipTo('Person', 'FRIEND', model=FriendRel)

# Connect with properties
rel = jim.friends.connect(bob, {'since': yesterday, 'met': 'Paris'})

# Access properties
print(rel.since)
print(rel.start_node().name)  # jim
print(rel.end_node().name)    # bob

# Update properties
rel.met = "Amsterdam"
rel.save()

# Get relationship between specific nodes
rel = jim.friends.relationship(bob)
```

### Relationship Inheritance
```python
class PersonalRelationship(StructuredRel):
    on_date = DateProperty(default_now=True)

class PersonalRelationshipWithStrength(PersonalRelationship):
    strength = FloatProperty(default=1.0)

class BasePerson(StructuredNode):
    name = StringProperty(required=True, unique_index=True)
    friends_with = RelationshipTo("BasePerson", "FRIENDS_WITH", model=PersonalRelationship)

class ExtendedBasePerson(BasePerson):
    role = StringProperty(required=True)
    friends_with = RelationshipTo("BasePerson", "FRIENDS_WITH", model=PersonalRelationshipWithStrength)
```

### Node Inheritance with Relationships
```python
class BasePerson(StructuredNode):
    name = StringProperty(required=True, unique_index=True)
    friends_with = RelationshipTo("BasePerson", "FRIENDS_WITH", model=PersonalRelationship)

class TechnicalPerson(BasePerson):
    expertise = StringProperty(required=True)

class PilotPerson(BasePerson):
    airplane = StringProperty(required=True)

# Works with inheritance - relationships resolve to correct types
tech_person = TechnicalPerson(name="Alice", expertise="AI").save()
pilot_person = PilotPerson(name="Bob", airplane="Boeing").save()

tech_person.friends_with.connect(pilot_person)
```

### Explicit Traversal
```python
from neomodel import Traversal, OUTGOING

definition = dict(
    node_class=Person,
    direction=OUTGOING,
    relation_type=None,  # Any relationship type
    model=None
)

relations_traversal = Traversal(jim, Person.__label__, definition)
all_jims_relations = relations_traversal.all()
```

## Property Types

### Basic Properties
```python
from neomodel import (
    StringProperty, IntegerProperty, FloatProperty, BooleanProperty,
    DateProperty, DateTimeProperty, JSONProperty, ArrayProperty
)

class Product(StructuredNode):
    name = StringProperty(required=True)
    price = FloatProperty(default=0.0)
    tags = ArrayProperty(StringProperty())
    metadata = JSONProperty()
    created = DateTimeProperty(default_now=True)
```

### Advanced Properties
```python
# Choices
status = StringProperty(choices=['active', 'inactive', 'pending'])

# Unique identifiers
uid = StringProperty(unique_index=True)

# Aliased properties
display_name = StringProperty(db_property='name')

# Reserved properties (avoid these names)
# element_id, id, element_id_property, id_property
```

## Querying

### Basic Queries
```python
# Get all nodes
people = Person.nodes.all()

# Filter nodes
active_users = User.nodes.filter(status='active')

# Get single node
user = User.nodes.get(username='john')

# First/last
first_user = User.nodes.first()
last_user = User.nodes.last()
```

### Filtering and Ordering
```python
# Complex filters
users = User.nodes.filter(
    age__gte=18,
    status__in=['active', 'premium']
).order_by('-created_date')

# Exclude
inactive_users = User.nodes.exclude(status='inactive')
```

### Relationships and Traversal
```python
# Traverse relationships
friends = person.friends.all()

# Match with relationships
friends_of_friends = person.friends.match(friend__friends__name='Alice')

# Path traversal
paths = person.friends.traverse('Person', 'FRIENDS_WITH', 'Person')
```

## Advanced Query Operations

### Aggregations
```python
from neomodel import Q

# Count
total_users = User.nodes.count()

# Aggregate with filtering
active_count = User.nodes.filter(status='active').count()

# Sum, average, min, max
total_age = Person.nodes.annotate(sum_age=Sum('age')).first().sum_age
avg_age = Person.nodes.annotate(avg_age=Avg('age')).first().avg_age
```

### Subqueries
```python
# Subquery for complex operations
popular_posts = Post.nodes.annotate(
    comment_count=Count('comments')
).filter(comment_count__gt=10)
```

### Cypher Integration
```python
# Raw Cypher queries
query = "MATCH (p:Person)-[:FRIENDS_WITH]-(friend) RETURN p, count(friend) as friend_count"
results = db.cypher_query(query)

# Parameterized queries
query = "MATCH (p:Person {name: $name}) RETURN p"
results = db.cypher_query(query, {'name': 'John'})
```

## Transactions

### Basic Transactions
```python
from neomodel import db

with db.transaction:
    user = User(name="John").save()
    profile = Profile(user=user, bio="Hello").save()
# Auto-commits on success, rolls back on exception
```

### Explicit Transactions
```python
from neo4j import Transaction

def create_user(tx: Transaction, name: str):
    query = "CREATE (u:User {name: $name}) RETURN u"
    return tx.run(query, name=name)

with db.driver.session() as session:
    with session.begin_transaction() as tx:
        result = create_user(tx, "John")
        tx.commit()
```

## Schema Management

### Defining Models
```python
from neomodel import StructuredNode, RelationshipTo, One, ZeroOrMore

class User(StructuredNode):
    username = StringProperty(unique_index=True, required=True)
    email = StringProperty(unique_index=True, required=True)
    posts = RelationshipTo('Post', 'AUTHORED', cardinality=ZeroOrMore)

class Post(StructuredNode):
    title = StringProperty(required=True)
    content = StringProperty()
    author = RelationshipFrom('User', 'AUTHORED', cardinality=One)
```

### Applying Constraints and Indexes
```python
# Install all labels and constraints
from neomodel import install_all_labels

install_all_labels()

# Or use command line
neomodel_install_labels

# Remove constraints
from neomodel import remove_all_labels
remove_all_labels()
```

## Configuration

### Connection Setup
```python
from neomodel import config

# Basic connection
config.DATABASE_URL = 'bolt://neo4j:password@localhost:7687'

# Advanced configuration
config.DRIVER_CONFIG = {
    'encrypted': True,
    'trust': 1,  # TRUST_SYSTEM_CA_SIGNED_CERTIFICATES
    'max_connection_pool_size': 100,
}

# Auto-install labels (deprecated in 5.3)
config.AUTO_INSTALL_LABELS = True

# Require timezones
config.FORCE_TIMEZONE = True
```

### Async Configuration
```python
import asyncio
from neomodel.async_ import config as async_config

async_config.DATABASE_URL = 'bolt://neo4j:password@localhost:7687'
```

## Batch Operations

### Bulk Create
```python
# Create multiple nodes
people_data = [
    {'name': 'Alice', 'age': 25},
    {'name': 'Bob', 'age': 30},
    {'name': 'Charlie', 'age': 35}
]

people = Person.create(*people_data)
```

### Create or Update
```python
# Create or update based on unique properties
person, created = Person.create_or_update(
    {'name': 'Alice'},
    {'name': 'Alice', 'age': 26}  # Updates age if Alice exists
)
```

### Get or Create
```python
# Get existing or create new
person = Person.get_or_create({'name': 'Alice'}, {'name': 'Alice', 'age': 25})
```

## Hooks

### Pre/Post Operation Hooks
```python
class User(StructuredNode):
    name = StringProperty()

    def pre_save(self):
        # Called before saving
        self.name = self.name.strip().title()

    def post_save(self):
        # Called after saving
        print(f"User {self.name} saved")

    def pre_delete(self):
        # Called before deletion
        print(f"Deleting user {self.name}")

    def post_delete(self):
        # Called after deletion
        print(f"User {self.name} deleted")
```

### Relationship Hooks
```python
class Person(StructuredNode):
    name = StringProperty()
    friends = RelationshipTo('Person', 'FRIENDS_WITH')

    def pre_connect(self, relationship):
        # Called before connecting relationships
        print(f"Connecting {self.name} to {relationship.end_node.name}")

    def post_connect(self, relationship):
        # Called after connecting relationships
        print("Connection established")
```

## Extending Neomodel

### Inheritance
```python
class Person(StructuredNode):
    name = StringProperty()

class Employee(Person):
    salary = IntegerProperty()
    department = StringProperty()

# Employee inherits name property and gets its own label
```

### Mixins
```python
class TimestampMixin:
    created_at = DateTimeProperty(default_now=True)
    updated_at = DateTimeProperty(default_now=True)

    def pre_save(self):
        self.updated_at = datetime.now()

class User(TimestampMixin, StructuredNode):
    name = StringProperty()
```

### Custom Labels
```python
class Person(StructuredNode):
    __label__ = 'Individual'  # Custom label instead of 'Person'

class Employee(Person):
    __label__ = 'Worker'  # Custom label for subclass
```

## Async Support

### Async Operations
```python
from neomodel.async_ import StructuredNode as AsyncStructuredNode
from neomodel.async_ import StringProperty, RelationshipTo

class AsyncPerson(AsyncStructuredNode):
    name = StringProperty()
    friends = RelationshipTo('AsyncPerson', 'FRIENDS_WITH')

# Async operations
person = await AsyncPerson(name="John").asave()
people = await AsyncPerson.nodes.filter(name__startswith="J").all()
```

### Async Database Operations
```python
from neomodel.async_ import AsyncDatabase

adb = AsyncDatabase()

# Async database operations
await adb.cypher_query("CREATE (n:Test) RETURN n")
await adb.install_labels()
```

## Semantic Indexes

### Full Text Index
```python
from neomodel import FulltextIndex

# Create fulltext index
Person.create_fulltext_index(['name', 'bio'])

# Search
results = Person.index.search("software engineer")
```

### Vector Index
```python
from neomodel import VectorIndex

# Create vector index
Person.create_vector_index('embedding', dimensions=384)

# Vector search
results = Person.index.search_vector(embedding_vector, limit=10)
```

## Best Practices

### Model Design
1. Use descriptive property names
2. Define cardinality constraints appropriately
3. Use unique indexes for natural keys
4. Consider inheritance for related entities
5. Use mixins for common functionality

### Query Optimization
1. Use indexes on frequently filtered properties
2. Prefer relationship traversal over complex Cypher
3. Use batch operations for bulk updates
4. Consider pagination for large result sets
5. Use transactions for data consistency

### Performance
1. Connection pooling is automatic
2. Reuse database connections
3. Use async operations for I/O bound tasks
4. Monitor query performance
5. Consider caching for frequently accessed data

### Error Handling
1. Wrap operations in transactions
2. Handle constraint violations
3. Use try/except for database operations
4. Log errors appropriately
5. Provide meaningful error messages

## Common Patterns

### User Management
```python
class User(StructuredNode):
    username = StringProperty(unique_index=True, required=True)
    email = StringProperty(unique_index=True, required=True)
    created_at = DateTimeProperty(default_now=True)

    posts = RelationshipTo('Post', 'AUTHORED')
    followers = RelationshipTo('User', 'FOLLOWS')

    @classmethod
    def create_user(cls, username, email):
        return cls(username=username, email=email).save()

    def get_posts(self):
        return self.posts.all()
```

### Social Graph
```python
class Person(StructuredNode):
    name = StringProperty(unique_index=True)
    friends = RelationshipTo('Person', 'FRIENDS_WITH')

    def add_friend(self, person):
        self.friends.connect(person)

    def get_friends_of_friends(self):
        return self.friends.match(friend__friends__name__ne=self.name)
```

### E-commerce
```python
class Product(StructuredNode):
    name = StringProperty(required=True)
    price = FloatProperty(required=True)
    category = StringProperty()

    orders = RelationshipFrom('Order', 'CONTAINS')

class Order(StructuredNode):
    total = FloatProperty()
    created_at = DateTimeProperty(default_now=True)

    customer = RelationshipTo('Customer', 'PLACED_BY', cardinality=One)
    items = RelationshipTo('Product', 'CONTAINS')

    def calculate_total(self):
        total = 0
        for item in self.items.all():
            total += item.price
        self.total = total
        return self.save()
```

This comprehensive reference covers the core functionality and advanced features of Neomodel for effective Neo4j graph database operations.
