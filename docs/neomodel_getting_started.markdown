# Getting Started with Neomodel

## Introduction
Neomodel is an Object-Graph Mapper (OGM) for the Neo4j graph database, providing a convenient way to map Neo4j nodes and relationships to Python objects. This guide covers the basics of getting started with Neomodel, including installation, configuration, and defining models.

## Installation
To install Neomodel, you need Python 3.7+ and a running Neo4j database (version 4.0+). Use pip to install Neomodel:

```bash
pip install neomodel
```

### Neo4j Installation
Neomodel requires a running Neo4j instance. You can install Neo4j Community Edition or use Neo4j Desktop. For a quick setup, you can use Docker:

```bash
docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/test neo4j
```

This starts a Neo4j instance with the default username `neo4j` and password `test`. The database will be accessible at `http://localhost:7474`.

## Configuration
Before using Neomodel, configure the connection to your Neo4j database. Set the connection URL using the `neomodel.config` module:

```python
from neomodel import config

config.DATABASE_URL = 'bolt://neo4j:test@localhost:7687'
```

The `DATABASE_URL` format is `bolt://username:password@host:port`. Replace `username`, `password`, `host`, and `port` with your Neo4j instance's credentials and address.

### Environment Variables
Alternatively, you can set the connection details using environment variables:

```bash
export NEO4J_BOLT_URL=bolt://neo4j:test@localhost:7687
```

Neomodel will automatically pick up the `NEO4J_BOLT_URL` environment variable if set.

## Defining Models
Neomodel allows you to define nodes and relationships as Python classes. Below is an example of defining a simple `Person` node and a `FRIENDS` relationship:

```python
from neomodel import StructuredNode, StringProperty, RelationshipTo

class Person(StructuredNode):
    name = StringProperty(unique_index=True)
    friends = RelationshipTo('Person', 'FRIENDS')
```

### Explanation
- `StructuredNode`: Base class for defining nodes.
- `StringProperty`: Defines a property for the node (e.g., `name`).
- `RelationshipTo`: Defines a relationship to another node of the same or different type (e.g., `FRIENDS` relationship to another `Person`).

## Creating and Saving Nodes
Once a model is defined, you can create and save nodes to the database:

```python
from neomodel import db

# Create two Person nodes
alice = Person(name='Alice').save()
bob = Person(name='Bob').save()

# Create a FRIENDS relationship
alice.friends.connect(bob)
```

### Explanation
- `save()`: Persists the node to the Neo4j database.
- `connect()`: Creates a relationship between two nodes.

## Querying Nodes
You can query nodes using Neomodel’s query API:

```python
# Find a person by name
alice = Person.nodes.get(name='Alice')

# Get all friends of Alice
friends = alice.friends.all()
for friend in friends:
    print(friend.name)  # Outputs: Bob
```

### Explanation
- `nodes.get()`: Retrieves a single node matching the query.
- `friends.all()`: Retrieves all nodes connected via the `FRIENDS` relationship.

## Relationships with Properties
You can define properties on relationships by creating a `StructuredRel` class:

```python
from neomodel import StructuredRel, IntegerProperty

class Friendship(StructuredRel):
    since = IntegerProperty()

class Person(StructuredNode):
    name = StringProperty(unique_index=True)
    friends = RelationshipTo('Person', 'FRIENDS', model=Friendship)
```

Then, create a relationship with properties:

```python
alice = Person(name='Alice').save()
bob = Person(name='Bob').save()
alice.friends.connect(bob, {'since': 2023})
```

## Deleting Nodes
To delete a node and its relationships:

```python
alice = Person.nodes.get(name='Alice')
alice.delete()
```

## Transactions
For atomic operations, use transactions:

```python
from neomodel import db

with db.transaction:
    alice = Person(name='Alice').save()
    bob = Person(name='Bob').save()
    alice.friends.connect(bob)
```

## Indexes and Constraints
Neomodel automatically creates indexes and constraints defined in the model (e.g., `unique_index=True` on `name`). To apply these to the database:

```python
from neomodel import install_labels

install_labels(Person)
```

This creates the necessary indexes and constraints in Neo4j.

## Next Steps
- Explore advanced querying with `nodes.filter()` and `nodes.search()`.
- Learn about relationship cardinality (e.g., `ZeroOrOne`, `One`).
- Check out the full documentation for more features like async support and custom queries.

For more details, refer to the [Neomodel documentation](https://neomodel.readthedocs.io/en/latest/).