# Path Traversal — Neomodel 5.5.3 Documentation

## Introduction
Neo4j is about traversing the graph, which means leveraging nodes and relationships between them. This section will show you how to traverse the graph using Neomodel.

For this, the primary method to use is `traverse`.

**Note**: Until version 6, two other methods are available but deprecated: `traverse_relations` and `fetch_relations`. These two methods are mutually exclusive, so you cannot chain them.

For the examples in this section, we will use the following model:

```python
class Country(StructuredNode):
    country_code = StringProperty(unique_index=True)
    name = StringProperty()

class Supplier(StructuredNode):
    name = StringProperty()
    delivery_cost = IntegerProperty()
    country = RelationshipTo(Country, 'ESTABLISHED_IN')

class Coffee(StructuredNode):
    name = StringProperty(unique_index=True)
    price = IntegerProperty()
    suppliers = RelationshipFrom(Supplier, 'SUPPLIES')
```

## Traverse Relations
The `traverse` method allows you to define multiple, multi-hop traversals, optionally returning traversed elements.

For example, to find all `Coffee` nodes that have a supplier and retrieve the country of that supplier, you can do:

```python
Coffee.nodes.traverse("suppliers__country").all()
```

This generates a Cypher `MATCH` clause that traverses `Coffee<–Supplier–>Country` and, by default, returns all traversed nodes and relationships.

The `traverse` method allows you to define a more complex `Path` object, giving you greater control over the traversal.

You can specify which elements to return, like:

```python
# Return only the traversed nodes, not the relationships
Coffee.nodes.traverse(Path(value="suppliers__country", include_rels_in_return=False))

# Return only the traversed relationships, not the nodes
Coffee.nodes.traverse(Path(value="suppliers__country", include_nodes_in_return=False))
```

You can specify that your traversal should be optional, like:

```python
# Return only the traversed nodes, not the relationships
Coffee.nodes.traverse(Path(value="suppliers__country", optional=True))
```

You can also alias the path to reference it later in the query, like:

```python
Coffee.nodes.traverse(Path(value="suppliers__country", alias="supplier_country"))
```

The `Country` nodes matched will be available for the rest of the query with the variable name `country`. Note that aliasing is optional. See *Advanced Query Operations* for examples of how to use this aliasing.

**Note**: The `traverse` method can be used to traverse multiple paths, like:

```python
Coffee.nodes.traverse('suppliers__country', 'pub__city').all()
```

This generates a Cypher `MATCH` clause that traverses both paths `Coffee<–Supplier–>Country` and `Coffee<–Pub–>City`.

**Note**: When using `include_rels_in_return=True` (default), any relationship traversed using this method **must** have a model defined, even if only the default `StructuredRel`, like:

```python
class Person(StructuredNode):
    country = RelationshipTo(Country, 'IS_FROM', model=StructuredRel)
```

Otherwise, Neomodel will not be able to determine which relationship model to resolve into and will fail.

## Traverse Relations (Deprecated)
**Deprecated since version 5.5.0**: This method is set to disappear in version 6; use `traverse` instead.

The `traverse_relations` method allows you to filter on the existence of more complex traversals. For example, to find all `Coffee` nodes that have a supplier and retrieve the country of that supplier, you can do:

```python
Coffee.nodes.traverse_relations(country='suppliers__country').all()
```

This generates a Cypher `MATCH` clause that enforces the existence of at least one path like `Coffee<–Supplier–>Country`.

The `Country` nodes matched will be available for the rest of the query with the variable name `country`. Note that aliasing is optional. See *Advanced Query Operations* for examples of how to use this aliasing.

**Note**: The `traverse_relations` method can be used to traverse multiple relationships, like:

```python
Coffee.nodes.traverse_relations('suppliers__country', 'pub__city').all()
```

This generates a Cypher `MATCH` clause that enforces the existence of at least one path like `Coffee<–Supplier–>Country` and `Coffee<–Pub–>City`.

## Fetch Relations (Deprecated)
**Deprecated since version 5.5.0**: This method is set to disappear in version 6; use `traverse` instead.

The syntax for `fetch_relations` is similar to `traverse_relations`, except that the generated Cypher will return all traversed objects (nodes and relationships):

```python
Coffee.nodes.fetch_relations(color='suppliers__country').all()
```

**Note**: Any relationship you intend to traverse using this method **must** have a model defined, even if only the default `StructuredRel`, like:

```python
class Person(StructuredNode):
    country = RelationshipTo(Country, 'IS_FROM', model=StructuredRel)
```

Otherwise, Neomodel will not be able to determine which relationship model to resolve into and will fail.

## Optional Match (Deprecated)
**Deprecated since version 5.5.0**: This method is set to disappear in version 6; use `traverse` instead.

With both `traverse_relations` and `fetch_relations`, you can force the use of an `OPTIONAL MATCH` statement using the following syntax:

```python
from neomodel.match import Optional

# Return the Person nodes, and if they have suppliers, return the suppliers as well
results = Coffee.nodes.fetch_relations(Optional('suppliers')).all()
```

**Note**: You can fetch one or more relations within the same call to `.fetch_relations()` and mix optional and non-optional relations, like:

```python
Person.nodes.fetch_relations('city__country', Optional('country')).all()
```

## Unique Variables
If you want to use the same variable name for traversed nodes when chaining traversals, you can use the `unique_variables` method:

```python
# This does not guarantee that coffees__species will traverse the same nodes as coffees
# So coffees__species can traverse the Coffee node "Gold 3000"
nodeset = (
    Supplier.nodes.fetch_relations("coffees", "coffees__species")
    .filter(coffees__name="Nescafe")
)

# This guarantees that coffees__species will traverse the same nodes as coffees
# So when fetching species, it will only fetch those of the Coffee node "Nescafe"
nodeset = (
    Supplier.nodes.fetch_relations("coffees", "coffees__species")
    .filter(coffees__name="Nescafe")
    .unique_variables("coffees")
)
```

## Resolve Results
By default, `fetch_relations` returns a list of tuples. If your path looks like `(startNode:Coffee)<-[r1]-(middleNode:Supplier)-[r2]->(endNode:Country)`, you will get a list of results where each result is a list of `(startNode, r1, middleNode, r2, endNode)`. These will be resolved by Neomodel, so `startNode` will be a `Coffee` class as defined in Neomodel, for example.

Using the `resolve_subgraph` method, you can get a list of “subgraphs” where each returned `StructuredNode` element will contain its relations and neighbor nodes. For example:

```python
results = Coffee.nodes.fetch_relations('suppliers__country').resolve_subgraph().all()
```

In this example, `results[0]` will be a `Coffee` object with a `_relations` attribute. This will have a `suppliers` and a `suppliers_relationship` attribute, which will contain the `Supplier` object and the relation object, respectively. Recursively, the `Supplier` object will have a `country` attribute, which will contain the `Country` object.

**Note**: The `resolve_subgraph` method is only available for `fetch_relations` queries. This is because `traverse_relations` queries do not return any relations, so there is no need to resolve them.