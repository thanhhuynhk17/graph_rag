# Advanced Query Operations — Neomodel 5.5.3 Documentation

## Introduction
Neomodel provides ways to enhance your queries beyond filtering and traversals. This section covers advanced query operations, including annotations, aggregations, intermediate transformations, subqueries, and helper methods.

## Annotate - Aliasing
The `annotate` method allows you to add transformations to your elements. This section explores available transformations and how to use them effectively.

## Aggregations
Neomodel implements some of the aggregation methods available in Cypher, including:

- `Collect` (with a `distinct` option)
- `Last`

These can be used as follows:

```python
from neomodel.sync_.match import Collect, Last

# `distinct` is optional and defaults to False. When True, objects are deduplicated
Supplier.nodes.traverse_relations(available_species="coffees__species") \
    .annotate(Collect("available_species", distinct=True)) \
    .all()

# `Last` is used to get the last element of a list
Supplier.nodes.traverse_relations(available_species="coffees__species") \
    .annotate(Last(Collect("last_species"))) \
    .all()
```

**Note**: The `annotate` method is used to add aggregation methods to the query.

**Note**: Using `Last()` immediately after a `Collect()` without setting an ordering will return the last element in the list as returned by the database, which may not be desired. To address this, you must provide explicit ordering using an intermediate transformation step (see below). This is because the `order_by` method adds ordering as the final step of the Cypher query, whereas you may need to order results before applying an aggregation, requiring an intermediate `WITH` clause.

## Intermediate Transformations
The `intermediate_transform` method allows you to add a `WITH` clause to your query, enabling operations on results before returning them. This is particularly useful when you need to order results before applying an aggregation, like so:

```python
from neomodel.sync_.match import Collect, Last

# This will return all Coffee nodes with their most expensive supplier
Coffee.nodes.traverse_relations(suppliers="suppliers") \
    .intermediate_transform(
        {"suppliers": {"source": "suppliers"}}, ordering=["suppliers.delivery_cost"]
    ) \
    .annotate(supps=Last(Collect("suppliers")))
```

### Options for `intermediate_transform` Variables
- `source`: String or Resolver - The variable to use as the source for the transformation. Works with resolvers (see below).
- `source_prop`: String - Optionally, a property of the source variable to use as the source for the transformation.
- `include_in_return`: Boolean - Whether to include the variable in the return statement. Defaults to `False`.

### Additional Options for `intermediate_transform`
- `distinct`: Boolean - Whether to deduplicate the results. Defaults to `False`.

Here is a full example:

```python
await Coffee.nodes.fetch_relations("suppliers") \
    .intermediate_transform(
        {
            "coffee": {"source": "coffee", "include_in_return": True},  # Only coffee will be returned
            "suppliers": {"source": NodeNameResolver("suppliers")},
            "r": {"source": RelationNameResolver("suppliers")},
            "cost": {
                "source": NodeNameResolver("suppliers"),
                "source_prop": "delivery_cost",
            },
        },
        distinct=True,
        ordering=["-r.since"],
    ) \
    .annotate(oldest_supplier=Last(Collect("suppliers"))) \
    .all()
```

## Subqueries
The `subquery` method allows you to perform a Cypher subquery inside your query, enabling operations in isolation from the rest of your query:

```python
from neomodel.sync_.match import Collect, Last

# This will create a CALL{} subquery
# And return a variable named supps usable in the rest of your query
Coffee.nodes.filter(name="Espresso") \
    .subquery(
        Coffee.nodes.traverse_relations(suppliers="suppliers") \
            .intermediate_transform(
                {"suppliers": {"source": "suppliers"}}, ordering=["suppliers.delivery_cost"]
            ) \
            .annotate(supps=Last(Collect("suppliers"))),
        ["supps"],
        [NodeNameResolver("self")]
    )
```

### Options for `subquery` Calls
- `return_set`: List of strings - The subquery variables that should be included in the outer query result.
- `initial_context`: Optional list of strings or Resolver - The outer query variables that will be injected at the beginning of the subquery.

**Note**: In the example above, `self` is referenced in the `initial_context` to inject the outer variable corresponding to the `Coffee` node. This can be confusing, and better approaches are welcome if suggested.

## Helpers
Explicit aliasing is often used in queries, as shown in:

```python
traverse_relations(suppliers="suppliers").annotate(Collect("suppliers"))
```

This allows referencing generated Cypher variables in transformation steps. However, when aliasing is not possible (e.g., with `fetch_relations`), Neomodel provides resolver methods (`NodeNameResolver` and `RelationshipNameResolver`) to avoid guessing variable names in the generated Cypher. For example:

```python
from neomodel.sync_.match import Collect, NodeNameResolver, RelationshipNameResolver

Supplier.nodes.fetch_relations("coffees__species") \
    .annotate(
        all_species=Collect(NodeNameResolver("coffees__species"), distinct=True),
        all_species_rels=Collect(RelationNameResolver("coffees__species"), distinct=True)
    ) \
    .all()
```

**Note**: When using resolvers with a traversal, they resolve the variable name of the last element in the traversal (e.g., the `Species` node for `NodeNameResolver` and the `Coffee–Species` relationship for `RelationshipNameResolver`).

Another example is referencing the root node itself:

```python
subquery = await Coffee.nodes.subquery(
    Coffee.nodes.traverse_relations(suppliers="suppliers") \
        .intermediate_transform(
            {"suppliers": {"source": "suppliers"}}, ordering=["suppliers.delivery_cost"]
        ) \
        .annotate(supps=Last(Collect("suppliers"))),
    ["supps"],
    [NodeNameResolver("self")]  # This is the root Coffee node
)
```