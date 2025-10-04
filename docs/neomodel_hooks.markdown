# Hooks — Neomodel 5.5.3 Documentation

## Introduction
Neomodel provides hooks to allow custom logic to be executed at specific points in the lifecycle of a node or relationship. These hooks are useful for tasks such as validation, default value setting, or triggering side effects before or after database operations.

## Available Hooks
Hooks are defined as methods on a `StructuredNode` or `StructuredRel` class. The following hooks are supported:

- `pre_save`: Called before a node or relationship is saved to the database.
- `post_save`: Called after a node or relationship is saved to the database.
- `pre_delete`: Called before a node or relationship is deleted from the database.
- `post_delete`: Called after a node or relationship is deleted from the database.
- `post_create`: Called after a node or relationship is created in the database.

## Defining Hooks
Hooks are implemented by defining methods with specific names in your model classes. Below is an example of a `Person` node with hooks:

```python
from neomodel import StructuredNode, StringProperty, db

class Person(StructuredNode):
    name = StringProperty(unique_index=True)
    email = StringProperty()

    def pre_save(self):
        # Ensure email is lowercase before saving
        if self.email:
            self.email = self.email.lower()

    def post_save(self):
        # Log or perform an action after saving
        print(f"Person {self.name} saved with email {self.email}")

    def pre_delete(self):
        # Perform checks or cleanup before deletion
        print(f"Preparing to delete Person {self.name}")

    def post_delete(self):
        # Perform actions after deletion
        print(f"Person {self.name} deleted")

    def post_create(self):
        # Perform actions after creation
        print(f"Person {self.name} created")
```

### Explanation
- `pre_save`: Useful for validating or modifying properties before saving to the database. In the example, the email is converted to lowercase.
- `post_save`: Useful for logging or triggering side effects after a save operation.
- `pre_delete`: Allows cleanup or validation before a node is deleted.
- `post_delete`: Useful for logging or cleanup after deletion.
- `post_create`: Called only when a node is created (not updated), useful for initialization tasks.

## Using Hooks
To use hooks, simply define them in your model and perform operations as usual. The hooks will be triggered automatically:

```python
# Create a person
person = Person(name="Alice", email="ALICE@example.com").save()
# Output: Person Alice created
# Output: Person Alice saved with email alice@example.com

# Update the person
person.email = "alice.new@example.com"
person.save()
# Output: Person Alice saved with email alice.new@example.com

# Delete the person
person.delete()
# Output: Preparing to delete Person Alice
# Output: Person Alice deleted
```

## Hooks on Relationships
Hooks can also be defined on `StructuredRel` classes for relationships. For example:

```python
from neomodel import StructuredRel, IntegerProperty, RelationshipTo

class Friendship(StructuredRel):
    since = IntegerProperty()

    def pre_save(self):
        # Ensure 'since' is a positive integer
        if self.since and self.since < 0:
            raise ValueError("'since' must be a positive integer")

    def post_save(self):
        print(f"Friendship saved with since {self.since}")

class Person(StructuredNode):
    name = StringProperty(unique_index=True)
    friends = RelationshipTo('Person', 'FRIENDS', model=Friendship)
```

### Using Relationship Hooks
```python
alice = Person(name="Alice").save()
bob = Person(name="Bob").save()

# Create a relationship with a hook
alice.friends.connect(bob, {"since": 2023})
# Output: Friendship saved with since 2023

# Attempt to create an invalid relationship
try:
    alice.friends.connect(bob, {"since": -1})
except ValueError as e:
    print(e)  # Output: 'since' must be a positive integer
```

## Transactions and Hooks
Hooks are executed within the context of the database transaction. If a transaction fails, any changes made in `pre_save` or `post_create` will not be persisted, and `post_save` or `post_delete` hooks will not be called.

Example with a transaction:

```python
with db.transaction:
    person = Person(name="Charlie", email="CHARLIE@example.com").save()
    # If the transaction fails, post_create and post_save hooks won't persist
```

## Notes
- **Hook Execution Order**: Hooks are executed in the order of the operation lifecycle (`post_create` → `pre_save` → `post_save` for creation; `pre_save` → `post_save` for updates; `pre_delete` → `post_delete` for deletion).
- **Exceptions in Hooks**: If a hook raises an exception, the operation (save, delete, etc.) will fail, and the transaction will be rolled back.
- **Performance**: Be cautious with complex logic in hooks, as they are executed as part of the database operation and can impact performance.
- **Inheritance**: Hooks defined in a parent class are inherited by subclasses unless overridden.

For more details, refer to the [Neomodel documentation](https://neomodel.readthedocs.io/en/latest/).