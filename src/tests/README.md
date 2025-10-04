# Test Suite Documentation

This directory contains comprehensive tests for the restaurant order management system using Neomodel ORM.

## Test Structure

```
src/tests/
├── __init__.py              # Package initialization
├── conftest.py              # Pytest fixtures and configuration
├── pytest.ini              # Pytest configuration
├── requirements-test.txt    # Test-specific dependencies
├── README.md               # This file
├── test_models.py          # Unit tests for Neomodel models
└── test_order_manager.py   # Integration tests for business logic
```

## Test Categories

### Unit Tests (`test_models.py`)
- **Model Creation & Validation**: Test creating nodes with various data types
- **Relationship Management**: Test creating, updating, and deleting relationships
- **Query Operations**: Test filtering, searching, and ordering
- **Edge Cases**: Test error handling and boundary conditions
- **Data Integrity**: Test constraints and data consistency

### Integration Tests (`test_order_manager.py`)
- **Business Logic**: Test OrderManager methods with mocked database
- **Workflow Testing**: Test complete order creation workflows
- **Error Handling**: Test error scenarios and recovery
- **Performance**: Test with large datasets and concurrent operations

## Running Tests

### Prerequisites

1. **Install test dependencies**:
   ```bash
   cd src/tests
   pip install -r requirements-test.txt
   # or
   uv add -r requirements-test.txt
   ```

2. **Set up test database**:
   - Ensure Neo4j is running
   - Set environment variable: `TEST_NEO4J_BOLT_URL=bolt://neo4j:test@localhost:7687/test`

### Run All Tests

```bash
# From project root
cd src/tests
python -m pytest

# Or using uv
uv run python -m pytest src/tests/
```

### Run Specific Test Categories

```bash
# Unit tests only
python -m pytest src/tests/test_models.py -m unit

# Integration tests only
python -m pytest src/tests/test_order_manager.py -m integration

# Database tests only
python -m pytest -m database

# Slow tests only
python -m pytest -m slow
```

### Run with Coverage

```bash
# Generate coverage report
python -m pytest --cov=src --cov-report=html

# View coverage in browser
open htmlcov/index.html
```

### Run Specific Tests

```bash
# Run specific test class
python -m pytest src/tests/test_models.py::TestCustomerModel

# Run specific test method
python -m pytest src/tests/test_models.py::TestCustomerModel::test_customer_creation

# Run tests matching pattern
python -m pytest -k "test_customer"
```

## Test Fixtures

### Database Fixtures
- `setup_test_database`: Sets up clean test database for session
- `sample_customer_data`: Provides test customer data
- `sample_dish_data`: Provides test dish data
- `sample_order_data`: Provides test order data
- `sample_datetime`: Provides test datetime in Asia/Ho_Chi_Minh timezone

### Mock Fixtures
- `mock_driver`: Mocked Neo4j driver for unit testing
- `order_manager`: OrderManager instance with mocked dependencies
- `mock_embedding`: Mock embedding vector for testing

## Test Data

### Sample Data Structure

**Customer**:
```python
{
    "_id": "test_customer_001",
    "full_name": "Nguyễn Văn Test",
    "phone": ["0901234567", "0987654321"],
    "email": "test@example.com"
}
```

**Dish**:
```python
{
    "_id": "dish_001",
    "type_of_food": "MÓN KHAI VỊ",
    "name_of_food": "Bánh xèo",
    "current_price": 145000.0,
    "combine_info": "Combined information for search..."
}
```

**Order**:
```python
{
    "_id": "order_001",
    "total_bill": 295000.0,
    "is_takeaway": False,
    "table_id": 1,
    "notes": "Không hành, không ớt"
}
```

## Writing New Tests

### Adding Test Fixtures

```python
@pytest.fixture
def my_custom_fixture():
    # Setup code
    data = {"key": "value"}
    yield data
    # Cleanup code
```

### Testing Async Code

```python
import pytest

@pytest.mark.asyncio
async def test_async_function():
    result = await async_function()
    assert result == expected
```

### Testing with Mocked Dependencies

```python
def test_with_mocks(mock_driver, order_manager):
    # Mock the driver behavior
    order_manager.driver.execute_query = AsyncMock(return_value=mock_response)

    # Test your function
    result = order_manager.some_method()

    # Assert expectations
    assert result == expected
```

## Best Practices

### Test Organization
- **One assertion per test**: Keep tests focused and easy to debug
- **Descriptive test names**: Use `test_<functionality>_<scenario>`
- **Arrange-Act-Assert**: Structure tests clearly
- **Independent tests**: Tests should not depend on each other

### Database Testing
- **Clean state**: Each test starts with clean database state
- **Proper cleanup**: Use fixtures for automatic cleanup
- **Realistic data**: Use data that matches production scenarios

### Mocking
- **Mock external dependencies**: Database, APIs, file systems
- **Mock only what you need**: Keep mocks simple and focused
- **Verify mock interactions**: Ensure mocks are called as expected

## Debugging Tests

### Common Issues

1. **Database connection errors**:
   - Check Neo4j is running
   - Verify `TEST_NEO4J_BOLT_URL` environment variable
   - Ensure test database exists

2. **Import errors**:
   - Install test dependencies: `pip install -r requirements-test.txt`
   - Check Python path includes project root

3. **Async test issues**:
   - Use `@pytest.mark.asyncio` for async tests
   - Ensure proper async/await usage

### Debug Mode

```bash
# Run with detailed output
python -m pytest -v -s

# Run specific test with debugger
python -m pytest --pdb test_specific_file.py::TestClass::test_method

# Stop at first failure
python -m pytest -x
```

## Continuous Integration

### GitHub Actions Example

```yaml
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'
    - name: Install dependencies
      run: |
        pip install -r src/tests/requirements-test.txt
    - name: Run tests
      run: |
        cd src/tests && python -m pytest --cov=src
```

## Coverage Goals

Aim for:
- **Overall coverage**: > 90%
- **Model coverage**: > 95%
- **Business logic coverage**: > 90%
- **Error handling coverage**: > 85%

## Contributing

When adding new features:
1. Write tests before implementing features (TDD)
2. Ensure all tests pass before submitting PR
3. Update test documentation if needed
4. Add appropriate test markers for new test categories

## Troubleshooting

### Neo4j Connection Issues
- Ensure Neo4j Desktop/Community is running
- Check authentication credentials
- Verify database URL format: `bolt://user:password@host:port/database`

### Test Discovery Issues
- Ensure test files start with `test_`
- Check `python_files` pattern in `pytest.ini`
- Verify package structure and imports

### Performance Issues
- Use `pytest -m "not slow"` to skip slow tests
- Consider using database transactions for faster cleanup
- Mock expensive operations when possible
