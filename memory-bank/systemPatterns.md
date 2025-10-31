# System Patterns and Architecture

## System Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   MCP Server    │    │  Hybrid Search   │    │   Order Mgmt    │
│  (FastAPI)      │◄──►│  Pipeline        │◄──►│   Graph DB      │
│                 │    │  - BM25          │    │  (Neo4j)        │
│ - REST API      │    │  - Embeddings    │    │                 │
│ - MCP Tools     │    │  - Reranking     │    │ - Graph Models  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                         │                   │
         └─────────────────────────┼───────────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │        Neo4j Graph DB       │
                    │ - Chunk nodes (embeddings) │
                    │ - Order relationships      │
                    │ - Customer/Order/Dish      │
                    └─────────────────────────────┘
```

## Key Technical Decisions

### 1. MCP + REST Dual Interface Architecture
- **Decision**: Implement both MCP tools and REST API endpoints
- **Rationale**: Support both modern MCP clients and traditional API integrations
- **Pattern**: Shared `OrderService` logic ensures consistency between interfaces

### 2. Hybrid Retrieval Pipeline
**Components**:
- BM25Retriever: Keyword-based lexical search
- Neo4jVector: Dense embedding similarity search
- EnsembleRetriever: Weighted combination of retrievers
- BGE Reranker: ML-based result reranking

**Data Flow**:
```
Query → Vietnamese NLP Preprocessing → Ensemble Retrieval → ML Reranking → Results
```

### 3. Graph-Based Order Management
**Node Types**:
- `Customer` - customer_id, full_name, phone[], email
- `Order` - order_id, total_bill, table_id, is_takeaway, notes
- `Dish` - dish_id, name_of_food, current_price, type_of_food, combine_info

**Relationship Types**:
- `PLACED` (Customer→Order): arrived_at, created_at timestamps
- `CONTAINS` (Order→Dish): quantity, price (at time of order)

### 4. Service Layer Architecture (NEW)
**Shared OrderService Pattern**:
- Unifies business logic between MCP tools and REST APIs
- Ensures consistency across different client interfaces
- Centralizes order validation, creation, and error handling
- Supports both synchronous MCP and asynchronous REST operations

## Component Relationships

### MCP Server (src/mcp_server.py)
**Responsibilities**:
- FastMCP server setup and lifecycle management
- Database connection and cleanup
- Tool definitions (multi_dish_lookup, menu_value_count_and_price, take_order)
- REST API endpoints mirroring MCP tools

**Dependencies**:
- OrderService: Shared business logic for order operations
- Neo4j driver: Direct graph database access
- OrderManager: Graph-based order orchestration

### Hybrid Search (src/utils/hybridsearch.py)
**Core Pattern**: Lazy-loaded global pipeline singleton
```python
_pipeline = None
def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = HybridRetrieverPipeline(...)
    return _pipeline
```

**Pipeline Structure**:
- Preloaded documents from Neo4j Chunk nodes
- BM25Retriever for keyword search
- Neo4jVector retriever for embedding similarity
- ML reranker for result refinement

### Order Management (src/models/order_manager.py)
**Key Patterns**:
- Asynchronous validation pipeline
- Table assignment algorithm (90-minute conflict window)
- Graph transaction management
- Relationship creation with metadata

**Critical Implementation Path**:
```
validate_and_prepare_dishes() → calculate_total() → assign_table() → create_customer_graph() → create_order_graph() → create_relationships()
```

### Neo4j Graph Models (src/models/order_graph.py)
**Neomodel ORM Usage**:
- Declarative node definitions with property validation
- Relationship definitions with cardinality constraints
- Automatic property indexing
- JSON serialization support

## Data Flow Patterns

### 1. Search Request Flow
```
User Query → MCP Tool → Vietnamese NLP → Ensemble Retriever → BGE Reranker → Structured Results
```

### 2. Order Creation Flow
```
API/MCP Request → OrderService.validate_customer() → OrderManager.create_order() → Graph Transaction → Response
```

### 3. Data Ingestion Flow
```
CSV Files → chunk_docs_neo4j.py → Neo4j Chunk Nodes → Embedding Generation → Vector Index Creation
```

## Vietnamese NLP Patterns

### Text Preprocessing Pipeline
1. **Normalization**: `helpers.normalize_vnese()` - Unicode normalization, diacritic standardization
2. **Tokenization**: `underthesea.word_tokenize()` - Vietnamese-aware word segmentation
3. **Stopword Removal**: Domain-specific filtering (numbers, common Vietnamese stopwords)
4. **Sequence Processing**: Split by commas, normalize each segment

### Query Enhancement
- Case-insensitive matching
- Vietnamese abbreviation handling ("ĐT" → "điện thoại")
- Contextual term expansion
- Multi-field search (name, type, ingredients, description)

## Error Handling Patterns

### Database Transaction Pattern
```python
try:
    # Graph operations
    customer.save()
    order.save()
    customer.placed.connect(order, {...})
except Neo4jError as e:
    # Rollback logic
    raise Exception(f"Database error: {str(e)}")
```

### Validation Cascade Pattern
```python
def validate_and_prepare_dishes(dishes):
    for dish in dishes:
        validate_format(dish)
        validate_quantity(dish)
        check_existence(dish['dish_id'])
        fetch_price(dish)
```

## Configuration Management

### Environment-Based Config
- **Centralized Loading**: `dotenv` in each module
- **Validation**: Required vs optional variables
- **Logging**: Configurable log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)

### Runtime Behavior Control
- `AUTO_LOAD_DISHES`: Automatic CSV ingestion on startup
- `FORCE_REFRESH_DISHES`: Data safety override
- `LOG_LEVEL`: Debugging verbosity control

## Deployment Patterns

### Docker Compose Architecture
```yaml
services:
  neo4j:        # Graph database
  embedding:    # Qwen3 embedding service
  reranker:     # BGE reranking service
  app:          # MCP server application
```

### Service Communication
- **Internal**: Docker network with service names
- **External**: Host networking for development
- **Database**: Bolt protocol connections

## Testing Patterns

### Component Isolation
- `conftest.py`: Shared test fixtures (Neo4j connection, sample data)
- Unit tests for individual components (order manager, hybrid search)
- Integration tests for API endpoints and MCP tools

### Test Data Management
- CSV-based fixture loading
- Graph cleanup between tests
- Deterministic order creation for reliable assertions

## Performance Optimization Patterns

### Query Optimization
- **Bulk Operations**: Single Cypher queries over multiple round-trips
- **Index Usage**: Automatic property indexes via neomodel
- **Connection Pooling**: Neo4j driver connection reuse

### Memory Management
- **Lazy Loading**: Global pipeline singleton pattern
- **Batch Processing**: Ensemble retriever configuration
- **Document Preloading**: Cached document loading in hybrid search

## Extensibility Patterns

### Plugin Architecture
- **MCP Tools**: Easy addition of new search/filter tools
- **Model Extensions**: Neomodel inheritance for new graph entities
- **NLP Modules**: Swappable preprocessing components

### Configuration-Driven Behavior
- **Model Selection**: Environment-based embedding/reranker switching
- **Parameter Tuning**: Configurable k-values, thresholds
- **Feature Flags**: Runtime behavior modification
