# Active Context

## Current Work Focus

### ✅ RECENTLY COMPLETED (This Sprint)
- **Order Management Field Consistency**: Fixed dish_id vs id naming mismatch across all components
- **Order History API**: Implemented get_user_orders endpoint with full dish details
- **Enhanced Schemas**: Added UserOrdersResponse for structured order history responses
- **Shared Business Logic**: Created OrderService class to unify MCP and REST API implementations
- **Integration Testing**: Core order management tests now passing at integration level

### Immediate Priority: Production Readiness & Performance
**Status**: Planning Phase
**Focus Areas**:
- Graph query optimization for order history retrieval
- Caching layer implementation for frequent operations
- Production monitoring and logging enhancements
- Security hardening for API endpoints

**Current Approach**:
- Assess current performance bottlenecks
- Design caching strategy for Neo4j queries
- Implement comprehensive health checks
- Add request/response logging and metrics

## Recent Changes & Context

### Latest Code Patterns Observed
- **Hybrid Approach**: MCP tools and REST APIs share OrderService for consistency
- **Graph-First Design**: Orders stored as graph relationships (Customer→Order→Dish)
- **Vietnamese NLP Pipeline**: Multi-stage preprocessing with UnderTheSea tokenization
- **Ensemble Retrieval**: BM25 + embeddings with ML reranking for Vietnamese search

### Architectural Insights
- **FastMCP + FastAPI**: Dual interface support for modern and traditional clients
- **Neomodel ORM**: Declarative graph models with automatic indexing
- **Async Validation**: Non-blocking order validation with graph constraint checking
- **Environment-Driven Config**: Runtime behavior controlled via .env flags

## Next Steps (Updated Priority Order)

### High Priority (Next Sprint - Performance & Production)
1. **Caching Layer Implementation**
   - Redis integration for frequent queries and order lookups
   - Cache Neo4j query results to reduce database load
   - Implement cache invalidation strategies

2. **Monitoring & Observability**
   - Application performance metrics
   - Graph query performance profiling
   - Error rate and latency tracking
   - Health check endpoints enhancement

3. **Production Security**
   - API authentication and authorization
   - Input sanitization and validation hardening
   - Rate limiting and abuse protection

### Medium Priority (Next Sprint)
4. **Advanced Error Handling**
   - Graph transaction rollback improvements
   - Vietnamese localized error messages
   - Recovery patterns for network failures

5. **Scalability Enhancements**
   - Connection pooling configuration
   - Batch processing for bulk operations
   - Consider read replicas for search operations

### Future Considerations (Q1-Q2 Planning)
6. **Deployment Automation**
   - Production Docker orchestration (Kubernetes)
   - Automated backup/restore procedures
   - Configuration management for multi-environment deployments

## Important Patterns & Preferences

### Code Style Preferences
- **Async First**: All new functions should be async unless synchronous required
- **Type Hints**: Pydantic models preferred over plain dictionaries
- **Graph Thinking**: Model relationships as nodes/edges, not foreign keys
- **Environment Config**: All magic numbers should be configurable

### Error Handling Patterns
- **Structured Exceptions**: Custom exception types with context
- **Logging Levels**: INFO for business logic, WARN for recoverable issues, ERROR for failures
- **Rollback Logic**: Graph operations wrapped in transactions with cleanup
- **Graceful Degradation**: Continue operation when auxiliary features fail

### Testing Patterns
- **Graph Isolation**: Clean database state between tests
- **Fixture Reuse**: Shared test data for consistent validation
- **Integration Focus**: API endpoint testing over pure unit tests
- **Real Data**: Use actual CSV data for realistic test scenarios

## Key Insights & Learning

### Vietnamese NLP Challenges
- **Context Matters**: "điện thoại Samsung" ≠ "Samsung điện thoại" in Vietnamese search
- **Abbreviation Handling**: Users write "ĐT" for "điện thoại" - needs normalization
- **Cultural Patterns**: Vietnamese queries often mix product categories and specs
- **Tokenization Limits**: UnderTheSea may miss domain-specific culinary terms

### Graph Database Insights
- **Relationship Metadata**: PLACED and CONTAINS relationships carry timestamps and quantities
- **Constraint Management**: Automatic uniqueness constraints via neomodel decorators
- **Query Complexity**: Cypher queries grow complex with multi-hop traversals
- **Memory Patterns**: Large graphs require careful lazy loading strategies

### MCP Protocol Benefits
- **Tool Discovery**: Automatic tool registration without API documentation
- **Type Safety**: Pydantic schemas ensure parameter validation
- **Client Flexibility**: Same tools work across different MCP-compatible clients
- **Deployment Simplicity**: Single server serves both modern and legacy clients

## Active Decisions & Considerations

### Current Technical Debt
- **Model Loading**: Cold start time (30-60s) impacts user experience
- **Memory Usage**: Full document preloading may not scale to 10k+ products
- **Error Messages**: Vietnamese localization needed for production use
- **Testing Coverage**: Order management tests incomplete pending field fix

### Performance Trade-offs
- **Accuracy vs Speed**: Ensemble retrieval improves accuracy but adds ~100ms latency
- **Memory vs Computation**: Preloaded documents faster but use ~2-4GB RAM
- **Flexibility vs Consistency**: Graph schema changes require migration planning
- **Development vs Production**: Environment flags add flexibility but increase complexity

### Scalability Planning
- **Horizontal Scaling**: API servers can scale independently of Neo4j
- **Read Optimization**: Separate read replicas for search operations
- **Cache Strategy**: Redis layer for frequent embedding lookups
- **Data Partitioning**: Product categories as graph partitions

## Risk Mitigation

### Technical Risks
- **Neo4j Lock-in**: Migration path needed if graph database changes required
- **GPU Dependency**: CPU-only fallback for embedding generation
- **Model Updates**: Backward compatibility when upgrading Qwen3 versions
- **API Breaking Changes**: MCP protocol evolution could affect clients

### Operational Risks
- **Data Corruption**: Graph integrity checks needed before production
- **Backup Procedures**: APOC-based export/import strategy required
- **Monitoring Gaps**: Graph performance metrics not yet instrumented
- **Security Surface**: Additional attack vectors from vector search and MCP protocol

### Business Risks
- **Accuracy Thresholds**: Vietnamese search accuracy may not meet production requirements
- **Cultural Adaptation**: Vietnamese e-commerce patterns differ from Western models
- **Competitive Response**: Building specialized system takes time vs off-the-shelf solutions
- **Talent Requirements**: Vietnamese NLP expertise may be scarce
