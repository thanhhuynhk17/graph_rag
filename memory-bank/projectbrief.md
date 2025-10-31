# Project Brief: SUPER NOVA RAG

## Core Purpose
Build a high-performance hybrid Retrieval-Augmented Generation (RAG) pipeline specifically optimized for Vietnamese mobile phone e-commerce data, combining Neo4j graph storage with advanced ML embeddings and reranking models.

## Core Requirements

### Functional Requirements
1. **Data Ingestion**: Process CSV files containing mobile phone product data with Vietnamese text
2. **Hybrid Retrieval**: Implement ensemble BM25 + embedding-based search
3. **Reranking**: Apply ML-based reranking for improved accuracy
4. **Neo4j Integration**: Store and query data as graph structures
5. **API Services**: Provide MCP server and REST API endpoints
6. **Order Management**: Graph-based order processing system with validation

### Performance Requirements
- Hit@1: Target >85% accuracy for Vietnamese product queries
- Hit@5: Target >95% accuracy
- Support for large datasets (1000+ products)
- Real-time query processing (<500ms response time)

### Technical Requirements
- Use Qwen3 models (0.6B and 8B) for embeddings
- BAAI/bge-reranker-v2-m3 for reranking
- Neo4j for graph storage with APOC and GDS plugins
- Python-based implementation with async processing
- Docker containerization for deployment
- MCP protocol support for tool integration

## Success Criteria
- Vietnamese language query understanding and product matching
- Superior performance over pure embedding-only retrieval
- Reliable graph-based order management
- Scalable architecture for future expansion
- Clear benchmarks and evaluation metrics

## Scope Boundaries
- Focus on Vietnamese mobile phone e-commerce data
- MVP scope: CSV ingestion, hybrid search, reranking, basic order management
- Future scope: Multi-language support, additional data types, advanced UI

## Key Stakeholders
- Data scientists (Vietnamese NLP focus)
- Backend engineers (graph databases, ML integration)
- DevOps (containerization, deployment)
- Product team (e-commerce domain expertise)

## Timeline Milestones
- Phase 1: Data ingestion and basic retrieval (complete)
- Phase 2: Hybrid search optimization (complete)
- Phase 3: Order management system (in progress)
- Phase 4: Production deployment preparation
