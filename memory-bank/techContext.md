# Technology Context

## Core Technology Stack

### Programming Languages & Frameworks
- **Python 3.12+**: Primary implementation language
- **TypeScript**: UI components and type definitions
- **React**: Frontend components (generative-ui.tsx)
- **FastAPI**: REST API framework with automatic OpenAPI docs
- **FastMCP**: Model Context Protocol server implementation

### Machine Learning & NLP
- **Qwen3-0.6B**: Vietnamese-language embeddings (OpenAI-compatible)
- **Qwen3-8B**: Advanced embedding model for production
- **BAAI/bge-reranker-v2-m3**: Cross-encoder reranking model
- **UnderTheSea**: Vietnamese NLP tokenization and POS tagging
- **LangChain**: Retrieval-Augmented Generation framework

### Database & Search
- **Neo4j 5.26**: Graph database with native vector search
- **APOC (Awesome Procedures on Cypher)**: Graph algorithms and utilities
- **Graph Data Science (GDS)**: Advanced graph analytics
- **Neomodel**: Python ORM for Neo4j graph operations

### Infrastructure & Deployment
- **Docker & Docker Compose**: Containerized deployment
- **VLLM**: High-performance LLM inference server
- **NVIDIA GPU**: CUDA acceleration for ML models
- **uv**: Fast Python package manager

## Development Environment Setup

### Local Development Requirements
```bash
# System Requirements
- Python 3.12+
- Node.js 18+ (for UI components)
- Docker & Docker Compose
- NVIDIA GPU (optional, for ML acceleration)
- 16GB+ RAM recommended
```

### Python Environment Management
```bash
# Using uv (recommended)
pip install uv
uv pip install -r requirements.txt --index-strategy unsafe-best-match

# Virtual environment activation
uv venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
```

### Environment Configuration (.env)
```bash
# Neo4j Graph Database
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=12345678
NEO4J_DATABASE=neo4j

# Embedding Service (Qwen3-0.6B)
OPENAI_BASE_URL_EMBED=http://localhost:8080/v1
OPENAI_API_KEY_EMBED=dummy_text
OPENAI_API_MODEL_NAME_EMBED=Qwen/Qwen3-Embedding-0.6B
EMBED_DIM=1024

# Reranker Service (BGE-Reranker)
OPENAI_BASE_URL_RERANK=http://localhost:8081
OPENAI_API_KEY_RERANK=dummy_text
OPENAI_API_MODEL_NAME_RERANK=BAAI/bge-reranker-v2-m3

# Application Behavior
AUTO_LOAD_DISHES=true
FORCE_REFRESH_DISHES=false
LOG_LEVEL=INFO
```

## Dependencies and Libraries

### Core Dependencies (pyproject.toml)

#### Graph Database & ORM
```toml
neomodel = "^5.3.1"        # Neo4j Python ORM
neo4j = "^5.23.0"          # Official Neo4j Python driver
```

#### Machine Learning & AI
```toml
langchain = "^0.2.14"      # RAG framework
langchain-openai = "^0.1.17"  # OpenAI-compatible API client
langchain-community = "^0.2.12"  # Additional retrievers
langchain-neo4j = "^0.4.0"     # Neo4j vector search integration
```

#### Vietnamese NLP
```toml
underthesea = "^6.8.4"     # Vietnamese NLP toolkit
```

#### Web Framework & API
```toml
fastapi = "^0.112.1"       # Modern async web framework
uvicorn = "^0.30.5"        # ASGI server
fastmcp = "^0.9.0"         # MCP protocol implementation
```

#### Utilities
```toml
python-dotenv = "^1.0.1"   # Environment variable management
pydantic = "^2.8.2"        # Data validation
pendulum = "^3.0.0"        # Datetime handling with timezone support
```

### New Utility Modules

#### Order Service Layer
```python
# src/utils/order_service.py
class OrderService:
    """Unified business logic for order operations across MCP and REST APIs"""

    async def create_api_response(self, request: OrderRequest) -> OrderResponse:
        """REST API order creation with structured response"""
        return await self._create_order_with_validation(request)

    def create_mcp_response(self, ...) -> ToolResult:
        """MCP tool order creation with structured content"""
        return self._create_order_with_validation_mcp(...)
```

#### Enhanced Schema Management
```python
# src/utils/pydantic_helpers.py
# Additional Pydantic utilities and schema enhancements
def create_response_model(name: str, **fields) -> BaseModel:
    """Dynamic Pydantic model creation helper"""
    pass
```

### Development Dependencies
```toml
pytest = "^8.3.2"          # Testing framework
pytest-asyncio = "^0.23.8" # Async test support
black = "^24.8.0"          # Code formatting
mypy = "^1.11.1"           # Type checking
```

## Tool Usage Patterns

### Code Quality & Development
```bash
# Testing
uv run pytest src/tests/ -v

# Code formatting
uv run black src/

# Type checking
uv run mypy src/

# Running MCP server
uv run uvicorn src.mcp_server:app --host 0.0.0.0 --port 8000 --reload

# Running with auto data loading
AUTO_LOAD_DISHES=true uv run uvicorn src.mcp_server:app --host 0.0.0.0 --port 8000 --reload
```

### Docker Deployment
```bash
# Full stack deployment
bash run_docker.sh

# Individual services
docker-compose up neo4j
docker-compose up embedding-service
docker-compose up reranker-service
docker-compose up app
```

### Data Processing
```bash
# Ingest CSV data to Neo4j
python src/chunk_docs_neo4j.py --file src/file_paths.txt
```

## Technical Constraints & Limitations

### Vietnamese Language Processing
- **Model Size**: Qwen3-0.6B requires ~4GB VRAM, 8B version needs ~16GB
- **Context Window**: 8192 tokens limit for embedding models
- **Tokenization**: UnderTheSea library may not handle all domain-specific terms
- **Cultural Context**: Vietnamese food terminology requires domain expertise

### Graph Database Constraints
- **Memory Usage**: Full graph loading may exceed RAM limits for large datasets
- **Query Complexity**: Complex Cypher queries can impact performance
- **Schema Flexibility**: Graph model requires careful relationship design
- **Backup/Restore**: APOC procedures needed for data management

### Performance Limitations
- **Cold Start**: ML model loading takes 30-60 seconds on startup
- **Concurrent Users**: Limited by Neo4j connection pool (default: 100 connections)
- **Batch Size**: Embedding generation limited by VRAM and batch processing
- **Network Latency**: External VLLM services introduce network overhead

### Infrastructure Constraints
- **GPU Dependency**: ML acceleration requires NVIDIA GPU with CUDA
- **Port Conflicts**: Multiple services require specific ports (7474, 7687, 8080, 8081, 8000)
- **Disk Space**: Models and embeddings require significant storage (~50GB+)
- **Container Networking**: Docker network configuration affects service discovery

## System Requirements

### Minimum Hardware Requirements
- **CPU**: 4-core x64 processor
- **RAM**: 16GB system memory
- **Storage**: 100GB available disk space
- **Network**: Stable internet for model downloads

### Recommended Hardware Requirements
- **CPU**: 8-core processor with AVX2 support
- **RAM**: 32GB+ system memory
- **GPU**: NVIDIA RTX 3060+ (12GB VRAM) or A100 (40GB VRAM)
- **Storage**: 500GB NVMe SSD for model caching
- **Network**: 100Mbps+ internet connection

### Software Requirements
- **OS**: Linux (Ubuntu 20.04+), macOS (12.0+), or Windows 11 with WSL2
- **Docker**: Version 24.0+ with Compose V2
- **Python**: Version 3.12+ with pip
- **Node.js**: Version 18+ (for UI development)

## Deployment Patterns

### Development Environment
- **Local Services**: Docker Compose with host networking
- **Hot Reload**: FastAPI auto-reload for rapid development
- **Debug Logging**: Configurable log levels for troubleshooting
- **Test Isolation**: Clean database setup between test runs

### Production Environment
- **Container Registry**: Pre-built Docker images
- **Orchestration**: Docker Compose or Kubernetes
- **Monitoring**: Application logs and health checks
- **Backup Strategy**: Neo4j database snapshots

### Cloud Deployment Considerations
- **GPU Instances**: AWS P3/P4, GCP A100 instances, or Azure NCv3 series
- **Storage**: Managed Neo4j service (Neo4j Aura) or self-hosted
- **Scaling**: Horizontal pod scaling for API services
- **Security**: Network isolation and API authentication

## Third-Party Service Integrations

### VLLM Model Serving
- **Local Deployment**: Docker container with GPU acceleration
- **OpenAI Compatibility**: Standard API interface for embeddings/reranking
- **Model Caching**: Persistent volume for model storage
- **Health Checks**: Service availability monitoring

### Neo4j Graph Database
- **Enterprise Features**: APOC and GDS plugins for advanced analytics
- **Vector Search**: Native vector similarity search capabilities
- **Cypher Queries**: Declarative graph query language
- **ACID Transactions**: Reliable multi-operation transactions

### CI/CD Considerations
- **Automated Testing**: Pytest integration with coverage reporting
- **Code Quality**: Black formatting and mypy type checking
- **Security Scanning**: Dependency vulnerability checks
- **Performance Testing**: Benchmark suites for search accuracy metrics
