# Product Context: SUPER NOVA RAG

## Why This Project Exists

Vietnamese e-commerce platforms struggle with accurate product search and matching, particularly for mobile phones where:
1. **Language Complexity**: Vietnamese has nuanced word variations, abbreviations (ví dụ: "điện thoại" → "ĐT", "phone" → "điện thoại")
2. **Product Specificity**: Mobile phones have complex specifications requiring precise matching
3. **Query Diversity**: Users ask questions in natural language mixing product names, specs, and requirements
4. **Data Quality Issues**: E-commerce data often contains inconsistencies and abbreviations

## Problems Solved

### Current E-commerce Search Problems
- **Poor Vietnamese Text Understanding**: Standard embedding models fail with Vietnamese morphological complexity
- **Inaccurate Product Matching**: Exact string matching misses semantic equivalents
- **No Context-Aware Search**: Can't understand relationships between products and specifications
- **Limited Scalability**: Traditional search systems struggle with large Vietnamese datasets

### SUPER NOVA RAG Solutions
- **Hybrid Vietnamese Retrieval**: Combines BM25 (keyword-based) + Qwen embeddings (semantic understanding)
- **Specialized Reranking**: ML-based reranking optimized for Vietnamese e-commerce queries
- **Graph-Based Context**: Neo4j enables understanding of product relationships and hierarchies
- **Real-time Performance**: Sub-500ms query processing for seamless user experience

## How It Should Work

### For End Users (E-commerce Customers)
**Natural Language Queries** → **Instant Accurate Results**
```
User Query: "Cho tôi hỏi điện thoại Samsung Galaxy S23 hiện tại giá bao nhiêu?"
System Response: Returns Samsung Galaxy S23 pricing and specifications immediately
```

**Flexible Search Patterns**:
- Product names in Vietnamese/English: "iPhone 15", "điện thoại iPhone 15"
- Spec-based queries: "điện thoại 8GB RAM", "pin 5000mAh"
- Mixed queries: "điện thoại Samsung dưới 10 triệu"
- Conversation queries: "Tôi muốn mua máy 5G có camera tốt"

### For Product Teams (Merchants)
**Automated Data Processing**:
- Upload CSV → Automatic graph building
- Real-time inventory synchronization
- Performance benchmarking and A/B testing

### For Developers (API Integration)
**Clean MCP Protocol**:
- Standardized tools for search, order management
- Consistent response formats
- Extensible architecture for new features

## User Experience Goals

### Performance (Speed & Accuracy)
- **Sub-500ms Response Time**: Users expect instant results
- **Hit@1 >85%**: First result should be correct 85% of the time
- **Hit@5 >95%**: Top 5 results should contain correct answer 95% of the time

### Reliability
- **99.9% Uptime**: Production-grade availability
- **Data Consistency**: Graph validation prevents conflicting information
- **Error Recovery**: Graceful handling of corrupted data or model failures

### Scalability
- **10,000+ Products**: Support large e-commerce catalogs
- **Concurrent Users**: Handle multiple simultaneous queries
- **Model Updates**: Easy deployment of improved ML models

## Success Metrics

### Technical Metrics
- Query Processing Latency: <500ms P95
- Vietnamese Query Accuracy: Hit@1 >85%
- System Availability: >99.9%
- Data Processing Throughput: >100 products/minute

### User Experience Metrics
- Customer Satisfaction Score: >4.5/5 stars
- Conversion Rate Improvement: >20% over baseline
- Search Session Duration: Reduced by 40%
- Support Ticket Reduction: >30% fewer "can't find product" issues

## Market Positioning

**The Vietnamese E-commerce Search Standard**
- First production-ready Vietnamese-specialized RAG system
- Benchmark dataset: Hoàng Hà Mobile (Vietnam's leading phone retailer)
- Open-source foundation with commercial deployment options

**Competitive Advantages**
- Specialized Vietnamese language models (Qwen3 fine-tuned)
- Hybrid retrieval combining traditional + AI approaches
- Graph-based data understanding (relationships, hierarchies)
- Production-hardened architecture from day one
