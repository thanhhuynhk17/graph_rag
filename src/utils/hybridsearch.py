#!/usr/bin/env python3
"""
hybrid_search.py

This module provides an implementation of a hybrid search retriever using both keyword-based BM25 and
vector-based similarity search from Neo4j. It is designed for use with the MCP (Model Context Protocol)
tool or similar document/question answering frameworks.

It includes:
- Loading environment variables from a .env file.
- Setting up HuggingFace embedding for Neo4j vector search.
- Preprocessing Vietnamese text for BM25.
- Aggregating both retrievers using LangChain's EnsembleRetriever.

Usage:
    You can run this file directly to test hybrid search results.
    Or you can import `hybrid_search()` from this module into your MCP pipeline.

Environment Variables Required:
- NEO4J_URI
- NEO4J_USER
- NEO4J_PASSWORD
- NEO4J_DATABASE

Example:
    from hybrid_search import HybridSearchQuery, hybrid_search
    pipeline = HybridRetrieverPipeline()  # preload once
    results = await hybrid_search(HybridSearchQuery(query="your question", k=10))
"""

from dotenv import load_dotenv
import os

# Access environment variables
load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE")

print(f"NEO4J_URI: {NEO4J_URI}")
print(f"USERNAME: {NEO4J_USER}")
print(f"DATABASE: {NEO4J_DATABASE}")

from langchain_community.retrievers import BM25Retriever
from langchain_neo4j import Neo4jVector
from langchain.retrievers import EnsembleRetriever
from neo4j import GraphDatabase
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

from typing import List, Literal, Callable, Dict, Any
from pydantic import BaseModel, Field

class HybridSearchQuery(BaseModel):
    query: str = Field(..., description="""
                       Tool này được sử dụng sau cùng để tiết kiệm tài nguyên.

                       Từ khóa món ăn + loại món ăn (món cá, món khai vị, món ăn chơi, món rau, món gỏi, món gà, vịt & trứng, món tôm & mực, món xào, nước mát nhà làm, lẩu, món thịt, món sườn & đậu hũ, món canh, các loại khô, tráng miệng), ví dụ: 'món cá, cá kho'
                       """)
    k: Literal[5, 10, 20] = Field(5, description="Minimum 5, maximum 20. Số lượng gợi ý món ăn liên quan nhất.")

class HybridRetrieverPipeline:
    """
    Hybrid search with BM25 + Neo4j, re-ranked by Cohere reranker.

    Call `search(query)` to:
    1. Retrieve candidate documents.
    2. Apply Cohere Rerank v2.
    """

    def __init__(
        self,
        bm25_preprocessing_func: Callable,
        rerank_func: Callable[[str, List[str], int], List[Dict[str, Any]]]
    ):
        load_dotenv()

        # Preload docs & retrievers
        self.docs = load_neo4j_documents()
        self.bm25 = BM25Retriever.from_documents(
            documents=self.docs,
            preprocess_func=bm25_preprocessing_func
        )
        self.neo4j = load_neo4j_retriever()
        # Assign reranker function
        self.rerank_func = rerank_func

        # # Cohere clients (rerank)
        # self.reranker = cohere.ClientV2(
        #     api_key=os.getenv("OPENAI_API_KEY_EMBED", None),
        #     base_url=os.getenv("OPENAI_BASE_URL_EMBED", None)
        # )
        # self.rerank_model = os.getenv("OPENAI_API_MODEL_NAME_RERANK", None)

    def get_ensemble(self, k: int , is_bm25_enable=True):
        """Get an ensemble retriever with top-k results."""
        self.bm25.k = k
        neo4j_retriever = load_neo4j_retriever(k=k)
        if is_bm25_enable:
            return EnsembleRetriever(
                retrievers=[self.bm25, neo4j_retriever],
                id_key="id"
            )
            
        return EnsembleRetriever(
            retrievers=[neo4j_retriever], # embedding only for now to test performance
            id_key="id"
        )
        

    def search(self, query: str, embedding_k: int = 100, k: int = 5, is_bm25_enable=True):
        """Retrieve with ensemble, then rerank with Cohere."""
        ensemble = self.get_ensemble(k=embedding_k, is_bm25_enable=is_bm25_enable)
        docs = ensemble.invoke(query, config={})
        docs_reranked = self.rerank_func(
            query, 
            [ d.page_content.strip() for d in docs],
            top_n=k
        )
        
        if isinstance(docs_reranked,dict): # novita
            reranked_idxes = [ r["index"] for r in docs_reranked["results"]]
        else: # cohere
            reranked_idxes = [ r.index for r in docs_reranked.results]
        
        # Reorder origin_documents based on rerank_results order
        sorted_documents = [docs[r] for r in reranked_idxes]
        
        return sorted_documents


def load_neo4j_retriever(k: int = 50):

    embedder = OpenAIEmbeddings(
        model=os.getenv("OPENAI_API_MODEL_NAME_EMBED", None),
        base_url=os.getenv("OPENAI_BASE_URL_EMBED", None),
        api_key=os.getenv("OPENAI_API_KEY_EMBED", None),
        tiktoken_enabled=False,
        dimensions=int(os.getenv("EMBED_DIM"))
    )

    text_node_properties = ["text"]
    embedding_node_property = "embedding"
    fts_name = "keyword"
    embed_name = "vietnamese_docs"

    query = f"""
    RETURN reduce(
        str = '',
        k IN {text_node_properties} |
        str + '\n' + k + ': ' + coalesce(node[k], '')
    ) AS text,
    node {{
        .*,
        `{embedding_node_property}`: Null,
        {', '.join(f"`{prop}`: Null" for prop in text_node_properties)}
    }} AS metadata,
    score
    """

    store = Neo4jVector.from_existing_graph(
        embedder,
        node_label="Chunk",
        url=NEO4J_URI,
        username=NEO4J_USER,
        password=NEO4J_PASSWORD,
        embedding_node_property=embedding_node_property,
        text_node_properties=text_node_properties,
        index_name=embed_name,
        keyword_index_name=fts_name,
        search_type="vector",
        database=NEO4J_DATABASE,
        retrieval_query=query
    )
    return store.as_retriever(search_type="similarity", search_kwargs={"k": k})


# FIXME: write a Cypher query to fetch all columns
def load_neo4j_documents() -> List[Document]:
    if not all([NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD]):
        raise EnvironmentError("Missing Neo4j credentials.")

    auth = (NEO4J_USER, NEO4J_PASSWORD)
    driver = GraphDatabase.driver(NEO4J_URI, auth=auth)

    def get_all_documents(tx):
        query = """
        MATCH (n:Chunk)
        RETURN n.id AS id, n._id as _id, n.text AS page_content, n.source AS source
        """
        return tx.run(query).data()

    with driver.session() as session:
        neo4j_docs = session.execute_read(get_all_documents)
    driver.close()

    return [
        Document(
            page_content=doc["page_content"],
            metadata={
                "id": doc["id"],
                "source": doc["source"], 
                "_id": doc["_id"]
                }
        )
        for doc in neo4j_docs
    ]

from pyvi import ViTokenizer, ViPosTagger
from src.utils import helpers
from underthesea import word_tokenize

def bm25_preprocessing_func(text: str) -> List[str]:
    """
    First: MÓN GÀ, VỊT & TRỨNG, Vịt kho gừng, Thịt vịt chặt khúc, kho cùng gừng, mắm, đường cho săn, Vịt, gừng, mắm, đường, món mặn, Gừng nồng, mùi vịt kho dậy mùi, 2-3 người
    Final: ['gà', 'vịt trứng', 'vịt kho gừng', 'thịt vịt chặt khúc', 'kho gừng', 'mắm', 'đường săn', 'vịt', 'gừng', 'mắm', 'đường', 'mặn', 'gừng nồng', 'mùi vịt kho dậy mùi', '2 3 người']
    
    First: MÓN GÀ, VỊT & TRỨNG, Trứng chiên thịt, Trứng gà đánh tan, trộn thịt băm, nêm gia vị, chiên vàng, Trứng gà, thịt ba chỉ băm, món béo, mặn, Trứng chiên vàng thơm, hành lá, 2-3 người
    Final: ['gà', 'vịt trứng', 'trứng chiên thịt', 'trứng gà đánh tan', 'trộn thịt băm', 'nêm gia vị', 'chiên vàng', 'trứng gà', 'thịt ba chỉ băm', 'béo', 'mặn', 'trứng chiên vàng thơm', 'hành lá', '2 3 người']

    First: NƯỚC MÁT NHÀ LÀM, Trà đá, Trà nấu, Trà, Nước, Trà, 1 người
    Final: ['nước mát', 'trà đá', 'trà nấu', 'trà', 'nước', 'trà', '1 người']
    
    First: NƯỚC MÁT NHÀ LÀM, Coca / 7 UP, Nước ngọt có gas, Nước ngọt, Soft Drink, Gas, 1 người
    Final: ['nước mát', 'coca 7 up', 'nước ngọt gas', 'nước ngọt', 'soft drink', 'gas', '1 người']
    
    First: NƯỚC MÁT NHÀ LÀM, Nước suối, Nước suối thanh lọc, Nước suối, Nước, Nước, 1 người
    Final: ['nước mát', 'nước suối', 'nước suối thanh lọc', 'nước suối', 'nước', 'nước', '1 người']
    
    First: NƯỚC MÁT NHÀ LÀM, Bia các loại (Tiger, Heineken, Saigon), Bia các loại, Vị lúa mạch, Beer, Bia, 1 người
    Final: ['nước mát', 'bia loại tiger', 'heineken', 'saigon', 'bia loại', 'vị lúa mạch', 'beer', 'bia', '1 người']
    """
    normalized = helpers.normalize_vnese(text)
    print("First:", normalized)
    normalized = helpers.clean_vietnamese_text(normalized)
    # print("0.1:", normalized)
    normalized = ' '.join(word_tokenize(normalized))
    # print("0.5:", normalized)
    normalized = ViTokenizer.tokenize(normalized)
    # print("1:", normalized)
    sequences = [str(helpers.normalize_record(text=seq)).lower() for seq in normalized.split(" , ")]
    # print("2:", sequences)
    sequences = [helpers.remove_stopwords_and_not_vi(text=seq) for seq in sequences]
    # print("3:", sequences)
    sequences = [str(helpers.normalize_record(text=seq)).lower() for seq in sequences]

    sequences = [seq.replace("_", " ") for seq in sequences if seq]
    print("Final:", sequences)
    return sequences

from src.utils.helpers import rerank_novita, rerank_cohere # for vllm localhost
# global variable
_pipeline = None  
def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = HybridRetrieverPipeline(
            bm25_preprocessing_func=bm25_preprocessing_func,
            rerank_func=rerank_novita # rerank_novita / rerank_cohere
        )
    return _pipeline

def run_hybrid_search(query: str, k: int, is_bm25_enable: bool):
    """
    Perform hybrid search using BM25 and vector-based retrieval.

    This function normalizes the input query (supports Vietnamese text),
    loads the hybrid search pipeline, and retrieves relevant documents
    by combining BM25 lexical search with dense vector embeddings.

    Args:
        query (str): The search query string. Must not be None or empty.
        k (int): Number of top results to return.
        is_bm25_enable (bool): Whether to include BM25 in the hybrid search.
            - True → Combine BM25 and embedding search.
            - False → Use embedding-only retrieval.

    Returns:
        list[dict]: A list of search results, where each result is a dictionary
        containing metadata (e.g., document ID, score, content).

    Raises:
        ValueError: If `query` is None or empty.
    """
    if not query:
        raise ValueError("Query must not be None")

    query = helpers.normalize_vnese(query)
    pipeline = get_pipeline()
    results = pipeline.search(
        query,
        embedding_k=100, # fixed
        k=k,
        is_bm25_enable=is_bm25_enable,
    )
    return results
