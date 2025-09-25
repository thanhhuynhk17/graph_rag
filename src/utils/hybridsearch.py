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
from src.utils.helpers import normalize_vnese

class HybridSearchQuery(BaseModel):
    """
    Schema for retrieving restaurant and dish information 
    from the knowledge base.

    Supports:
    - Restaurant details (name, location, cuisine, opening hours, etc.)
    - Dish details (ingredients, cooking style, dietary info, etc.)

    Increasing 'k' allows fetching more candidate results for better coverage.
    """
    query: str = Field(
        ...,
        description=(
            "Natural language search query. "
            "Can be about restaurants (location, cuisine type, opening hours) "
            "or dishes (name, ingredients, dietary preferences, cooking style)."
        )
    )
    k: Literal[30,50] = Field(
        ...,
        description=(
            "Number of top results to retrieve. "
            "Use 30 for a smaller, faster search; 50 for broader coverage. "
            "If results are empty, increase k. If still empty, the requested restaurant or dish "
            "may not exist in the database."
        )
    )

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


def bm25_preprocessing_func(text: str) -> List[str]:
    normalized = normalize_vnese(text).lower()
    return normalized.split()

from src.utils.helpers import rerank_novita, rerank_cohere # for vllm localhost
# global variable
_pipeline = None  
def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = HybridRetrieverPipeline(
            bm25_preprocessing_func=bm25_preprocessing_func,
            rerank_func=rerank_cohere # rerank_novita / rerank_cohere
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

    query = normalize_vnese(query)
    pipeline = get_pipeline()
    results = pipeline.search(
        query,
        embedding_k=100, # fixed
        k=k,
        is_bm25_enable=is_bm25_enable,
    )
    return results
