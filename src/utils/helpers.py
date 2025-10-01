from __future__ import annotations

import os
import re
import unicodedata
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Protocol
from underthesea import word_tokenize, pos_tag

import pandas as pd
import requests
import cohere
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import CSVLoader
from langchain_neo4j import Neo4jVector
from pathlib import Path
from typing import Optional, Union
# VietnameseToneNormalization.md
# https://github.com/VinAIResearch/BARTpho/blob/main/VietnameseToneNormalization.md


class VietnameseTextProcessor:
    """Xử lý văn bản tiếng Việt: chuẩn hóa dấu, loại bỏ ký tự đặc biệt, tách từ, loại bỏ stopword."""
    
    TONE_NORM_VI = {
        'òa': 'oà', 'Òa': 'Oà', 'ÒA': 'OÀ',\
        'óa': 'oá', 'Óa': 'Oá', 'ÓA': 'OÁ',\
        'ỏa': 'oả', 'Ỏa': 'Oả', 'ỎA': 'OẢ',\
        'õa': 'oã', 'Õa': 'Oã', 'ÕA': 'OÃ',\
        'ọa': 'oạ', 'Ọa': 'Oạ', 'ỌA': 'OẠ',\
        'òe': 'oè', 'Òe': 'Oè', 'ÒE': 'OÈ',\
        'óe': 'oé', 'Óe': 'Oé', 'ÓE': 'OÉ',\
        'ỏe': 'oẻ', 'Ỏe': 'Oẻ', 'ỎE': 'OẺ',\
        'õe': 'oẽ', 'Õe': 'Oẽ', 'ÕE': 'OẼ',\
        'ọe': 'oẹ', 'Ọe': 'Oẹ', 'ỌE': 'OẸ',\
        'ùy': 'uỳ', 'Ùy': 'Uỳ', 'ÙY': 'UỲ',\
        'úy': 'uý', 'Úy': 'Uý', 'ÚY': 'UÝ',\
        'ủy': 'uỷ', 'Ủy': 'Uỷ', 'ỦY': 'UỶ',\
        'ũy': 'uỹ', 'Ũy': 'Uỹ', 'ŨY': 'UỸ',\
        'ụy': 'uỵ', 'Ụy': 'Uỵ', 'ỤY': 'UỴ'
        }

    # Repo-relative default stopwords path: src/data/stopwords-vietnamese.txt
    DEFAULT_STOPWORDS_PATH = Path(__file__).resolve().parent.parent / "data" / "stopwords-vietnamese.txt"

    def normalize_vnese(self, text: str)-> str:
        for i, j in self.TONE_NORM_VI.items():
            text = text.replace(i, j)
        # Remove control characters (ASCII 0–31, plus DEL 127)
        text = re.sub(r'[\x00-\x1F\x7F]', '', text)
        # normalize spacing
        text = text.replace('\xa0', ' ')
        # Normalize input text to NFC
        text = unicodedata.normalize("NFC", text)
        return text
    
    def load_stopwords(self, path: Optional[Path] = None) -> set[str]:
        path = Path(path) if path else Path(self.DEFAULT_STOPWORDS_PATH)
        if not path.exists():
            raise FileNotFoundError(path)
        return set(path.read_text(encoding="utf-8").splitlines())

    def remove_stopwords_vi(
        self,
        text: str,
        path_documents_vi: Optional[Union[str, Path]] = None,
        keep_nouns: bool = True,
    ) -> str:
        """Loại bỏ stop-word, nhưng LUÔN giữ lại danh từ."""
        sw_path = Path(path_documents_vi or self.DEFAULT_STOPWORDS_PATH)
        if not sw_path.exists():
            raise FileNotFoundError(f"Stopwords file not found: {sw_path}")
        stop_words = {w.strip().lower() for w in sw_path.read_text(encoding="utf-8").splitlines()}

        # Tách phần CSV (nếu có)
        tokens = text.split(",")
        if len(tokens) >= 4:
            id_str, link_str, name_str = tokens[:3]
            content = ",".join(tokens[3:])
        else:
            id_str = link_str = name_str = ""
            content = text

        # POS tagging
        pos_tags = pos_tag(content)  # nhận str, trả [(word, pos), ...]

        # Lọc: chỉ loại bỏ khi vừa là stop-word vừa KHÔNG phải danh từ
        filtered = [
            word
            for word, pos in pos_tags
            if word.lower() not in stop_words or pos.startswith("N")
        ]

        cleaned = " ".join(filtered)
        if id_str:
            return f"{id_str},{link_str},{name_str},{cleaned}"
        return cleaned

    def run_preprocess_sequences(self, data_source_path: str, stopwords_path: Optional[Union[str, Path]] = None) -> list[str]:
        """Load CSV, convert rows to strings, remove stopwords and normalize records."""
        df = CSVProcessor.load_excel(data_source_path)
        sequences = CSVProcessor.convert_table_to_rows(df)
        if stopwords_path is None:
            stopwords_path = str(self.DEFAULT_STOPWORDS_PATH)
        sequences = [self.remove_stopwords_vi(seq, stopwords_path) for seq in sequences]
        sequences = [str(self.normalize_record(seq)).lower() for seq in sequences]
        return sequences
    
    def clean_vietnamese_text(self, text: str) -> str:
        """Loại bỏ ký tự không phải chữ cái hoặc số tiếng Việt."""
        VIETNAMESE_CHARS = (
            "QWERTYUIOPASDFGHJKLZXCVBNMMM" # english uppercase
            "qwertyuiopasdfghjklzxcvbnm" # english lowercase
            "àáảãạăằắẳẵặâầấẩẫậ" # vietnamese
            "đ"
            "èéẻẽẹêềếểễệ"
            "ìíỉĩị"
            "òóỏõọôồốổỗộơờớởỡợ"
            "ùúủũụưừứửữự"
            "ỳýỷỹỵ"
            "ÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬ"
            "Đ"
            "ÈÉẺẼẸÊỀẾỂỄỆ"
            "ÌÍỈĨỊ"
            "ÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢ"
            "ÙÚỦŨỤƯỪỨỬỮỰ"
            "ỲÝỶỸỴ"
            "0123456789"  # numbers
            ",-" # keep comma as it is used as a separator
        )
        pattern = f"[^{VIETNAMESE_CHARS}]"
        cleaned_text = re.sub(pattern, " ", text)
        return re.sub(r'\s+', ' ', cleaned_text).strip()

    def normalize_record(self, text: str, fix_inch_heu=False) -> str:
        """Normalize a text record.

        fix_inch_heu is kept for backward compatibility with the Helpers wrapper.
        Currently it is unused but accepted to avoid calling signature mismatches.
        """
        if not text:
            return text
        text = re.sub(r'<.*?>', ' ', text).replace('\r', ' ').replace('\n', ' ').replace('\t', ' ')
        text = re.sub(r'\b[nN][aA][nN]\b', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

class CSVProcessor:
    """Đọc và xử lý file CSV."""
  
    @staticmethod
    def load_excel(path: str) -> pd.DataFrame:
        if not os.path.exists(path):
            raise FileNotFoundError(f"CSV file not found: {path}")
        df = pd.read_csv(path)
        # Normalize current_price
        df["current_price"] = (
            df["current_price"]
            .astype(str)
            .str.replace(r"[^\d]", "", regex=True)
            .replace("", None)
        )
        df["current_price"] = df["current_price"].astype(float).dropna().astype("Int64").astype(str) + " vnd"
        # df = add_column_names_to_values(df)
        return df
    
    @staticmethod
    def convert_table_to_rows(df: pd.DataFrame) -> list[str]:
        result_list = []
        for _, row in df.iterrows():
            row_string = ", ".join(str(v) for v in row)
            result_list.append(re.sub(r'<[^>]*>|\s+', ' ', row_string).strip())
        return result_list

class Neo4jHandler:
    """Thao tác với Neo4j: schema, load CSV, vector store."""
    
    def __init__(self, driver):
        self.driver = driver

    def init_schema(self): # okay
        statements = [
            # --- unique constraints ---
            "CREATE CONSTRAINT chunk_unique_id         IF NOT EXISTS FOR (c:Chunk)      REQUIRE c._id          IS UNIQUE",
            "CREATE CONSTRAINT customer_unique_id      IF NOT EXISTS FOR (cus:Customer) REQUIRE cus.customer_id IS UNIQUE",
            "CREATE CONSTRAINT order_unique_id         IF NOT EXISTS FOR (o:Order)      REQUIRE o.order_id     IS UNIQUE",
            "CREATE CONSTRAINT bill_unique_id          IF NOT EXISTS FOR (b:Bill)       REQUIRE b.bill_id      IS UNIQUE",
            "CREATE CONSTRAINT question_unique_id      IF NOT EXISTS FOR (q:Question)   REQUIRE q.question_id  IS UNIQUE",


            # --- regular indexes ---
            "CREATE INDEX customer_phone     IF NOT EXISTS FOR (cus:Customer) ON (cus.phone)",
            "CREATE INDEX customer_email     IF NOT EXISTS FOR (cus:Customer) ON (cus.email)",
            "CREATE INDEX order_customer     IF NOT EXISTS FOR (o:Order)      ON (o.customer_id)",
            "CREATE INDEX order_dish         IF NOT EXISTS FOR (o:Order)      ON (o.dish_id)",
            "CREATE INDEX question_customer  IF NOT EXISTS FOR (q:Question)   ON (q.customer_id)",
            "CREATE INDEX question_intent    IF NOT EXISTS FOR (q:Question)   ON (q.intent_name)",
        ]

        def _run_all(tx):
            for stmt in statements:
                tx.run(stmt)

        with self.driver.session() as session:
            session.execute_write(_run_all)

        self.driver.execute_query(
            'CREATE FULLTEXT INDEX chunkCombineFT IF NOT EXISTS FOR (c:Chunk) ON EACH [c.combine_info]'
        )

    def up_customer(self):
        query = """
        LOAD CSV WITH HEADERS FROM 'file:///customers.csv' AS row
        MERGE (cus:Customer {customer_id: row.customer_id})
        SET cus.full_name = row.full_name,
            cus._phone     = row._phone,
            cus.email     = row.email,
            cus.dob       = date(row.dob),
            cus.gender    = row.gender,
            cus.notes     = row.notes
        """
        with self.driver.session() as sess:
            sess.run(query=query)

    def up_order(self):
        query = """
        LOAD CSV WITH HEADERS FROM 'file:///orders.csv' AS row
        MERGE (o:Order {_id: row._id})
        SET o.customer_id  = row.customer_id,
            o.dish_id      = row.dish_id,
            o.order_time   = datetime(row.order_time),
            o.people_count = toInteger(row.people_count),
            o.unit_price   = toInteger(row.unit_price),
            o.quantity     = toInteger(row.quantity),
            o.notes        = row.notes
        MERGE (cus:Customer {customer_id: row.customer_id})
        MERGE (d:Chunk {_id: row.dish_id})
        MERGE (cus)-[:PLACED]->(o)-[:CONTAINS]->(d)
        MERGE (o)-[:ON]->(d) // add relationship Order -[ON]-> Chunk
        """
        with self.driver.session() as sess:
            sess.run(query=query)

    def up_question(self):
        query = """
        LOAD CSV WITH HEADERS FROM 'file:///questions.csv' AS row
        MERGE (q:Question {question_id: row.question_id})
        SET q._text        = row._text,
            q.created_at  = datetime(row.created_at)
        MERGE (cus:Customer {customer_id: row.customer_id})
        MERGE (cus)-[:ASKED]->(q)
        """
        with self.driver.session() as sess:
            sess.run(query=query)

    def init_data_neo4j(self) -> bool:

        self.init_schema()
        self.up_customer()
        self.up_order()
        self.up_question()
        
        return True

class EmbedToChunkNeo4j:
    """Nhúng dữ liệu thành chunks và lưu embedding vào Neo4j."""

    SUPPORTED_EXTS = ["csv"]

    def text_to_chunks(self, file_path: str, file_ext: str, chunk_size=512, chunk_overlap=0):
        print("🔄 Loading documents...")
        print("file_ext", file_ext)
        documents = []
        if file_ext == "csv":
            try:
                # get all columns
                df = pd.read_csv(file_path, nrows=1, encoding="utf-8")
                all_columns = df.columns.tolist()
                # Define which column will be the content
                content_column = "combined_info"
                # All other columns go into metadata (exclude content_column)
                metadata_columns = [col for col in all_columns if col != content_column]
                
                loader = CSVLoader(file_path,
                    csv_args={
                        'delimiter': ',',
                        'quotechar': '"'
                    },
                    content_columns=[content_column], # only this column goes into .page_content
                    metadata_columns=metadata_columns, # only this column goes into metadata
                    encoding='utf-8'
                )
                documents = loader.load()
                for d in documents:
                    if d.page_content.startswith("combined_info:"):
                        d.page_content = d.page_content.split("combined_info:", 1)[1].strip()

                    # chuẩn hóa văn bản
                    d.page_content = VietnameseTextProcessor().normalize_record(d.page_content)
            
            except Exception as e:
                print(
                    f"[BENCHMARK ERROR] {e}\n"
                    "CSV must have:\n"
                    "- column 'combined_info' (for embedding)\n"
                    "- column '_id' (for benchmark purpose)"
                )

        print("✂️ Splitting documents into chunks...")
        splitter = CharacterTextSplitter(separator="\n",chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        docs = splitter.split_documents(documents)
        return docs

    async def process_and_embed_to_neo4j(
        self,
        embedding_model,
        file_path: str,
        neo4j_url: str,
        username: str,
        password: str,
        database: str = "neo4j",
        chunk_size: int = 512,
        chunk_overlap: int = 0,
        index_name: str = "fulltext_index"
    ):
        """
        Load files, normalize Vietnamese content, split into chunks,
        and store embeddings in Neo4j with full-text search enabled.

        Args:
            embedding_model: embedding model.
            file_path (str): Path to a supported file.
            neo4j_url (str): URL for Neo4j (e.g., "bolt://localhost:7687").
            username (str): Neo4j username.
            password (str): Neo4j password.
            database (str): Neo4j database name (default: "neo4j").
            chunk_size (int): Size of each document chunk.
            chunk_overlap (int): Overlap between chunks.
            index_name (str): Name of the vector index in Neo4j.
        """
        if embedding_model is None:
            raise ValueError("embedding_model must be provided (e.g., OpenAIEmbeddings, HuggingFaceEmbeddings, etc.)")
        
        file_ext = os.path.splitext(file_path)[1]
        file_ext = file_ext[1:].lower()
        if file_ext not in self.SUPPORTED_EXTS:
            raise ValueError(f"Unsupported file extension: '{file_ext}'. Supported extensions are: {', '.join(self.SUPPORTED_EXTS)}.")

        docs = self.text_to_chunks(file_path, file_ext, chunk_size, chunk_overlap)
        
        print(f"💾 Saving embeddings to Neo4j (DB: {database})...")
        batch_size = 10
        for i in range(0, len(docs), batch_size):
            batch = docs[i : i + batch_size]
            
            Neo4jVector.from_documents(
                documents=batch,
                embedding=embedding_model,
                url=neo4j_url,
                username=username,
                password=password,
                database=database,
                index_name=index_name,
                search_type="hybrid",
            )

        print(f"✅ Embedded {len(docs)} chunks to Neo4j vector store (index: {index_name})")


def rerank_novita(
    query: str,
    documents: list[str],
    top_n: int,
    base_url: str | None = None,
    api_key: str | None = None,
    model: str | None = None
) -> list[dict]:
    """
    Call Novita's rerank API.

    Args:
        query (str): The search/query string.
        documents (list[str]): List of documents to rerank.
        model (str): Reranker model (default: baai/bge-reranker-v2-m3).
        top_n (int): How many top results to return.
        api_key (str): Novita API key (default: read from OPENAI_API_KEY_EMBED).
        base_url (str): Rerank endpoint.

    Returns:
        List of dicts with reranked results.
    """
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY_EMBED")
        if not api_key:
            raise ValueError("API key not provided and OPENAI_API_KEY_EMBED not set in .env")
    if base_url is None:
        base_url = os.getenv("OPENAI_BASE_URL_RERANK")
        if not base_url:
            raise ValueError("API key not provided and OPENAI_BASE_URL_RERANK not set in .env")
    if model is None:
        model = os.getenv("OPENAI_API_MODEL_NAME_RERANK")
        if not model:
            raise ValueError("API key not provided and OPENAI_API_MODEL_NAME_RERANK not set in .env")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "query": query,
        "documents": documents,
        "top_n": top_n
    }

    response = requests.post(f"{base_url}/rerank", headers=headers, json=payload)
    response.raise_for_status()  # raise error if status != 200

    return response.json()

def rerank_cohere(
    query: str,
    documents: list[str],
    top_n: int
    )->Dict:
    api_key = os.getenv("OPENAI_API_KEY_RERANK")
    base_url = os.getenv("OPENAI_BASE_URL_RERANK")
    model = os.getenv("OPENAI_API_MODEL_NAME_RERANK")
    if not api_key:
        raise ValueError("OPENAI_API_KEY_RERANK is not set")
    if not base_url:
        raise ValueError("OPENAI_BASE_URL_RERANK is not set")
    if not model:
        raise ValueError("OPENAI_API_MODEL_NAME_RERANK is not set")

    client = cohere.ClientV2(api_key=api_key, base_url=base_url)
    resp = client.rerank(model=model, query=query, documents=documents, top_n=top_n)

    results: list[dict] = []
    for r in resp.results:
        idx = getattr(r, "index", None)
        score = getattr(r, "relevance_score", None)
        text = documents[idx] if isinstance(idx, int) and 0 <= idx < len(documents) else None
        results.append({"index": idx, "relevance_score": score, "text": text})

    return {"results": results}

from cohere import ClientV2


class RerankerProvider(Protocol):
    def rerank(self, query: str, documents: List[str], top_n: int) -> List[Dict[str, Any]]:
        ...


class NovitaProvider:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY_EMBED")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL_RERANK")
        self.model = model or os.getenv("OPENAI_API_MODEL_NAME_RERANK")
        if not all((self.api_key, self.base_url, self.model)):
            raise ValueError("Thiếu thông tin Novita provider (api_key, base_url, model)")

    def rerank(self, query: str, documents: List[str], top_n: int) -> List[Dict[str, Any]]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": top_n,
        }
        resp = requests.post(f"{self.base_url}/rerank", headers=headers, json=payload)
        resp.raise_for_status()
        return resp.json()["results"]  # Novita trả về {"results": [...]}


class CohereProvider:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        api_key_val = api_key or os.getenv("OPENAI_API_KEY_RERANK")
        base_url_val = base_url or os.getenv("OPENAI_BASE_URL_RERANK")
        model_val = model or os.getenv("OPENAI_API_MODEL_NAME_RERANK")
        if not all((api_key_val, base_url_val, model_val)):
            raise ValueError("Thiếu thông tin Cohere provider (api_key, base_url, model)")
        # store as non-Optional strings
        self.api_key: str = str(api_key_val)
        self.base_url: str = str(base_url_val)
        self.model: str = str(model_val)
        self.client = ClientV2(api_key=self.api_key, base_url=self.base_url)

    def rerank(self, query: str, documents: List[str], top_n: int) -> List[Dict[str, Any]]:
        resp = self.client.rerank(model=self.model, query=query, documents=documents, top_n=top_n)
        return [
            {"index": r.index, "relevance_score": r.relevance_score, "text": documents[r.index]}
            for r in resp.results
        ]


class Reranker:
    """Factory-class: chỉ cần đổi tên provider, không đổi code gọi."""

    _providers = {"novita": NovitaProvider, "cohere": CohereProvider}

    def __init__(self, provider: str = "cohere", **kwargs: Any) -> None:
        if provider.lower() not in self._providers:
            raise ValueError(f"Provider '{provider}' không hợp lệ. Chọn trong {list(self._providers)}")
        self.engine: RerankerProvider = self._providers[provider.lower()](**kwargs)

    def rerank(self, query: str, documents: List[str], top_n: int = 3) -> List[Dict[str, Any]]:
        """Unified API – kết quả luôn có dạng [{"index": int, "relevance_score": float, "text": str}, ...]"""
        return self.engine.rerank(query, documents, top_n)
    
class Helpers:
    """Light wrapper class exposing helpers as methods for optional class-based usage.

    This keeps the original module-level functions intact for backward compatibility,
    but allows callers to import and use a `Helpers` instance if they prefer OOP style:

        from src.utils.helpers import helpers
        helpers.normalize_vnese(text)
        helpers.process_and_embed_to_neo4j(...)
    """

    def __init__(self):
        self.processor = VietnameseTextProcessor()
        self.embedder = EmbedToChunkNeo4j()

    def normalize_vnese(self, text: str):
        return self.processor.normalize_vnese(text)

    def load_excel(self, path: str) -> pd.DataFrame:
        # CSV loading belongs to CSVProcessor
        return CSVProcessor.load_excel(path)

    def load_stopwords(self, path: Optional[Union[str, Path]] = None) -> set[str]:
        return self.processor.load_stopwords(path)
    
    def convert_table_to_rows(self, df: pd.DataFrame) -> list[str]:
        return CSVProcessor.convert_table_to_rows(df)

    def remove_stopwords_and_not_vi(self, text: str, path_documents_vi: Optional[str] = None, keep_nouns: bool = True) -> str:
        return self.processor.remove_stopwords_vi(text, path_documents_vi, keep_nouns=keep_nouns)

    def clean_vietnamese_text(self, text: str) -> str:
        return self.processor.clean_vietnamese_text(text)

    def normalize_record(self, text: str, fix_inch_heu=False) -> str:
        return self.processor.normalize_record(text, fix_inch_heu)

    def run_preprocess_sequences(self, data_source_path: str, stopwords_path: Optional[str] = None) -> list[str]:
        return self.processor.run_preprocess_sequences(data_source_path, stopwords_path)

    def init_data_neo4j(self, driver):
        handler = Neo4jHandler(driver)
        return handler.init_data_neo4j()

    def text_to_chunks(self, file_path: str, file_ext: str, chunk_size=512, chunk_overlap=0):
        return self.embedder.text_to_chunks(file_path, file_ext, chunk_size, chunk_overlap)

    async def process_and_embed_to_neo4j(self, *args, **kwargs):
        return await self.embedder.process_and_embed_to_neo4j(*args, **kwargs)

    def rerank_novita(self, query: str, documents: list[str], top_n: int, base_url: str | None = None, api_key: str | None = None, model: str | None = None) -> list[dict]:
        return rerank_novita(query, documents, top_n, base_url, api_key, model)

    def rerank_cohere(self, query: str, documents: list[str], top_n: int) -> Dict:
        return rerank_cohere(query, documents, top_n)
    
    
# processor = VietnameseTextProcessor()
# helpers = Helpers()

# Re-export supported extensions for embeddings usage (e.g., chunk_docs_neo4j)
SUPPORTED_EXTS = EmbedToChunkNeo4j.SUPPORTED_EXTS

# Public API exported names
__all__ = [
    "SUPPORTED_EXTS",
    "EmbedToChunkNeo4j",
    "Helpers",
    "VietnameseTextProcessor",
    "CSVProcessor",
    "Neo4jHandler",
    "processor",
    "helpers"
]

# Quick usage:

# class VietnameseTextProcessor
# Usage:
# from src.utils.helpers import VietnameseTextProcessor
# processor = VietnameseTextProcessor()
# processor.normalize_vnese(text)
# processor.remove_stopwords_vi(text, path_documents_vi)
# etc.

# class Helpers (All tools :))
# from src.utils.helpers import helpers
# helpers.normalize_vnese(text)
# helpers.process_and_embed_to_neo4j(...)
# etc.

# class Reranker (Chưa test)
# Usage:
# from src.utils.helpers import Reranker
# reranker = Reranker(provider="cohere")  # or provider="novita"
# results = reranker.rerank(query, documents, top_n=3)

# class CohereProvider (Chưa test)
# Usage:
# from src.utils.helpers import CohereProvider
# cohere_provider = CohereProvider(api_key="your_api_key", base_url="your_base_url", model="your_model")
# results = cohere_provider.rerank(query, documents, top_n=3)