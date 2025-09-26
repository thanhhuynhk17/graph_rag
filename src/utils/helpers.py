import requests
import unicodedata
import re
import pandas as pd
from underthesea import word_tokenize
# VietnameseToneNormalization.md
# https://github.com/VinAIResearch/BARTpho/blob/main/VietnameseToneNormalization.md

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


def normalize_vnese(text):
    for i, j in TONE_NORM_VI.items():
        text = text.replace(i, j)
    # Remove control characters (ASCII 0–31, plus DEL 127)
    text = re.sub(r'[\x00-\x1F\x7F]', '', text)
    # normalize spacing
    text = text.replace('\xa0', ' ')
    # Normalize input text to NFC
    text = unicodedata.normalize("NFC", text)
    return text

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

def convert_table_to_rows(df: pd.DataFrame) -> list[str]:
    result_list = []
    for _, row in df.iterrows():
        row_string = ", ".join(str(v) for v in row)
        result_list.append(re.sub(r'<[^>]*>|\s+', ' ', row_string).strip())
    return result_list

def remove_stopwords_vi(text: str, path_documents_vi: str='stopwords-vietnamese.txt') -> str:
    if not os.path.exists(path_documents_vi):
        raise FileNotFoundError(f"Stopwords file not found: {path_documents_vi}")
    
    tokens = text.split(',')
    id_str, link_str, name_str = tokens[:3]
    content = ','.join(tokens[3:])
    
    stop_words = set(open(path_documents_vi, encoding="utf-8").read().splitlines())
    filtered_tokens = [w.strip() for w in word_tokenize(content, format="text").split(',') if w.strip().lower() not in stop_words]
    
    return f"{id_str},{link_str},{name_str}," + ', '.join(filtered_tokens)

def normalize_record(text: str, fix_inch_heu=False) -> str:
    if not text:
        return text
    text = re.sub(r'<.*?>', ' ', text).replace('\r', ' ').replace('\n', ' ').replace('\t', ' ')
    text = re.sub(r'\b[nN][aA][nN]\b', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def run_preprocess_sequences(data_source_path: str, stopwords_path: str = "./src/data/stopwords-vietnamese.txt") -> list[str]:
    
    df = load_excel(data_source_path)
    sequences = convert_table_to_rows(df)
    sequences = [remove_stopwords_vi(seq, stopwords_path) for seq in sequences]
    sequences = [str(normalize_record(seq)).lower() for seq in sequences]
    return sequences

def init_data_neo4j(driver):
    
    def init_schema(driver): # okay
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

        with driver.session() as session:
            session.execute_write(_run_all)
            
        driver.execute_query(
            'CREATE FULLTEXT INDEX chunkCombineFT IF NOT EXISTS FOR (c:Chunk) ON EACH [c.combine_info]'
        )    

    def up_chunk(driver):
        query = """
        LOAD CSV WITH HEADERS FROM 'file:///dishes.csv' AS row
        MERGE (c:Chunk {_id: row._id})
        SET c.name_of_food          = row.name_of_food,
            c.type_of_food          = row.type_of_food,
            c.how_to_prepare        = row.how_to_prepare,
            c.main_ingredients      = row.main_ingredients,
            c.taste                 = row.taste,
            c.outstanding_fragrance = row.outstanding_fragrance,
            c.current_price         = row.current_price,
            c.number_of_people_eating = row.number_of_people_eating,
            c.combine_info          = row.combine_info
        """
        with driver.session() as sess:
            sess.run(query=query)
        
        
    def up_customer(driver):
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
        with driver.session() as sess:
            sess.run(query=query)
        
    def up_orther(driver):
        query = """
        LOAD CSV WITH HEADERS FROM 'file:///orders.csv' AS row
        MERGE (o:Order {_id: row._id})
        SET o.customer_id  = row.customer_id,
            o.dish_id      = row.dish_id,
            o.order_time   = datetime(row.order_time),
            o.people_count = toInteger(row.people_count),
            o.unit_price   = toInteger(row.unit_price),
            o.quantity     = toInteger(row.quantity)
        MERGE (cus:Customer {customer_id: row.customer_id})
        MERGE (d:Chunk {_id: row.dish_id})
        MERGE (cus)-[:PLACED]->(o)-[:CONTAINS]->(d)
        """
        with driver.session() as sess:
            sess.run(query=query)
    
    def up_question(driver):
        query = """
        LOAD CSV WITH HEADERS FROM 'file:///questions.csv' AS row
        MERGE (q:Question {question_id: row.question_id})
        SET q._text        = row._text,
            q.created_at  = datetime(row.created_at)
        MERGE (cus:Customer {customer_id: row.customer_id})
        MERGE (cus)-[:ASKED]->(q)
        """
        with driver.session() as sess:
            sess.run(query=query)
    
    init_schema(driver)
    # up_chunk(driver)
    
    up_customer(driver)
    up_orther(driver)
    up_question(driver)


from langchain_community.document_loaders import Docx2txtLoader, JSONLoader, CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_neo4j import Neo4jVector

import cohere # reranker

import pandas as pd
import os

def text_to_chunks(file_path: str, file_ext: str, chunk_size=512, chunk_overlap=0):
    print("🔄 Loading documents...")
    print("file_ext", file_ext)
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
                d.page_content = normalize_vnese(d.page_content)
        
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
    # for d in docs:
    #     d.page_content = f"{prefix_txt}\n\n{d.page_content}"
        
    return docs

SUPPORTED_EXTS = ["csv"]

async def process_and_embed_to_neo4j(
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
    Load .docx files, normalize Vietnamese content, split into chunks,
    and store embeddings in Neo4j with full-text search enabled.

    Args:
        embedding_model: embedding model.
        file_path_pattern (str): Glob pattern for ["docx", "json"] files.
        neo4j_url (str): URL for Neo4j (e.g., "bolt://localhost:7687").
        username (str): Neo4j username.
        password (str): Neo4j password.
        database (str): Neo4j database name (default: "neo4j").
        chunk_size (int): Size of each document chunk.
        chunk_overlap (int): Overlap between chunks.
        index_name (str): Name of the full-text index in Neo4j.
    """
    if embedding_model is None:
        raise ValueError("embedding_model must be provided (e.g., OpenAIEmbeddings, HuggingFaceEmbeddings, etc.)")
    
    file_ext = os.path.splitext(file_path)[1] # get ext
    file_ext = file_ext[1:] # remove the dot
    if file_ext not in SUPPORTED_EXTS:
        raise ValueError(f"Unsupported file extension: '{file_ext}'. Supported extensions are: {', '.join(SUPPORTED_EXTS)}.")

    docs = text_to_chunks(file_path, file_ext, chunk_size, chunk_overlap)
    
    
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
    # return vectorstore
    


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

from typing import List, Dict, Any

def rerank_cohere(
    query: str,
    documents: list[str],
    top_n: int
    )->Dict:
    client = cohere.ClientV2(
        api_key=os.getenv("OPENAI_API_KEY_RERANK", None),
        base_url=os.getenv("OPENAI_BASE_URL_RERANK", None)
    )
    response = client.rerank(
        model=os.getenv("OPENAI_API_MODEL_NAME_RERANK", None),
        query=query,
        documents=documents,
        top_n=top_n
    ) # response.results is a list of {index, relevance_score, document["text"]}
    
    results: list[dict] = [
        {
            "index": r.index,
            "relevance_score": r.relevance_score,
            "text": r.document["text"] if isinstance(r.document, dict) else r.document
        }
        for r in response.results
    ]
    response={
        "results" : results
    }
    return response