import os
from dotenv import load_dotenv
load_dotenv()

import asyncio
import logging
from langchain_openai import OpenAIEmbeddings
from utils.helpers import EmbedToChunkNeo4j
import argparse

# Configure logger
logging.basicConfig(
    filename="embedding_errors.log",
    level=logging.ERROR,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# Supported extensions
from utils.helpers import SUPPORTED_EXTS

# Define async main
async def main():
    parser = argparse.ArgumentParser(description="Embed supported files into Neo4j")
    parser.add_argument("--file", type=str, help="Path to a .txt file containing list of file paths")
    args = parser.parse_args()

    if args.file:
        try:
            with open(args.file, "r", encoding="utf-8") as f:
                file_paths = [
                    line.strip() for line in f
                    if line.strip() and os.path.splitext(line.strip())[1][1:].lower() in SUPPORTED_EXTS
                ]
                print(f"📄 Loaded {len(file_paths)} supported file paths from {args.file}")
        except FileNotFoundError:
            print(f"❌ File not found: {args.file}")
            return


    print(file_paths)

    model = os.getenv("OPENAI_API_MODEL_NAME_EMBED")
    base_url = os.getenv("OPENAI_BASE_URL_EMBED")
    api_key = os.getenv("OPENAI_API_KEY_EMBED")
    if not model or not base_url or not api_key:
        raise RuntimeError("Missing embedding env vars: OPENAI_API_MODEL_NAME_EMBED, OPENAI_BASE_URL_EMBED, OPENAI_API_KEY_EMBED")

    from pydantic import SecretStr
    embeddings = OpenAIEmbeddings(
        model=model,
        base_url=base_url,
        api_key=SecretStr(api_key),
        # dimensions=int(os.getenv("EMBED_DIM")),
        tiktoken_enabled=False,
    )

    neo4j_url = os.getenv("NEO4J_URI")
    username = os.getenv("NEO4J_USER")
    password = os.getenv("NEO4J_PASSWORD")
    database = os.getenv("NEO4J_DATABASE")
    if not neo4j_url or not username or not password or not database:
        raise RuntimeError("Missing Neo4j env vars: NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE")
    index_name = "vietnamese_docs"

    tasks = []
    embedder = EmbedToChunkNeo4j()
    for file_path in file_paths:
        tasks.append(
            embedder.process_and_embed_to_neo4j(
                file_path=file_path,
                neo4j_url=neo4j_url,
                username=username,
                password=password,
                database=database,
                embedding_model=embeddings,
                chunk_size=1,
                chunk_overlap=0,
                index_name=index_name
            )
        )

    results = await asyncio.gather(*tasks)
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logging.error(
                f"Error processing file '{file_paths[i]}': {result}",
                exc_info=True
            )
            print(f"❌ Failed: {file_paths[i]} - Check log for details")
        else:
            print(result)
            print(f"✅ Success: {file_paths[i]}")

# Run the async main
if __name__ == "__main__":
    asyncio.run(main())