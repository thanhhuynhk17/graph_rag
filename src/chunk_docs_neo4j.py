import asyncio
import logging
import glob
from langchain_community.embeddings import OpenAIEmbeddings
from utils.helpers import process_and_embed_to_neo4j
import argparse
import os

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
    else:
        file_paths = []
        for ext in SUPPORTED_EXTS:
            file_paths.extend(glob.glob(f"data/34NghiQuyet/*.{ext}"))
        print(f"📂 Found {len(file_paths)} supported files in data/34NghiQuyet")

    print(file_paths)

    embeddings = OpenAIEmbeddings(
        model="Qwen/Qwen3-Embedding-0.6B",
        base_url="http://localhost:8000/v1",
        api_key="text",
        tiktoken_enabled=False
        # dimensions=1024
    )

    neo4j_url = "bolt://localhost:7687"
    username = "neo4j"
    password = "12345678"
    database = "neo4j"
    index_name = "vietnamese_docs"

    tasks = []
    for file_path in file_paths:
        tasks.append(
            process_and_embed_to_neo4j(
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