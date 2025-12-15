import os
from utils import connect_milvus, store_collection, build_documents, read_from_milvus
from dotenv import load_dotenv
load_dotenv()
env = os.getenv("ENV")

def run_deal_ingestion():

    COLLECTION_NAME = "ma_deals_knowledge"
    DEALS_PATH = "data/deals"

    print("Starting M&A deal ingestion pipeline...")

    # Connect to Milvus
    milvus_host = os.getenv("MILVUS_HOST", "localhost")
    milvus_port = os.getenv("MILVUS_PORT", "19530")
    milvus_password = os.getenv("MILVUS_PASSWORD", "")

    print(f"Connecting to Milvus at {milvus_host}:{milvus_port}")
    connect_milvus(milvus_host, milvus_port)
    print("Milvus connected successfully")

    # Build documents from deals
    print(f"Building documents from deals at: {DEALS_PATH}")
    documents = build_documents(DEALS_PATH)
    print(f"Prepared {len(documents)} documents")

    if not documents:
        print("No documents found. Exiting.")
        return

    # Store in Milvus
    print("Storing deal documents in Milvus...")
    store_collection(
        deals=documents,
        collection_name=COLLECTION_NAME,
        embed_model="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("\nM&A deal ingestion pipeline completed successfully!")


if __name__ == "__main__":
    # run_deal_ingestion()
    milvus_host = os.getenv("MILVUS_HOST", "localhost")
    milvus_port = os.getenv("MILVUS_PORT", "19530")
    read_from_milvus(milvus_host, milvus_port)
