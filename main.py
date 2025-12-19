import os

from pymilvus.orm import utility

from agent import build_agent
from utils import connect_milvus, store_collection, build_documents, read_from_milvus, search_deals
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

    print(f"Connecting to Milvus at {milvus_host}:{milvus_port}")
    connect_milvus()
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
        collection_name=COLLECTION_NAME)

    print("\nM&A deal ingestion pipeline completed successfully!")

def run_cli():
    print("M&A Deal Intelligence Assistant")
    print("Type 'exit' to quit\n")

    connect_milvus()
    app = build_agent()

    while True:
        query = input("Your question: ").strip()
        if not query:
            print("Please enter a valid question.\n")
            continue

        if query.lower() in {"exit", "quit"}:
            break

        result = app.invoke({"query": query})

        print("\nAnswer:\n")
        print(result["answer"])
        print("-" * 80 + "\n")


if __name__ == "__main__":

    #run_deal_ingestion()
    #read_from_milvus("ma_deals_knowledge")
    run_cli()
'''
    COLLECTION_NAME = "ma_deals_knowledge"

    print("M&A Deal Search CLI (type 'exit' to quit)")
    
    while True:
        query = input("\nSearch query: ").strip()
        if not query:
            print("No search query provided. Try again.")
            continue

        if query.lower() in {"exit", "quit"}:
            break

        results = search_deals(
            query=query,
            collection_name=COLLECTION_NAME,
            top_k=5,
            document_type="risks"
        )

        print("\nTop results:\n")
        for i, r in enumerate(results, 1):
            print(f"[{i}] Score: {r['score']:.4f}")
            print("Metadata:", r["metadata"])
            print("Risks:", r["risks"])
            print("-" * 60)
'''
