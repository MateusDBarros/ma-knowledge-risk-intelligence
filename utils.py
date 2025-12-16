import logging, os, json
from pathlib import Path
from typing import List
from dotenv import load_dotenv
from langchain_core.documents import Document
from pymilvus import connections, Collection, FieldSchema, DataType, CollectionSchema
from pymilvus.orm import utility
from sentence_transformers import SentenceTransformer


logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)

load_dotenv()
env = os.getenv("ENV")

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
_embed_model = None

def get_sentence_transformer():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBED_MODEL)
    return _embed_model


def embed_documents(documents: List[Document]) -> List[list]:
    model = get_sentence_transformer()
    texts = [d.page_content for d in documents]
    vectors = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return vectors.tolist()

def connect_milvus():
    milvus_host = os.getenv("MILVUS_HOST")
    milvus_port = os.getenv("MILVUS_PORT")
    milvus_password = os.getenv("MILVUS_APIKEY")

    if milvus_host == 'localhost':
        connections.connect(host=milvus_host, port=milvus_port)
    else:
        connections.connect (
            host=milvus_host,
            port=milvus_port,
            secure=True,
            server_name=milvus_host,
            user="ibmlhapikey_MateusBarros@ibm.com",
            password=milvus_password)
    logger.info(f"Connection established {milvus_host}:{milvus_port}")


def read_from_milvus():
    connect_milvus()
    logger.info(f"Milvus connected")

    collections = utility.list_collections()

    if not collections:
        logger.info(f"No collections found in Milvus, exiting")
        return

    for collection_name in collections:
        logger.info(f"Reading from collection {collection_name}")

        collection = Collection(collection_name)
        collection.load()

        logger.info(f"Reading from collection {collection_name}")
        for field in collection.schema.fields:
            logger.info(f"Reading from field {field}")

        expr = " "
        output_field = ["id", "summary", "risks", "outcome"]

        try:
            results = collection.search(expr=expr, output_fields=[output_field], limit=10)
            if results:
                logger.info(f"Found {len(results)} results for {collection_name}")
                for result in results:
                    logger.info(f"Result: {result}")
            else:
                logger.info(f"No results for {collection_name}")
        except Exception as e:
            logger.info(f"Exception: {e}")



def create_collection(collection_name: str, dim: int = 384):

    if utility.has_collection(collection_name):
        logger.info(f"Collection {collection_name} already exists")
        return Collection(collection_name)

    fields = [
        FieldSchema(name="id",dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="summary", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="risks", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="outcome" , dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="metadata" , dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim)
    ]

    schema = CollectionSchema(fields=fields, description="Collection schema")
    collection = Collection(name=collection_name, schema=schema)

    index_params = {
        "metric_type": "IP",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 1024},
    }
    collection.create_index(index_params=index_params, field_name="embedding")
    collection.load()
    logger.info(f"Created collection {collection_name}")
    return collection



def store_collection(deals: List[Document], collection_name: str):

    model = get_sentence_transformer()
    collection = create_collection(collection_name)

    summaries, texts_for_embedding, risks, outcomes, metadata = [], [], [], [], []
    for doc in deals:
        text = doc.page_content.strip()
        if not text:
            continue

        doc_type = doc.metadata.get("document_type")

        if doc_type == "summary":
            summaries.append(text)
            risks.append("")
            outcomes.append("")
        elif doc_type == "risks":
            summaries.append("")
            risks.append(text)
            outcomes.append("")
        elif doc_type == "outcome":
            summaries.append("")
            risks.append("")
            outcomes.append(text)
        else:
            summaries.append(text)
            risks.append("")
            outcomes.append("")

        metadata.append(json.dumps(doc.metadata))
        texts_for_embedding.append(text)

    if not texts_for_embedding:
        logger.info(f"No embeddings found for {collection_name}")
        return

    embeddings = model.encode(texts_for_embedding, show_progress_bar=True, convert_to_numpy=True).tolist()

    collection.insert([summaries, risks, outcomes, metadata, embeddings])
    collection.flush()
    logger.info(f"Inserted {len(embeddings)} embeddings for {collection_name}")


def build_documents(deals_root:str):

    deals_path = Path(deals_root)
    documents = []

    for deal_dir in deals_path.iterdir():
        if not deal_dir.is_dir():
            continue

        metadata_path = deal_dir / "metadata.json"
        if not metadata_path.is_file():
            continue

        with open(metadata_path) as f:
            base_metadata = json.load(f)

        doc_map = {
            "summary.txt": "summary",
            "risks.txt": "risks",
            "outcome.txt": "outcome"
        }

        for file_name, doc_type in doc_map.items():
            file_path = deal_dir / file_name
            if not file_path.exists():
                continue

            with open(file_path, encoding="utf-8") as f:
                text = f.read().strip()

            if not text:
                continue

            documents.append(Document(
                page_content=text,
                metadata={**base_metadata, "document_type": doc_type}
            ))

    return documents


def search_deals(query: str, collection_name: str, top_k: int, sector: str | None = None, document_type: str | None = None):

    connect_milvus()

    collection = Collection(collection_name)
    collection.load()

    model = get_sentence_transformer()
    query_embedding = model.encode([query])[0].tolist()

    expr = []
    if sector:
        expr.append(f'metadata like "%\\"sector\\": \\"{sector}\\"%"')
    if document_type:
        expr.append(f'metadata like "%\\"document_type\\": \\"{document_type}\\"%"')

    filter_expr = " and ".join(expr) if expr else None

    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param={"metric_type": "IP", "params": {"nprobe": 10}},
        limit=top_k,
        expr=filter_expr,
        output_fields=["summary", "risks", "outcome", "metadata"]
    )

    documents = []
    for hit in results[0]:
        documents.append({
            "score": hit.score,
            "summary": hit.entity.get("summary"),
            "risks": hit.entity.get("risks"),
            "outcome": hit.entity.get("outcome"),
            "metadata": hit.entity.get("metadata")
        })

    return documents
