import pickle
from src import config
from langchain_community.retrievers import BM25Retriever
from src.logger import get_logger
from qdrant_client import QdrantClient
from langchain_core.documents import Document
from src import config
logger = get_logger(__name__)

def save_chunks(chunks, file_path:str = config.CHUNKS_PATH):
    """Save document chunks to disk."""
    with open(file_path, "wb") as f:
        pickle.dump(chunks, f)

def load_chunks(file_path: str = config.CHUNKS_PATH):
    """Load document chunks from disk."""

    with open(file_path, "rb") as f:
        return pickle.load(f)



def get_all_chunks_from_qdrant():
    client = QdrantClient(
        url=config.QDRANT_URL,
        api_key=config.QDRANT_API
    )

    chunks = []
    offset = None

    while True:
        points, offset = client.scroll(
            collection_name=config.QDRANT_COLLECTION_NAME,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False
        )

        for point in points:
            payload = point.payload or {}

            chunks.append(
                Document(
                    page_content=payload["page_content"],
                    metadata=payload.get("metadata", {})
                )
            )

        if offset is None:
            break

    return chunks

def get_bm25_retriever(chunks):
    """Create a BM25 keyword retriever from the document chunks."""
    logger.info("Using BM 25 keyword Retriever")
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = config.BM25_TOP_K
    return bm25_retriever
