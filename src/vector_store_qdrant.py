#step 4
import os
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from src import config
from src.embeddings import create_embeddings

from src.logger import get_logger
# a client is an app's connection to talk to qdrant cloud
logger = get_logger(__name__)

def build_vector_store(chunks):
    """
    Embed every chunk and upload it into a Qdrant collection
    """
    logger.info(f"Embeding {len(chunks)} and upload it into a Qdrant collection {config.QDRANT_COLLECTION_NAME}")
    embedding_model = create_embeddings()
    vector_store = QdrantVectorStore.from_documents(
        chunks,
        embedding_model,
        url = config.QDRANT_URL,
        api_key = config.QDRANT_API,
        collection_name = config.QDRANT_COLLECTION_NAME
    )
    logger.info(f"Qdrant collection {config.QDRANT_COLLECTION_NAME} has been succesfully built!!")
    return vector_store

#no need of save_vec tor_store_here


def load_vector_store():
    """
    Connect to a Qdrant Cloud collection that was already built before.
    """
    logger.info("Connecting to Qdrant cloud")
    embeddings = create_embeddings()
    vector_store = QdrantVectorStore.from_existing_collection(
        collection_name=config.QDRANT_COLLECTION_NAME,
        embedding=embeddings,
        url = config.QDRANT_URL,
        api_key = config.QDRANT_API,
    )
    logger.info("Connected to Qdrant cloud Sccesfully!")

    return vector_store

def check_vector_store_exists() -> bool:
    """Check whether a saved Qdrant store already exists."""
    client = QdrantClient(
        url=config.QDRANT_URL,
        api_key=config.QDRANT_API
    )
    return client.collection_exists(
        collection_name=config.QDRANT_COLLECTION_NAME
    )

def get_semantic_retriever(vector_store, k: int = config.TOP_K_RESULTS):
    """Convert the vector store into a semantic retriever
    that returns the top-k matching chunks"""
    return vector_store.as_retriever(search_kwargs={"k":k})

