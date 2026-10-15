#step 4
import os
from langchain_community.vectorstores import FAISS
from src import config
from src.embeddings import create_embeddings

from src.logger import get_logger

logger = get_logger(__name__)

def build_vector_store(chunks):
    """
    Embed every chunk and build a searchable fais index in a memory
    """
    embedding_model = create_embeddings()
    vector_store = FAISS.from_documents(chunks, embedding_model)
    logger.info("Vector store has been succesfully built!!")
    return vector_store

def save_vector_store(vector_store, file_path:str = config.VECTOR_STORE_PATH) ->  None:
    """
    Save the faiss index in a disk so we dont have to built it every time
    """
    vector_store.save_local(file_path)
    logger.info("Saved the faiss index in a disk")


def load_vector_store(file_path:str = config.VECTOR_STORE_PATH):
    embeddings = create_embeddings()
    return FAISS.load_local(file_path, embeddings, allow_dangerous_deserialization=True)

def check_vector_store_exists(file_path:str = config.VECTOR_STORE_PATH) -> bool:
    """Check whether a saved FAISS index exists on disk."""
    return os.path.exists(os.path.join(file_path, "index.faiss" ))

def get_semantic_retriever(vector_store, k: int = config.TOP_K_RESULTS):
    """Convert the vector store into a semantic retriever
    that returns the top-k matching chunks"""
    return vector_store.as_retriever(search_kwargs={"k":k})

