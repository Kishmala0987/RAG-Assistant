#Step 3
from langchain_community.embeddings import JinaEmbeddings
from src import config
from src.logger import get_logger

logger = get_logger(__name__)
def create_embeddings():
    "Return JINA Embeddings model"
    logger.info("Initialized Embedding model:", config.EMBEDDING_MODEL_NAME)
    return JinaEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)