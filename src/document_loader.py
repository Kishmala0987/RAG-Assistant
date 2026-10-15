#Step 1
from langchain_community.document_loaders import TextLoader
from src import config
from src.logger import get_logger

logger = get_logger(__name__)

def load_documents(file_path = config.DATA_FILE_PATH):
    """Load Documents into Text Loader"""
    text_loader = TextLoader(file_path, encoding="utf-8")
    logger.info("Loading documents from  loader", file_path)
    documents = text_loader.load()
    logger.info("Loaded documents - Length of documents:", len(documents))

    return documents