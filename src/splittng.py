#Step 2
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src import config
from src.logger import get_logger

logger = get_logger(__name__)

def split_into_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = config.CHUNK_SIZE,
    chunk_overlap  = config.CHUNK_OVERLAP 
    )
    logger.info("Initialized a Recursive Character Text Splitter object")
    chunks = text_splitter.split_documents(documents)
    logger.info("Documents are split into chunks. Total Chunks:", len(chunks))
    return chunks