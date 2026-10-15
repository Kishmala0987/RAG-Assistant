
import os
from dotenv import load_dotenv

load_dotenv()

# Load environment variables from .env file
GROQ_API_KEY = os.getenv("GROQ_API_KEY") #sicne ii have already give it to portkey platform I dont need to write this api key here anymore
JINA_API_KEY = os.getenv("JINA_API_KEY")

#Tracing
LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "false")
LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY") 
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT")

# DATA PATH of DATA and VECTORS
DATA_FILE_PATH = os.path.join("data", "hr_policy.txt")

#persistant memory: It is storing the data in folder so it can be used later on also instead of storig it again -> doing it for vector database.

#VECTOR_STORE_PATH = os.path.join("data", "faiss_index") # fais_index  is constructed automatically
CHUNKS_PATH =os.path.join("data", "chunks.pkl")

#CLOUD MEMORY
QDRANT_API = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME")


#LLM Gateway Portkey
PORTKEY_API_KEY = os.getenv("PORTKEY_API_KEY")

#MODELS
LLM_MODEL_NAME = "openai/gpt-oss-120b"
EMBEDDING_MODEL_NAME = "jina-embeddings-v2-base-en"
RERANKING_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"

#Guard Model
GUARD_MODEL_NAME = "openai/gpt-oss-safeguard-20b"





# chunk/ text splitting
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

TOP_K_RESULTS = 20
BM25_TOP_K = 5

RRF_K = 60

def check_api_keys():
    if not GROQ_API_KEY:
        raise ValueError("Missing GROQ API key")
    if not JINA_API_KEY:
            raise ValueError("Missing Embeeding model API key")

SYSTEM_PROMPT = """
    You are a friendly Hr assistant. 
    Always use the search hr_policy tool to look up facts 
    before answering. If the answer isn't in the search 
    results, say you dont know instead of guessing.
    """