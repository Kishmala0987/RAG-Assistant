#Step 6
#from langchain_groq import ChatGroq
#from src import config
from src.logger import get_logger
from src.gateway  import get_gateway_llm
logger = get_logger(__name__)

# def get_llm():
#     """Return Groq Chat Model"""
#     logger.info("Initializing an LLM:", config.LLM_MODEL_NAME)
#     return ChatGroq(
#         model_name = config.LLM_MODEL_NAME,
#         temperature=0.3
#     )
def get_llm():
    """Return Groq Chat Model"""
    logger.info("Initializing an LLM via Portkey")
    return get_gateway_llm()