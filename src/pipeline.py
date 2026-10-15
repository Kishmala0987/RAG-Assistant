from src import config
from src.document_loader import load_documents
from src.splittng import split_into_chunks
from src.llm import get_llm
from src.tools import create_search_tool
from src.agent import create_hr_agents
# from src.vector_store import (
#     check_vector_store_exists,
#     save_vector_store,
#     build_vector_store,
#     get_semantic_retriever,
#     load_vector_store
# )
from src.vector_store_qdrant import (
    check_vector_store_exists,
    build_vector_store,
    get_semantic_retriever,
    load_vector_store
)
from src.chunks import (
    save_chunks,
    load_chunks,
    get_all_chunks_from_qdrant,
    get_bm25_retriever
)
from src.reranking import get_reranker

from src.logger import get_logger
from src.tracing import check_langsmith_tracing
from src.guardrails import check_input, check_output, REFUSAL_MESSAGE
logger = get_logger(__name__)
#Data Ingestion
def build_vector_store_for_documents(file_path: str = config.DATA_FILE_PATH):
    """Load + split + embed the document, reusing the Qdrant collection if have one"""
    if check_vector_store_exists():
        logger.info("Found an existing Qdrant Collection, connecting to one")
        vector_store = load_vector_store()
        chunks = get_all_chunks_from_qdrant()
        return vector_store, chunks

    logger.info("No Qdrant collection founded, building from scratch")
    documents = load_documents(file_path)
    chunks = split_into_chunks(documents)
    vector_store = build_vector_store(chunks)
    save_chunks(chunks)
    logger.info("Vector store Builded and uploaded to Qdrant safelty")
    return vector_store, chunks

#Data Retrievel
def build_hr_agent():
    """Build the full Rag agent Ready to answer Questions"""
    logger.info("HR Agent is building")
    config.check_api_keys()
    check_langsmith_tracing()
    vector_store, chunks = build_vector_store_for_documents()
    semantic_retriever = get_semantic_retriever(vector_store) 
    keyword_retriever = get_bm25_retriever(chunks)

    reranker = get_reranker()
    search_tool = create_search_tool(semantic_retriever, keyword_retriever, reranker)

    llm = get_llm()
    agent = create_hr_agents(llm, [search_tool])
    logger.info("HR Agent has been built")
    return agent

def ask(agent,question: str):
    """Ask the question to agent and return its answer as a plain text."""

    #Input guard - to get safe inputs
    logger.info(f"User  Question:{question}")


    is_safe, _ = check_input(question)
    if not is_safe:
        return REFUSAL_MESSAGE
    response = agent.invoke({"messages": [{"role": "user", "content": question}]})

    #Output Guard - to check if agent gives a safe answer
    answer = response["messages"][-1].content
    is_safe, _ = check_output(answer)
    if not is_safe:
        return REFUSAL_MESSAGE
    logger.info(f"Final Answer:{answer}")
    return answer

    

