#Step 5
from langchain.tools import tool
from src.retrieval import reciprocal_rank_fusion
from src.reranking import rerank_documents
from src.logger import get_logger

logger = get_logger(__name__)

def create_search_tool(semantic_retriever, keyword_retriever, reranker):
    """Return a @tool function that searches an HR policy document.."""

    @tool
    def search_hr_policy(question:str) -> str:
        """    Search the HR policy database for information related to
        company policies, leave, benefits, working hours, remote work,
        and other HR-related questions.
        """
        logger.info("Search HR Policy called with question:", question)
        semantic_chunks = semantic_retriever.invoke(question)
        keyword_chunks = keyword_retriever.invoke(question)

        #Hybrid Retrieval using RRF
        rrf_results = reciprocal_rank_fusion([semantic_chunks, keyword_chunks])

        #Reranking using CrossEncoder
        ranked_documents = rerank_documents(question, rrf_results, reranker)
        
        return "\n\n".join([doc.page_content for doc,sc in ranked_documents[:3]])

    return search_hr_policy
