#If not using a tool

from src.retrieval import reciprocal_rank_fusion
from src.reranking import rerank_documents

def search_hr_policy(question:str,  semantic_retriever, keyword_retriever, reranker) -> str:
        """    Search the HR policy database for information related to
        company policies, leave, benefits, working hours, remote work,
        and other HR-related questions.
        """
        semantic_chunks = semantic_retriever.invoke(question)
        keyword_chunks = keyword_retriever.invoke(question)

        #Hybrid Retrieval using RRF
        rrf_results = reciprocal_rank_fusion([semantic_chunks, keyword_chunks])

        #Reranking using CrossEncoder
        ranked_documents = rerank_documents(question, rrf_results, reranker)
        
        return "\n\n".join([doc.page_content for doc,sc in ranked_documents[:3]])