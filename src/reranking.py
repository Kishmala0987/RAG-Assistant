from sentence_transformers import CrossEncoder
from src import config

def get_reranker():
    "Return Reranker CrossEncoder model"
    return CrossEncoder(config.RERANKING_MODEL)

def rerank_documents(query: str, rrf_results, reranker):
    """Rerank RRF results using a CrossEncoder."""
    pairs = [(query, doc.page_content) for doc in rrf_results]
    scores = reranker.predict(pairs)
    ranked_documents = sorted(zip([doc for doc in rrf_results], scores),
    key=lambda x: x[1], reverse = True)
    return ranked_documents