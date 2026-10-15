from src import config

def reciprocal_rank_fusion(results, k = config.RRF_K):
    """It will reward documents based on their rankin on each retrievel result list, and Combine multiple ranked retrieval results using
    Reciprocal Rank Fusion (RRF)"""
    scores = {}
    documents = {}
    for result in results:
        for i, doc in enumerate(result, start = 1):
            doc_id = doc.page_content
            scores[doc_id] = scores.get(doc_id,0) + 1/(k+i)
            # Keep the original Document object
            documents[doc_id] = doc

    rrf_results = sorted(
            scores.items(),
            key=lambda x: x[1], #Sort using the second item, which is the score in reverse
            reverse=True
        )

    return [documents[doc_id]  for doc_id, sc in rrf_results]
