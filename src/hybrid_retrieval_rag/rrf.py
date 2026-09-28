
def reciprocal_rank_fusion(
    dense_documents,
    sparse_documents,
    k=60
):
    rrf_scores = {}
    rrf_documents = {}

    # Dense results
    for rank, document in enumerate(
        dense_documents,
        start=1
    ):
        key = document.page_content

        rrf_documents[key] = document

        score = 1 / (k + rank)

        rrf_scores[key] = (
            rrf_scores.get(key, 0) + score
        )

    # SparseBM25 results
    for rank, (document, bm25_score) in enumerate(
        sparse_documents,
        start=1
    ):
        key = document.page_content

        rrf_documents[key] = document

        score = 1 / (k + rank)

        rrf_scores[key] = (
            rrf_scores.get(key, 0) + score
        )

    ranked_keys = sorted(
        rrf_scores,
        key=rrf_scores.get,
        reverse=True
    )

    hybrid_results = [
        (rrf_documents[key], rrf_scores[key])
        for key in ranked_keys
    ]

    return hybrid_results