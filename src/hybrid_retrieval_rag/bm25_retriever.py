from rank_bm25 import BM25Okapi

class BM25Retriever:
    def __init__(self, documents, k):
        self.documents = documents
        self.k = k

        tokenized_documents = [
            document.page_content.lower().split()
            for document in documents
        ]

        self.bm25 = BM25Okapi(tokenized_documents)

    def invoke(self, query):

        tokenized_query = query.lower().split()

        scores = self.bm25.get_scores(tokenized_query)

        scored_documents = list(
            zip(self.documents, scores)
        )

        scored_documents.sort(
            key=lambda x: x[1],
            reverse=True
        )

        return scored_documents[:self.k]
        