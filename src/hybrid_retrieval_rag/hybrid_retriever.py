from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document

from rrf import reciprocal_rank_fusion


class HybridRetriever(BaseRetriever):

    dense_retriever: object
    sparse_retriever: object
    final_top_k: int

    def _get_relevant_documents(
        self,
        query: str
    ) -> list[Document]:

        # Dense retrieval
        dense_documents = self.dense_retriever.invoke(query)


        print(f"Dense documents retrieved: "f"{len(dense_documents)}")

        for index, document in enumerate(dense_documents, start=1):
            print(f"\nDense Document {index}")
            print(f"Content: " f"{document.page_content[:200]}")


        # Sparse retrieval using BM25
        sparse_documents_bm25 = self.sparse_retriever.invoke(query)

        print(f"BM25 documents retrieved: "f"{len(sparse_documents_bm25)}")

        for index, (document,score) in enumerate(sparse_documents_bm25,start=1):
            print(f"\nBM25 Document {index}")
            print(f"Content: "f"{document.page_content[:200]}")

        # RRF
        hybrid_results = reciprocal_rank_fusion(
            dense_documents=dense_documents,
            sparse_documents=sparse_documents_bm25
        )

        return [
            document
            for document, score
            in hybrid_results[:self.final_top_k]
        ]