from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain
)

from loader import load_documents
from splitter import create_document_chunks
from embedding_model import get_embedding_model
from vector_store import create_vector_store
from llm import create_llm
from prompt import create_prompt

from bm25_retriever import BM25Retriever
from hybrid_retriever import HybridRetriever

from config import (
    DIRECTORY_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL,
    DENSE_RETRIEVAL_TOP_K,
    SPARSE_RETRIEVAL_TOP_K
)



documents = load_documents(DIRECTORY_PATH)
print(f"Loaded documents: {len(documents)}")


chunks = create_document_chunks(
    documents,
    CHUNK_SIZE,
    CHUNK_OVERLAP
)

print(f"Created chunks: {len(chunks)}")



# 3. Create Dense Retriever
embeddings = get_embedding_model(EMBEDDING_MODEL)
vector_store = create_vector_store(
    chunks,
    embeddings
)
dense_retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": DENSE_RETRIEVAL_TOP_K
    }
)

print("Dense retriever created successfully")


# 4. Create Sparse Retriever - BM25
# For BM25, we need access to the same chunks.
# So BM25 needs the same chunks that went into your vector store
bm25_retriever = BM25Retriever(
    chunks,
    k=SPARSE_RETRIEVAL_TOP_K
)

print("BM25 retriever created successfully")


# ============================================================
# 5. Create Hybrid Retriever
# ============================================================

hybrid_retriever = HybridRetriever(
    dense_retriever=dense_retriever,
    sparse_retriever=bm25_retriever,
    final_top_k=3
)

print("Hybrid retriever created successfully")



# 6. Create LLM
llm = create_llm()

# 7. Create Prompt
prompt = create_prompt()

# 8. Create Document Chain
document_chain = create_stuff_documents_chain(
    llm,
    prompt
)

# 9. Create RAG Chain
rag_chain = create_retrieval_chain(
    hybrid_retriever,
    document_chain
)



print("\n" + "-" * 60)

question = input("  ❓ Ask a question: ")

print("-" * 60)


print("\n========== 11. RUNNING HYBRID RETRIEVAL ==========")

response = rag_chain.invoke({
    "input": question
})

print("\n========== ANSWER ==========\n")


print(response["answer"])

print("\n" + "-" * 60)
print("  ✓ RAG pipeline completed successfully")
print("-" * 60)