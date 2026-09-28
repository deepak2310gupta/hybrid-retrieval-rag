from langchain_text_splitters import RecursiveCharacterTextSplitter

def create_document_chunks(documents, chunk_size, chunk_overlap):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # separators=["\n\n", "\n", " ", ""]
    )

    document_chunks = splitter.split_documents(documents)
    return document_chunks