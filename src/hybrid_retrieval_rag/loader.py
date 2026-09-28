from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

def load_documents(directory_path: str):
    loader = DirectoryLoader(
        directory_path,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True
    )

    documents = loader.load()
    return documents
