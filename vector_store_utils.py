from langchain_community.vectorstores import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

def create_vector_store(chunks):
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Create Chroma vector store in-memory (no persist_directory)
    vectordb = Chroma.from_texts(texts=chunks, embedding=embedding_model)
    return vectordb
