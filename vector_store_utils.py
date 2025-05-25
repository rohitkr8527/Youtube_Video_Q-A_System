from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

def create_vector_store(chunks):
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Create FAISS vector store in-memory (FAISS doesn't require SQLite)
    vectordb = FAISS.from_texts(texts=chunks, embedding=embedding_model)
    return vectordb
