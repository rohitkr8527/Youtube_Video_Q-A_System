from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
import os
import streamlit as st

def build_qa_chain(vectordb):

    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

    llm = ChatGroq(
        api_key=os.environ.get("GROQ_API_KEY"),
        model="llama-3.3-70b-versatile",
        temperature=0.5
    )

    retriever = vectordb.as_retriever(search_kwargs={"k": 4})

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    return qa_chain
