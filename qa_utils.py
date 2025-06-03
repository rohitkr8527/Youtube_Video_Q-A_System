from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

def build_qa_chain(vectordb):
    llm = ChatGroq(
        api_key="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",
        model="llama3-8b-8192",
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
