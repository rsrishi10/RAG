import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import time

from dotenv import load_dotenv
load_dotenv()

## load the Groq API key
groq_api_key=os.environ['GROQ_API_KEY']
os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")

st.title("Chatgroq With Llam3")

llm=ChatGroq(groq_api_key=groq_api_key,
             model_name="Llama3-8b-8192")

prompt= ChatPromptTemplate.from_template(
    
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}

"""
)

if "vectors" not in st.session_state:
    def vector_embedding():
        st.session_state.embeddings = OpenAIEmbeddings()
        st.session_state.loader = PyPDFDirectoryLoader("/Users/rathore/Desktop/Github/RAG/Chatbot_qroq/us_census")  # Data Ingestion
        st.session_state.docs = st.session_state.loader.load()  # Document Loading
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  # Chunk Creation
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20])  # Splitting
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)  # Vector store

    if st.button("Document Embedding"):
        vector_embedding()
        st.write("Vector store db is Ready")

# Check if vectors are available
if "vectors" in st.session_state:
    prompt1 = st.text_input("Enter Your Question From Documents")
    
    if prompt1:
        llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        start = time.process_time()
        response = retrieval_chain.invoke({"input": prompt1})
        st.write("Response time:", time.process_time() - start)
        st.write(response['answer'])

        with st.expander("Document Similarity Search"):
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.write("--------------------------------")
else:
    st.write("Please initialize the vector embedding by clicking the 'Document Embedding' button.")
