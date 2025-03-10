from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains import create_retrieval_chain
from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import streamlit as st

def load_document(pdf):
  loader = PyPDFLoader(pdf)
  docs = loader.load()
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
  chunks = text_splitter.split_documents(docs)

  return chunks



st.title('AI Powered PDF Document Q&A')
uploaded_file = st.file_uploader('Upload a PDF file', type='pdf')

if uploaded_file:
  temp_file = './temp.pdf'
  with open(temp_file, 'wb') as f:
    f.write(uploaded_file.getvalue())
    file_name = uploaded_file.name

  chunks = load_document(temp_file)
  st.write("Processing document...Thank you for your patience")
  embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
  vector_db = FAISS.from_documents(chunks, embeddings)
  retriever = vector_db.as_retriever()
  llm = ChatOpenAI(model_name='gpt-4o-mini', openai_api_key='key')

  system_prompt = (
    "You are a helpful assistant. Use the given context to answer the question."
        "If you don't know the answer, say you don't know. "
        "{context}"
  ) 
  prompt = ChatPromptTemplate.from_messages(
    [
      ("system", system_prompt),
      ("human", "{input}")
    ]
  )
  question_answer_chain = create_stuff_documents_chain(llm, prompt)
  chain = create_retrieval_chain(retriever, question_answer_chain)

  question = st.text_input('Ask a question about the document')
  if question:
    response = chain.invoke({"input": question})['answer']
    st.write(response)
