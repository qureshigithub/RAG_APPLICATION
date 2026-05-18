import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq


def load_documents(directory_path):
    loader = DirectoryLoader(
        directory_path,
        glob="**/*.pdf",        # all pdf files load karo
        loader_cls=PyPDFLoader  # PDF loader use it
    )
    documents = loader.load()
    print(f" Total pages load : {len(documents)}")
    return documents



def chunking_data(data):
    split_data = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = split_data.split_documents(data)
    return chunks
    

def data_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings
