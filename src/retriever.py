"""Impleemnts the Retriever class for retrieving data from the database"""

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

class Retriever:
    """Retrieves data from the database"""
    
    def __init__(self, vector_db_path):
        _embedding_model = OpenAIEmbeddings()
        _db = Chroma(persist_directory=vector_db_path, embedding_function=_embedding_model)
        self.retriever = _db.as_retriever()

    def run(self, query):
        """Retrieves data from the database"""
        return self.retriever.get_relevant_documents(query)