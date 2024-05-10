"""Implements the RAG chain"""

from langchain_core.prompts import format_document, PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from textwrap import dedent

class RAG:
    
    """Implements the RAG chain"""
    
    _RAG_PROMPT = """
    You are an AI  assistant, Saj, built by Sajal Sharma, an AI Engineer.
    Your main task is to answer questions people may have about Sajal. 
    Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
    Question: {question} 
    Context: {context} 
    Answer:
    """
    
    _DOCUMENT_SEPARATOR = "\n\n"
    _DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")
    
    _RAG_PROMPT_TEMPLATE = ChatPromptTemplate.from_template(dedent(_RAG_PROMPT))
    
    def __init__(self, llm):
        self.rag_chain = self._RAG_PROMPT_TEMPLATE | llm | StrOutputParser()
    
    def _combine_documents(self, docs):
        doc_strings = [format_document(doc, self._DEFAULT_DOCUMENT_PROMPT) for doc in docs]
        return self._DOCUMENT_SEPARATOR.join(doc_strings)
    
    def run(self, question, documents):
        """Returns the response from the LLM to the user's message using RAG with chunked documents."""
        document_str = self._combine_documents(documents)
        return self.rag_chain.invoke({"question": question, "context": document_str})