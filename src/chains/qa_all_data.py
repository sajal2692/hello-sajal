"""Implements a QA chain to run using the full data."""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class QAAllData:
    """Implements a QA chain to run using the full data"""

    _PROMPT_TEMPLATE = """
    You are an AI  assistant, Saj, built by Sajal Sharma, an AI Engineer.
    Your main task is to answer questions people may have about Sajal. 
    Use the following information about Sajal context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
    Question: {question} 
    Context: {context} 
    Answer:
    """

    _PROMPT = ChatPromptTemplate.from_template(_PROMPT_TEMPLATE)

    def __init__(self, llm, source_data_path):
        with open(source_data_path, "r") as file:
            self.full_markdown_document = file.read()
        self.qa_all_data_chain = self._PROMPT | llm | StrOutputParser()

    def run(self, question):
        """Returns the response from the LLM to the user's message using all data."""
        return self.qa_all_data_chain.invoke({"question": question, "context": self.full_markdown_document})