"""Implements the rephrase question chain"""

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

class RephraseQuestion:
    
    """Implements the rephrase question chain"""
    
    _CONDESE_QUESTION_PROMPT_TEMPLATE = """
    Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:
    """
    _CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_CONDESE_QUESTION_PROMPT_TEMPLATE)
    
    def __init__(self, llm):
        self.rephrase_question_chain = self._CONDENSE_QUESTION_PROMPT | llm | StrOutputParser()
    
    def run(self, message, history):
        """Returns the rephrased question from the LLM to the user's message."""
        return self.rephrase_question_chain.invoke({"chat_history": history, "question": message})