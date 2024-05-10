"""Responds to any smalltalk or off-topic messages."""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from textwrap import dedent

class Smalltalk:
    
    """Responds to any smalltalk or off-topic messages."""
    
    _SMALLTALK_PROMPT = """
    You are an AI assistant, Saj, built by Sajal Sharma, an AI Engineer. Given the following message and chat history, please respond to the user.
    You are allow to repond to smalltalk messages such as greetings or how are yous. For any message that is off topic, or is not a greeting, or not about Sajal, refuse to answer and ask the user to ask a question about Sajal.

    Chat History: {chat_history}

    User Message: {input}
    """
    
    _SMALLTALK_PROMPT_TEMPLATE = ChatPromptTemplate.from_template(dedent(_SMALLTALK_PROMPT))
    
    def __init__(self, llm):
        self.smalltalk_chain = self._SMALLTALK_PROMPT_TEMPLATE | llm | StrOutputParser()   
    
    def run(self, message, history):
        """Returns the response from the LLM to the user's message."""
        return self.smalltalk_chain.invoke({"input": message, "chat_history": history})
        