"""Implements the intents detection chain"""

from langchain.chains import create_tagging_chain
from langchain_core.prompts import ChatPromptTemplate
from textwrap import dedent

class IntentDetection:
    
    """Implements the intents detection chain"""

    _SCHEMA = {
        "properties": {
            "intent": {
                "type": "string",
                "enum": ["smalltalk", "sajal_question"],
                "description": "The intent of the user's message. The intent is sajal_question, if the user is asking questions about Sajal Sharma"
                    "for example related to his contact info, work experience, educational background, certifications, hobbies, etc, or any other questions about him." 
                    "Any questions about contact info, work experience, educational background, certifications, hobbies, should also be sajal_question."
                    "General greetings or smalltalk messages are smalltalk. Questions about anyone other than sajal are also smalltalk."
            }
        }
    }

    _TAGGING_PROMPT = """Extract the desired information from the following passage.

    Only extract the properties mentioned in the 'information_extraction' function.
    Use the chat history to guide your extraction.

    Chat History:
    {history}

    Passage:
    {input}
    """
    _TAGGING_PROMPT_TEMPLATE = ChatPromptTemplate.from_template(dedent(_TAGGING_PROMPT))
    
    def __init__(self, llm):
        self.tagging_chain = create_tagging_chain(self._SCHEMA, llm, prompt=self._TAGGING_PROMPT_TEMPLATE)
    
    def run(self, message, history):
        """Returns the detected intent"""
        result = self.tagging_chain.invoke({"input": message, "history": history})
        return result["text"]["intent"]
        
