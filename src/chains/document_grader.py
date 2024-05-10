"""Implements the document grader chain"""

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain.output_parsers.openai_tools import PydanticToolsParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from textwrap import dedent

import os
from dotenv import load_dotenv

load_dotenv()

# Data model
class grade(BaseModel):
    """Binary score for relevance check."""
    binary_score: str = Field(description="Relevance score 'yes' or 'no'")

class DocumentGrader:
    
    """Implements the document grader chain"""
    
    _GRADER_PROMPT_TEMPLATE = """
    You are a grader assessing relevance of a retrieved document to a user question. \n 
    Retrieved document: \n\n {context} \n\n
    User Question: {question} \n
    When assessing the relevance of a retrieved document to a user question, consider whether the document can provide a complete answer to the question posed. A document is considered relevant only if it contains all the necessary information to fully answer the user's inquiry without requiring additional context or assumptions.
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
    Do not return anything other than a 'yes' or 'no'.
    """
    
    _GRADER_PROMPT = PromptTemplate(template=dedent(_GRADER_PROMPT_TEMPLATE), input_variables=["context", "question"])
    
    def __init__(self):
        # seperate the model wrapper instance for the binded tool
        llm = ChatOpenAI(temperature=0, model=os.environ["OPENAI_MODEL"])
        grade_tool_oai = convert_to_openai_tool(grade)
        # LLM with tool and enforce invocation
        llm_with_tool = llm.bind(
            tools=[grade_tool_oai],
            tool_choice={"type": "function", "function": {"name": "grade"}},
        )
        # Parser
        parser_tool = PydanticToolsParser(tools=[grade])
        self._grader_chain = self._GRADER_PROMPT | llm_with_tool | parser_tool
        
    def run(self, question, context):
        """Returns the response from the document grader"""
        return self._grader_chain.invoke({"context": context, "question": question})
