"""Implements the graph to handle workflows for the Sajal assistant"""

from typing import Dict, TypedDict

from chains.intent_detection import IntentDetection
from chains.smalltalk import Smalltalk
from chains.document_grader import DocumentGrader
from chains.rephrase_question import RephraseQuestion
from chains.qa_all_data import QAAllData
from chains.rag import RAG

from retriever import Retriever

from langgraph.graph import END, StateGraph

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        keys: A dictionary where each key is a string.
    """
    keys: Dict[str, any]

class AssistantGraph:
    
    """Implements the graph to handle workflows for the Sajal assistant"""
    
    def __init__(self, llm, vector_db_path, source_data_path):
        self.intent_detector = IntentDetection(llm)
        self.smalltalk = Smalltalk(llm)
        self.document_grader = DocumentGrader()
        self.rephrase_question_chain = RephraseQuestion(llm)
        self.retriever = Retriever(vector_db_path=vector_db_path)
        self.qa_all_data = QAAllData(llm=llm, source_data_path=source_data_path)
        self.rag = RAG(llm)
        self.app = self.compile_graph()
        
    def run(self, inputs):
        return self.app.invoke(inputs)
    
    # define graph nodes and edges and compile graph
    def compile_graph(self):
        workflow = StateGraph(GraphState)
        ### define the nodes
        workflow.add_node("detect_intent", self.detect_intent)
        workflow.add_node("chat", self.chat)
        workflow.add_node("rephrase_question", self.rephrase_question)
        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("grade_documents", self.grade_documents)
        workflow.add_node("generate_answer_with_retrieved_documents", self.generate_answer_with_retrieved_documents)
        workflow.add_node("generate_answer_using_all_data", self.generate_answer_using_all_data)
        ### build the graph
        workflow.set_entry_point("detect_intent")
        workflow.add_conditional_edges(
            "detect_intent",
            self.decide_to_rag,
            {
                "rag": "rephrase_question",
                "chat": "chat",
            }
        )
        workflow.add_edge("rephrase_question", "retrieve")
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_conditional_edges(
            "grade_documents",
            self.decide_to_use_all_data,
            {
                "rag": "generate_answer_with_retrieved_documents",
                "generate_answer_using_all_data": "generate_answer_using_all_data",
            }
        )
        workflow.add_edge("generate_answer_with_retrieved_documents", END)
        workflow.add_edge("generate_answer_using_all_data", END)
        workflow.add_edge("chat", END)
        ### compile the graph
        app = workflow.compile()
        return app
    
    # define the nodes
    def detect_intent(self, state):
        """
        Detects the intent of a user's message

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, intent, that contains the detected intent
        """
        state = state["keys"]
        message = state["message"]
        history = state["history"]
        intent = self.intent_detector.run(message=message, history=history)
        return {"keys": {"message": message, "intent": intent, "history": history}}
    
    def chat(self, state):
        """
        Chat with the user

        Args:
            state (dict): The current graph state

        Returns:
            str: Updated graph state after adding response
        """
        state = state["keys"]
        input = state["message"]
        history = state["history"]
        response = self.smalltalk.run(message=input, history=history)
        return {"keys": {"message": input, "history": history, "response": response}}
    
    def grade_documents(self, state):
        """
        Determines whether the retrieved documents are relevant to the question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates documents key with relevant documents
        """

        print("---CHECK RELEVANCE---")
        state = state["keys"]
        question = state["standalone_question"]
        documents = state["documents"]

        # Score
        filtered_docs = []
        all_data = False  # Default do not opt to use all data for generation
        for d in documents:
            score = self.document_grader.run(question=question, context=d.page_content)
            grade = score[0].binary_score
            if grade == "yes":
                print("---GRADE: FOUND RELEVANT DOCUMENT---")
                filtered_docs.append(d)

        if not filtered_docs:
            all_data = True  # Opt to use all data for generation

        return {
            "keys": {
                "documents": filtered_docs,
                "standalone_question": question,
                "run_with_all_data": all_data,
            }
        }
        
    def rephrase_question(self, state):
        """
        Rephrase the question to be a standalone question
        
        Args:
            state (dict): The current graph state
        
        Returns:
            str: Updated graph state after adding standalone question
        """
        state = state["keys"]
        question = state["message"]
        chat_history = state["history"]
        result = self.rephrase_question_chain.run(message=question, history=chat_history)
        return {"keys": {"message": question, "history": chat_history, "standalone_question": result}}
    
    def retrieve(self, state):
        """
        Retrieve documents

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, documents, that contains retrieved documents
        """
        state = state["keys"]
        question = state["standalone_question"]
        chat_history = state["history"]
        documents = self.retriever.run(query=question)
        return {"keys": {"message": state["message"], "history": chat_history, "standalone_question": question, "documents": documents}}

    def generate_answer_using_all_data(self, state):
        """
        Generate an answer using all documents

        Args:
            state (dict): The current graph state

        Returns:
            str: Updated graph state after adding response
        """
        state = state["keys"]
        question = state["standalone_question"]
        response = self.qa_all_data.run(question=question)
        return {"keys": {"message": question, "response": response}}
    
    def generate_answer_with_retrieved_documents(self, state):
        """
        Generate an answer using the retrieved documents

        Args:
            state (dict): The current graph state

        Returns:
            str: Updated graph state after adding response
        """
        state = state["keys"]
        question = state["standalone_question"]
        documents = state["documents"]
        response = self.rag.run(question=question, documents=documents)
        return {"keys": {"message": question, "response": response}}
    
    # define the edges
    def decide_to_rag(self, state):
        """
        Decides whether to use RAG or not

        Args:
            state (dict): The current graph state

        Returns:
            str: Next node to call
        """
        state = state["keys"]
        intent = state["intent"]
        if intent == "sajal_question":
            return "rag"
        return "chat"

    def decide_to_use_all_data(self, state):
        """
        Determines whether to use all data for generation or not.

        Args:
            state (dict): The current state of the agent, including all keys.

        Returns:
            str: Next node to call
        """

        state = state["keys"]
        run_with_all_data = state["run_with_all_data"]

        if run_with_all_data:
            return "generate_answer_using_all_data"
        else:
            return "rag"
