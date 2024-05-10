import os 

import gradio as gr

from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from graph import AssistantGraph


# load the environment variables
load_dotenv()

VECTOR_DB_PATH = "data/chroma_db"
SOURCE_DATA_PATH = "data/source.md"

# define llm
llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL"), temperature=0)

# create instance of assistant graph
app = AssistantGraph(llm=llm, vector_db_path=VECTOR_DB_PATH, source_data_path=SOURCE_DATA_PATH)

def process_history(history):
    """Return the history as a list of HumanMessage and AIMessage tuples"""
    chat_history = []
    for pair in history:
        human_message, ai_message = pair
        chat_history.append(HumanMessage(content=human_message))
        chat_history.append(AIMessage(content=ai_message))
    return chat_history

def run(message, history):
    chat_history = process_history(history[1:]) # ignore the auto message
    inputs = {"keys": {"message": message, "history": chat_history}}
    result = app.run(inputs)
    response = result["keys"]["response"]
    return response

initial_message = "Hi there! I'm Saj, an AI assistant built by Sajal Sharma. I'm here to answer any questions you may have about Sajal. Ask me anything!"

if __name__ == "__main__":
    gr.ChatInterface(run, chatbot=gr.Chatbot(value=[[None, initial_message]])).launch(server_name="0.0.0.0", server_port=7860, share=False)