"""Script to ingest data to a ChromaDB vector store, and persist it to disk"""

import os
from dotenv import load_dotenv

from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# load the environment variables
load_dotenv()

# load the data
markdown_path = "data/source.md"
# read the markdown file and return the full document as a string
with open(markdown_path, "r") as file:
    full_markdown_document = file.read()

# split the data into chunks based on the markdown heading
headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
chunked_documents = markdown_splitter.split_text(full_markdown_document)

# create a vector store
embeddings_model = OpenAIEmbeddings()
db = Chroma.from_documents(chunked_documents, embeddings_model, persist_directory="data/chroma_db")
