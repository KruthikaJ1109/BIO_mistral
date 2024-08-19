pip install langchain_community
pip install langchain










from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS, Chroma
from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA, LLMChain

from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

nltk.download("punkt")
nltk.download("stopwords")

import pathlib
import textwrap
from IPython.display import display, Markdown

def to_markdown(text):
    text = text.replace('•', '  *')
    return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

import os
from getpass import getpass

HUGGINGFACEHUB_API_TOKEN = getpass("Enter your HuggingFace API Token: ")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN

from langchain.document_loaders import PyPDFDirectoryLoader

loader = PyPDFDirectoryLoader("/content/sepsis_book.pdf")
docs = loader.load()

print(docs)

from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load the document
loader = PyPDFDirectoryLoader("/content/sepsis_book.pdf")
docs = loader.load()

# Check if documents are loaded
if len(docs) == 0:
    print("No documents loaded. Please check the file path and ensure the PDF is accessible.")
else:
    print(f"Number of documents loaded: {len(docs)}")

# Proceed with text splitting
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = text_splitter.split_documents(docs)

print(f"Number of chunks created: {len(chunks)}")
if len(chunks) > 0:
    for i in range(min(1000000, len(chunks))):  # Display up to the first 5 chunks
        print(chunks[i])
else:
    print("No chunks created. Please check the text splitting parameters.")

# Import necessary libraries
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load the document
loader = PyPDFLoader("/content/sepsis_book.pdf")
docs = loader.load()

# Verify the document content
print(docs)

# Split the document into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = text_splitter.split_documents(docs)

# Display the number of chunks and the first few chunks
print(len(chunks))
for i in range(min(5, len(chunks))):  # Ensure not to exceed the number of chunks
    print(chunks[i])

text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = text_splitter.split_documents(docs)

print(len(chunks))
for i in range(5):  # Display the first 5 chunks
    print(chunks[i])

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

vectorstore = Chroma.from_documents(chunks, embeddings)

query = "who is at risk of sepsis?"
search = vectorstore.similarity_search(query)

to_markdown(search[0].page_content)

retriever = vectorstore.as_retriever(
    search_kwargs={'k': 5}
)

retriever.get_relevant_documents(query)

from google.colab import drive

drive.mount('/content/drive')

llm = LlamaCpp(
    model_path= "/content/drive/MyDrive/ggml-model-Q4_K_M.gguf",
    temperature=0.3,
    max_tokens=2048,
    top_p=1
)

from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import ChatPromptTemplate

template = """
You are an AI assistant that follows instructions extremely well.
Please be truthful and give direct answers
</s>

{query}
</s>
"""

prompt = ChatPromptTemplate.from_template(template)

rag_chain = (
    {"context": retriever,  "query": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

response = rag_chain.invoke("what are the symptoms of sepsis?")

to_markdown(response)

import sys

while True:
    user_input = input("Input Prompt: ")
    if user_input.lower() == 'exit':
        print('Exiting')
        sys.exit()
    if user_input == '':
        continue
    result = rag_chain.invoke(user_input)
    print("Answer: ", result)

import sys

while True:
    user_input = input("Input Prompt: ")
    if user_input.lower() == 'exit':
        print('Exiting')
        sys.exit()
    if user_input == '':
        continue
    result = rag_chain.invoke(user_input)
    print("Answer: ", result)