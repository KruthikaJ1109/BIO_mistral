# BIO_mistral
# Sepsis Prediction and Q&A System

This repository contains a project that focuses on the prediction of sepsis and provides a question-and-answer system related to sepsis. The project utilizes a machine learning model for prediction and a Retrieval-Augmented Generation (RAG) system powered by the LlamaCpp model to answer queries based on a sepsis-related book.

## Project Overview

This project is designed to:
1. Load and process a sepsis-related book in PDF format.
2. Split the book into manageable text chunks for easier processing.
3. Generate embeddings for the text chunks using the `HuggingFaceEmbeddings` model.
4. Store the embeddings in a vector store (`Chroma`) to enable similarity searches.
5. Provide a question-answering system that leverages the `LlamaCpp` model for generating responses based on retrieved information.

## Requirements

Ensure that you have the following dependencies installed:

```bash
      pip install langchain sentence-transformers chromadb llama-cpp-python langchain_community pypdf
