# A Simple Retrieval-Augmented Generation (RAG) System

This project sample is a simple demo about using RAG and Large Language Models (LLM) like gpt

## Features

- Retrieval: the system retrieves relevant documents or snippets from a knowledge based on the input query.
- Generation: previously retrieved information is used as context to generate a response using language models like gpt.

last but not least, RAG makes effective the generation for task like question asnwering, summarization, or chatbots.

## Installation

Follow these steps to install the project:
we folllo the installation by modules because an issue in the lanchain-python version: here is the [issue](https://stackoverflow.com/questions/76726419/langchain-modulenotfounderror-no-module-named-langchain) solved.

1. Clone the repository:

   ```bash
   # ssh
   git clone git@github.com:wild10/llm_rag.git

   # install dependencies:
   python3.10 -m pip install langchain
   python3.10 -m pip install -U langchain-community
   pip install faiss-cpu transformers
   ```
