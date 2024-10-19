# https://medium.com/@mehar.chand.cloud/how-to-build-a-simple-retrieval-augmented-generation-rag-system-f6ffaf8a705c

from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load documents
path_doc = "/home/wilderd/Documents/programming/nlp/llm_rag/"
file_paths = [path_doc + "doc1.txt",path_doc + "doc2.txt", path_doc + "doc3.txt"]

all_docs = []
# Load each file and append the documents to the list
for file_path in file_paths:
    loader = TextLoader(file_path)
    documents = loader.load()
    all_docs.extend(documents)

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)



# Use sentence-transformers for generating embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2") #"sentence-transformers/all-MiniLM-L6-v2")
embeddings = embedding_model.embed_documents([chunk.page_content for chunk in chunks])


from langchain.vectorstores import FAISS

# Create a FAISS vector store from document embeddings
vectorstore = FAISS.from_documents(documents=chunks, embedding=embedding_model)

from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# Initialize the retriever and language model
retriever = vectorstore.as_retriever()
llm = OpenAI()

# Create a RetrievalQA chain
rag_pipeline = RetrievalQA(llm=llm, retriever=retriever)

# Ask a question
response = rag_pipeline.run("What is Retrieval-Augmented Generation?")
print(response)

question = "What is the difference between retrieval and generation?"
response = rag_pipeline.run(question)
print(f"Q: {question}\nA: {response}")