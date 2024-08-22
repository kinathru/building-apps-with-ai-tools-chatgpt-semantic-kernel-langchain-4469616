import os

from dotenv import load_dotenv
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

# Load the .env file
load_dotenv()

user_api_key = os.getenv('OPENAI_API_KEY')

# load content from wikipedia Tea page
loader = WebBaseLoader("https://en.wikipedia.org/wiki/Tea")
documents = loader.load()

# Putting all data into ChatBot is not needed. So we split data into chunks.
# Chunks are pieces of data that's split up
text_splitter = CharacterTextSplitter(chunk_size=2500, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# Embeddings is a numerical representation of the text
embeddings = OpenAIEmbeddings()
docsearch = FAISS.from_documents(texts, embeddings)  # FAISS is going to create some embeddings from the texts we provided
# Learn about FAISS here -> https://python.langchain.com/v0.2/docs/integrations/vectorstores/faiss

# Here we are creating a Q&A retrieval chatbot, with OpenAI to power the chatbot which decides how to summarize the response from documents
qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.as_retriever())

while True:
    query = input("Ask a question about tea : \n")

    if query == 'exit':
        break

    # run Q&A
    print(qa.invoke(query))
