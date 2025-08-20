import dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

import os
from langchain.chains import RetrievalQA

print(f"Loading the OPENAI API KEY.....")
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

print(f"Loading the scanned data.....")
# load the text file
loader = TextLoader("data/unstructured_contract_sample.txt")
documents = loader.load()

print(f"Splitting the text data into chunks........")
# split the documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = text_splitter.split_documents(documents)

print(F"Initializing embedding - OPENAI........")
# Embeddings
embeddings = OpenAIEmbeddings()

print(f"Creating FIASS vector store in memory.......")
# Vector Store
vectorstore = FAISS.from_documents(chunks, embeddings)


# Query
query = "What are the services agreed to provide?"
#result = qa.invoke(query)
#print(result)

# Retrieve top 3 most similar chunks
results = vectorstore.similarity_search(query, k=3)
for i, doc in enumerate(results):
    print(f"\n--- Chunk #{i+1} ---")
    print(doc.page_content)
    print(f"Metadata: {doc.metadata}") # Shows source file and chunk location


print("\nCreating a retriever for further RAG steps...")
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
retriever_results = retriever.invoke(query)
print(f"Retriever found {len(retriever_results)} chunks.")

#Query
qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(model="gpt-3.5-turbo"), chain_type="stuff", retriever=vectorstore.as_retriever())
result = qa.invoke(query)
print(result)