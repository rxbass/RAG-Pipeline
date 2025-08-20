# RAG-Pipeline 

UseCase : Provide a messy, unstructured text document (e.g., a scanned contract). Using LangChain to chunk it, create embeddings, and store them in a FAISS in-memory index, then perform a simple similarity search and also a chatbot answering questions.


The Contract Query System is an AI-powered document analysis application built with Streamlit that enables users to interact with unstructured contract documents through natural language queries. The system leverages advanced language models and vector search technology to provide intelligent answers about contract content.

 - GPT-3.5-turbo = Answer generation
 - text-embedding-ada-002 (default) = Semantic search
 - FAISS = Vector storage & retrieval
