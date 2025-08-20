import streamlit as st
import dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

import os
from langchain.chains import RetrievalQA


#setup the page
st.set_page_config(
    page_title= "Unstructured Contract Search",
    page_icon = "📄",
    layout="wide"
)

# Initialize the system
@st.cache_resource

def initialize_system():
    """Initialize the document processing system"""
    with st.spinner("Initializing system..."):
        # Load Enivornment Variables
        load_dotenv()
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

        # Load the document
        loader = TextLoader("data/unstructured_contract_sample.txt")
        documents = loader.load()

        # split the documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = text_splitter.split_documents(documents)

        # Embeddings
        embeddings = OpenAIEmbeddings()

        # Vector Store
        vectorstore = FAISS.from_documents(chunks, embeddings)

        # Create Retriever
        retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

        #Initializing the retreival QA
        qa = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(model="gpt-3.5-turbo"), 
            chain_type="stuff", 
            retriever=vectorstore.as_retriever(),
            return_source_documents=False
        )

        return qa, vectorstore

def ask_question(question, qa_system):
    """Process a question and return the answer"""
    if not question.strip():
        return "Please enter a question.", []
    
    try:
        result = qa_system.invoke({"query": question})
        answer = result['result']
        return answer
    except Exception as e:
        return f"An error occurred: {str(e)}", []

def main():
    # Initialize the system
    qa_system, vectorstore = initialize_system()
    
    # UI Layout
    st.title("📄 Contract Query System")
    st.markdown("Ask questions about the contract document and get AI-powered answers.")
    
    # Sidebar
    with st.sidebar:
        st.header("About")
        st.info("This system uses LangChain and FAISS to process unstructed contract documents and answer questions about their content, also perform a simple similarity search.")
        
        st.header("Example Questions")
        st.markdown("""
        - What services are agreed to be provided?
        - What is the total contract value?
        - What services is the contractor required to provide?
        - What is the payment schedule for this contract?
        - How long does this agreement last?
        - What are the deliverables expected from the contractor?
        - What are the confidentiality obligations of both parties?
        - Who owns the intellectual property created under this agreement?
        - How can either party terminate this contract?
        - What is the limitation of liability for the contractor?
        - Which state's laws govern this agreement?
        """)
    
    # Main content
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Query input
        question = st.text_input(
            "Enter your question about the contract:",
            placeholder="What services are agreed to be provided?",
            key="question_input"
        )
        
        if st.button("Ask Question", type="primary"):
            if question:
                with st.spinner("Searching for answers..."):
                    answer = ask_question(question, qa_system)
                    
                    # Display answer
                    st.subheader("Answer:")
                    st.success(answer)
            else:
                st.warning("Please enter a question.")
    
    with col2:
        # Document information
        st.subheader("Unstructured Document Information")
        st.info("""
        SERVICE AGR33MENT CONTR4CT    PARTIES: CLIENT Acme lndustries LLC 123 Business Park Drive Cityville ST 12345     CONTRACTOR TechSolutions lnc. 456 lnnovation Blvd Suite 200 Techtown ST 67890 EFFECT1VE DATE january 15 2024 TERM This agreement shall c0mmence on the Effect1ve Date and c0ntinue f0r a peri0d 0f twelve 12 m0nths unless terminated earlier in acc0rdance with the pr0visi0ns herein SC0PE 0F W0RK The C0ntract0r agrees t0 pr0vide the f0ll0wing services S0ftware devel0pment and maintenance Technical c0nsulting services System integratin supp0rt 247 technical supp0rt during business h0urs C0MPENSATI0N T0tal c0ntract value $85000.00 Eighty five th0usand d0llars Payment schedule 30% up0n signing $25500.00 40% up0n milest0ne c0mpleti0n $34000.00 30% up0n pr0ject c0mpleti0n $25500.00 Payment terms Net 30 days fr0m inv0ice date DELIVERABLES 1 Cust0m s0ftware applicati0n 2 User d0cumentati0n and training materials 3 Technical specificati0ns d0cument 4 S0urce c0de and depl0yment packages C0NFIDENTIALITY B0th parties ackn0wledge that c0nfidential inf0rmati0n may be discl0sed during the c0urse 0f this agreement Each party agrees t0 maintain strict c0nfidentiality 0f all pr0prietary inf0rmati0n INTELLECTUAL PR0PERTY All w0rk pr0duct created under this agreement shall be the exclusive pr0perty 0f the Client except f0r C0ntract0rs pre existing intellectual pr0perty and general meth0d0l0gies TERMINATI0N Either party may terminate this agreement with thirty 30 days written n0tice ln case 0f material breach the n0n breaching party may terminate immediately up0n written n0tice LIABILITY LIMITATI0N C0ntract0rs t0tal liability shall n0t exceed the t0tal am0unt paid under this agreement C0NTRACT0R DISCLAIMS ALL WARRANTIES EXCEPT AS EXPRESSLY SET F0RTH HEREIN G0VERNING LAW This agreement shall be g0verned by the laws 0f the State 0f Calif0rnia with0ut regard t0c0nflict 0f law principles SIGNATURES CLIENT                           C0NTRACT0R j0hn Smith                       Sarah j0hns0n Title CE0                       Title President Date 1152024                  Date 1152024 Acme lndustries LLC              TechS0luti0ns lnc WITNESS Mark Wils0n Date 1152024 ADDENDUM A TECHNICAL SPECIFICATI0NS Hardware Requirements Minimum 8GB RAM 500GB st0rage space Wind0ws 10 0r later mac0S 10.15+ Netw0rk c0nnectivity required S0ftware Dependencies Database P0stgreSQL 12+ Framew0rk React 18x Runtime N0de.js 16+ Security Requirements SSLTLS encrypti0n f0r all data transmissi0n User authenticati0n via 0Auth 20 Regular security audits quarterly Data backup pr0cedures daily Perf0rmance Metrics Resp0nse time 2 sec0nds f0r 95% 0f requests Uptime 995% availability C0ncurrent users Supp0rt up t0 500 simultan0us users Data pr0cessing Handle 10000 transacti0ns per h0ur
        """)
        
        # Direct similarity search option
        st.subheader("Direct Document Search - top 3 most similar results")
        search_term = st.text_input("Search for specific terms in the document:")
        if st.button("Search Document"):
            if search_term:
                with st.spinner("Searching document..."):
                    results = vectorstore.similarity_search(search_term, k=3)
                    st.subheader("Search Results:")
                    for i, doc in enumerate(results):
                        with st.expander(f"Result {i+1}"):
                            st.text(doc.page_content)
                            st.caption(f"Metadata: {doc.metadata}")
            else:
                st.warning("Please enter a search term.")

if __name__ == "__main__":
    main()