import streamlit as st
from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.redis import Redis
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

# Load environment variables from .env file
load_dotenv()

# Ensure OpenAI API key is loaded from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key is None:
    st.error("Please set your OpenAI API key in the .env file.")
    st.stop()

# Sidebar content
with st.sidebar:
    st.title('🤗💬 Chat with your Data')
    st.markdown('''
                ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model
                ''')

def main():
    st.header("Chat with your own PDF")
    
    # File upload
    pdf = st.file_uploader("Upload your PDF", type="pdf")
    
    if pdf is not None:
        # Reading the PDF
        pdf_reader = PdfReader(pdf)
        texts = []
        for page in pdf_reader.pages:
            chunks = page.extract_text()
            if chunks:  # Only add non-empty chunks
                texts.append(chunks)
            else:
                st.warning("Empty text found on one of the pages.")
        
        # Display extracted text (optional, for debugging)
        st.write(texts[:3])  # Show only the first 3 chunks to avoid clutter

        # Initialize embeddings and vector store (Redis)
        embeddings = OpenAIEmbeddings()
        vector_store = Redis.from_texts(texts, embeddings, redis_url="redis://localhost:6379")

        # Input query from user
        query = st.text_input("Ask questions related to your PDF")

        if query:
            # Perform similarity search in Redis
            results = vector_store.similarity_search(query=query, k=3)
            
            # Initialize LLM and QA chain
            llm = OpenAI()
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            response = chain.run(input_documents=results, question=query)
            
            # Display response
            st.write(response)

if __name__ == '__main__':
    main()
