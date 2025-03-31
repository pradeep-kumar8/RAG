import os
import streamlit as st
import hashlib
from langchain.llms import HuggingFaceHub
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from streamlit_pdf_viewer import pdf_viewer

# Ensure required libraries are installed
os.system("pip install tabulate openpyxl")

def init_page() -> None:
    st.set_page_config(page_title="PDF Chatbot")
    st.subheader("💬 PDF Chat with multi LLMs")
    # st.write("Created by Pradeep Kumar")

def init_messages() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content="You are a helpful AI assistant. Reply in markdown format.")
        ]

def main() -> None:
    init_page()
    init_messages()

    # Initialize session state variables
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None
    if 'current_file_hash' not in st.session_state:
        st.session_state.current_file_hash = None

    # Sidebar: LLM selection and PDF file uploader
    with st.sidebar:
        st.title("Options")
        
        token = st.text_input("HuggingFace Token")
        selected_model = st.selectbox(
            "Select LLM",
            options=[
                "deepseek-ai/DeepSeek-V3",
                "Qwen/Qwen2.5-7B-Instruct",
                "meta-llama/Llama-3.1-8B-Instruct",
                "mistralai/Mistral-7B-Instruct-v0.3",
                "bigscience/bloom"
            ],
            index=0,
            key="selected_model"
        )
        uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
        if uploaded_file:
            binary_data = uploaded_file.getvalue()
            pdf_viewer(input=binary_data, width=300)

    # Initialize LLM
    llm = HuggingFaceHub(
        repo_id=st.session_state.selected_model,
        model_kwargs={"temperature": 0.5, "max_length": 500},
        huggingfacehub_api_token=token,
    )

    if uploaded_file:
        # Compute file hash to check for changes
        file_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest()

        # Process file if new or not yet processed
        if st.session_state.current_file_hash != file_hash or st.session_state.vectorstore is None:
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            loader = PyPDFLoader("temp.pdf")
            pages = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            texts = text_splitter.split_documents(pages)
            embeddings = HuggingFaceEmbeddings()
            vectorstore = FAISS.from_documents(texts, embeddings)
            st.session_state.vectorstore = vectorstore
            st.session_state.current_file_hash = file_hash

        # Chat interface
        if user_input := st.chat_input("Input your question about the PDF:"):
            st.session_state.messages.append(HumanMessage(content=user_input))
            with st.spinner("Analyzing ..."):
                try:
                    qa_chain = RetrievalQA.from_chain_type(
                        llm=llm,
                        chain_type="stuff",
                        retriever=st.session_state.vectorstore.as_retriever()
                    )
                    answer = qa_chain.run(user_input)
                except Exception as e:
                    answer = f"An error occurred: {str(e)}"
            st.session_state.messages.append(AIMessage(content=answer))

        # Display chat messages
        for message in st.session_state.get("messages", []):
            if isinstance(message, AIMessage):
                with st.chat_message("assistant", avatar="👽"):
                    st.markdown(message.content.split(user_input)[-1].strip())
            elif isinstance(message, HumanMessage):
                with st.chat_message("user", avatar="🙋‍♂️"):
                    st.markdown(message.content)

        # Clear conversation button
        if st.button("🧹 Clear Conversation", key="clear_chat"):
            st.session_state.messages = [
                SystemMessage(content="You are a helpful AI assistant. Reply in markdown format.")
            ]
            st.rerun()
    else:
        st.write("Please upload a PDF file to start querying.")

if __name__ == "__main__":
    main()
