# app.py
import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from dotenv import load_dotenv
import os
import tempfile
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()


def init_session_state():
    st.session_state.setdefault('history', [])
    st.session_state.setdefault('generated', ["Your responses will get displayed over here."])
    st.session_state.setdefault('past', ["Hi there! 👋 Ask me anything about the PDF documents 📄"])

def reset_chat():
    st.session_state['history'] = []
    st.session_state['past'] = ["Hi there! 👋 Ask me anything about the PDF documents 📄"]
    st.session_state.pop('chain', None)
    st.session_state.pop('vector_store', None)

def conversation_chat(query, chain, history):
    try:
        result = chain.invoke({"question": query, "chat_history": history})
        answer = result["answer"].strip()
        history.append((query, answer))
        return answer
    except Exception as e:
        logger.error(f"Chat error: {e}")
        error_msg = f"Sorry, I cannot process this request at the moment. Error: {e}"
        history.append((query, error_msg))
        return error_msg

def display_chat(chain):
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input("Question:", placeholder="Ask about the content of your PDF", key='input')
        if st.form_submit_button("Send") and user_input:
            with st.spinner("Generating response..."):
                output = conversation_chat(user_input, chain, st.session_state['history'])
                st.session_state['past'].append(user_input)
                st.session_state['generated'].append(output)

    # Display chat messages using streamlit_chat
    for i in reversed(range(len(st.session_state['generated']))):
        if i < len(st.session_state["past"]):
            message(st.session_state["past"][i], is_user=True, key=f"{i}_user")
        message(st.session_state["generated"][i], key=str(i))



def create_chain(vector_store):
    # Use ChatOpenAI with the specific chat endpoint
    llm = AzureChatOpenAI(deployment_name=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
                      model_name=os.getenv("AZURE_OPENAI_CHAT_MODEL_NAME"),
                      azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                      openai_api_version=os.getenv("OPENAI_API_CHAT_MODEL_VERSION"),
                      openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                      openai_api_type="azure")
    
    # Memory for the conversation history
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    # Retriever for fetching relevant documents from the vector store
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    # Prompt template for the conversational retrieval chain
    template = """Use the following context to answer the question:
Context:
{context}

Question: {question}

Answer:"""
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    # Create the conversational retrieval chain
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={'prompt': prompt},
        return_source_documents=False # Set to True if you want to show source chunks
    )

def process_pdfs(files):
    documents = []
    for f in files:
        temp_file_path = None
        try:
            # Create a temporary file to save the uploaded PDF
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(f.read())
                temp_file_path = tmp.name # Store the path to the temporary file

            # Explicitly close the file handle before passing the path to the loader
            # This helps prevent PermissionError on Windows
            # The 'with' block ensures the file object is closed upon exiting

            # Now load the document using the file path
            loader = PyPDFLoader(temp_file_path)
            docs = loader.load()
            documents.extend(docs)
            logger.info(f"Loaded {len(docs)} pages from {f.name}")

        except Exception as e:
            logger.error(f"Error loading PDF {f.name}: {e}")
            st.error(f"Failed to load PDF: {f.name}. Please check the file format.")
            # Continue processing other files even if one fails

        finally:
            # Ensure the temporary file is deleted even if loading fails
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                    logger.info(f"Deleted temporary file: {temp_file_path}")
                except OSError as e:
                    logger.error(f"Error deleting temporary file {temp_file_path}: {e}")
                    # Log the error but allow the app to continue

    if not documents:
        st.error("No documents were successfully loaded from the PDF files.")
        return None # Return None if no documents were loaded

    # Split the documents into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=250)
    chunks = splitter.split_documents(documents)
    logger.info(f"Split documents into {len(chunks)} chunks")

    # Create embeddings using Azure OpenAI Embeddings
    try:
        embeddings=AzureOpenAIEmbeddings(
        deployment=os.getenv("OPENAI_ADA_EMBEDDING_DEPLOYMENT_NAME"),
        model=os.getenv("OPENAI_ADA_EMBEDDING_MODEL_NAME"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        openai_api_type="azure",
        chunk_size=16)


        logger.info("AzureOpenAIEmbeddings initialized successfully.")
    except Exception as e:
         logger.error(f"Error initializing AzureOpenAIEmbeddings: {e}")
         st.error("Failed to initialize Azure OpenAI Embeddings. Check your environment variables and Azure deployment.")
         st.error(f"Specific error: {e}") # Add specific error for debugging
         return None

    # Create and persist the Chroma vector store
    try:
        logger.info("Creating Chroma vector store...")
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name="pdf_collection",
            persist_directory="./chroma_db" # Data will be stored here
        )
        logger.info("Chroma vector store created and persisted.")
        return vector_store
    except Exception as e:
        logger.error(f"Error creating Chroma vector store: {e}")
        st.error("Failed to create vector store. Check your embeddings setup and ChromaDB persistence.")
        st.error(f"Specific error: {e}") # Add specific error for debugging
        return None

# === Main App ===

def main():
    st.set_page_config(page_title="PDF Chatbot", page_icon="📚")
    st.title("📘 Chat with the uploded document.")
    init_session_state()
    
    # Sidebar for file upload and actions
    st.sidebar.header("Upload & Options")
    uploaded_files = st.sidebar.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

    # Reset button
    if st.sidebar.button("🔄 Reset Chatbot"):
        reset_chat()
        st.rerun() # Rerun the app to clear state and UI

    # Process files logic
    process_button = st.sidebar.button("🚀 Process Uploaded Files")

    if uploaded_files and (process_button or "chain" not in st.session_state or st.session_state['chain'] is None):
        # Clear existing state related to processing if new files are uploaded or process is triggered
        st.session_state.pop('chain', None)
        st.session_state.pop('vector_store', None)
        reset_chat() # Reset chat interface

        with st.spinner("Processing documents... This may take a few minutes for large files."):
            vector_store = process_pdfs(uploaded_files)
            if vector_store: # Only proceed if vector store creation was successful
                st.session_state['vector_store'] = vector_store
                st.session_state['chain'] = create_chain(vector_store)
                st.success("Documents processed successfully! You can now ask questions.")
                # No need to rerun here, the state is set and chat will display

    # Display chat interface if the chain is ready
    if "chain" in st.session_state and st.session_state['chain'] is not None:
        display_chat(st.session_state["chain"])
    else:
        # Display guidance if no files are processed yet
        st.info("Please upload PDF documents and click 'Process Uploaded Files' to start.")
        # st.warning("Ensure your Azure OpenAI environment variables in the .env file are correctly set, including the full endpoint URLs.")


if __name__ == "__main__":
    main()