# app.py

import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain_community.llms import HuggingFaceHub
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate 

import os
import tempfile

from dotenv import load_dotenv
load_dotenv()

HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
# print(HF_TOKEN) # Removed print statement

def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Xin chào! Hãy hỏi tôi bất kỳ điều gì về tài liệu PDF 📄"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Chào bạn! 👋"]

def conversation_chat(query, chain, history):
    result = chain.invoke({"question": query, "chat_history": history})
    raw_answer = result["answer"]

    # Post-process the answer to remove the unwanted prefix
    prefix_marker = "Trả lời hữu ích:" # <--- Updated marker
    prefix_index = raw_answer.find(prefix_marker)

    if prefix_index != -1:
        cleaned_answer = raw_answer[prefix_index + len(prefix_marker):].strip()
    else:

        cleaned_answer = raw_answer.strip()

    history.append((query, cleaned_answer))
    return cleaned_answer

def display_chat_history(chain):
    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Câu hỏi:", placeholder="Hỏi về nội dung trong PDF của bạn", key='input')
            submit_button = st.form_submit_button(label='Gửi')

        if submit_button and user_input:
            with st.spinner('Đang tạo phản hồi...'):
                output = conversation_chat(user_input, chain, st.session_state['history'])

            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                message(st.session_state["generated"][i], key=str(i), avatar_style="fun-emoji")





def create_conversational_chain(vector_store):
    llm = HuggingFaceHub(
        repo_id="LR-AI-Labs/vbd-llama2-7B-50b-chat",
        model_kwargs={"temperature": 0.9, "max_new_tokens": 1024},
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # --- Define a custom prompt template with Vietnamese instruction ---
    template_with_vietnamese_instruction = """Sử dụng các phần ngữ cảnh sau đây để trả lời câu hỏi ở cuối BẰNG TIẾNG VIỆT. Nếu bạn không biết câu trả lời, chỉ cần nói 'Tôi không biết' — đừng cố bịa ra câu trả lời. Đảm bảo rằng câu trả lời có đầy đủ thông tin, được viết đúng cách và rõ ràng dựa trên ngữ cảnh được cung cấp.

    {context}

    Câu hỏi: {question}
    """

    CUSTOM_PROMPT = PromptTemplate(
        template=template_with_vietnamese_instruction,
        input_variables=["context", "question"]
    )
    # ------------------------------------------------------------------

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type='stuff',
        retriever=vector_store.as_retriever(search_kwargs={"k": 1}),
        memory=memory,
        # Pass the custom prompt to the combine_docs_chain
        combine_docs_chain_kwargs={'prompt': CUSTOM_PROMPT}
    )
    return chain

def main():
    initialize_session_state()
    st.title("📘 Chatbot Tài Liệu PDF - Hỗ trợ tiếng Việt 🇻🇳")

    st.sidebar.title("Tải lên tài liệu PDF")
    uploaded_files = st.sidebar.file_uploader("Chọn file PDF", accept_multiple_files=True)

    if uploaded_files:
        text = []
        for file in uploaded_files:
            file_extension = os.path.splitext(file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(file.read())
                temp_file_path = temp_file.name

            if file_extension == ".pdf":
                loader = PyPDFLoader(temp_file_path)
                text.extend(loader.load())
                os.remove(temp_file_path)

        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        text_chunks = text_splitter.split_documents(text)

        # Embeddings (Multilingual)
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={"device": "cpu"} # Consider "cuda" if you have a GPU
        )

        # Create vector DB
        # Ensure embeddings object is passed correctly
        vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)


        # QA chain
        chain = create_conversational_chain(vector_store)

        # Chat UI
        display_chat_history(chain)


if __name__ == "__main__":
    main()