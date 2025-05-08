# app.py
import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.text_splitter import TokenTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from dotenv import load_dotenv
import os
import tempfile
import logging

# Cài đặt logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

def init_session_state():
    st.session_state.setdefault('history', [])
    st.session_state.setdefault('generated', ["Các phản hồi của bạn sẽ được hiển thị ở đây."])
    st.session_state.setdefault('past', ["Xin chào! 👋 Hãy hỏi tôi bất cứ điều gì về tài liệu PDF 📄"])

def reset_chat():
    st.session_state['history'] = []
    st.session_state['past'] = ["Xin chào! 👋 Hãy hỏi tôi bất cứ điều gì về tài liệu PDF 📄"]
    st.session_state.pop('chain', None)
    st.session_state.pop('vector_store', None)

def conversation_chat(query, chain, history):
    try:
        result = chain.invoke({"question": query, "chat_history": history})
        answer = result["answer"].strip()
        history.append((query, answer))
        return answer
    except Exception as e:
        logger.error(f"Lỗi hội thoại: {e}")
        error_msg = f"Xin lỗi, tôi không thể xử lý yêu cầu này ngay bây giờ. Lỗi: {e}"
        history.append((query, error_msg))
        return error_msg

def display_chat(chain):
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input("Câu hỏi:", placeholder="Hãy hỏi về nội dung của tài liệu PDF", key='input')
        if st.form_submit_button("Gửi") and user_input:
            with st.spinner("Đang tạo phản hồi..."):
                output = conversation_chat(user_input, chain, st.session_state['history'])
                st.session_state['past'].append(user_input)
                st.session_state['generated'].append(output)

    for i in reversed(range(len(st.session_state['generated']))):
        if i < len(st.session_state["past"]):
            message(st.session_state["past"][i], is_user=True, key=f"{i}_user")
        message(st.session_state["generated"][i], key=str(i))

def create_chain(vector_store):
    llm = AzureChatOpenAI(
        deployment_name=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
        model_name=os.getenv("AZURE_OPENAI_CHAT_MODEL_NAME"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        openai_api_version=os.getenv("OPENAI_API_CHAT_MODEL_VERSION"),
        openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        openai_api_type="azure"
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 2})

    template = """Hãy sử dụng ngữ cảnh sau để trả lời câu hỏi. Trả lời **chỉ bằng tiếng Việt**:
Ngữ cảnh:
{context}

Câu hỏi: {question}

Trả lời:"""
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={'prompt': prompt},
        return_source_documents=False
    )

def process_pdfs(files):
    documents = []
    for f in files:
        temp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(f.read())
                temp_file_path = tmp.name

            loader = PyPDFLoader(temp_file_path)
            docs = loader.load()
            documents.extend(docs)
            logger.info(f"Đã tải {len(docs)} trang từ {f.name}")

        except Exception as e:
            logger.error(f"Lỗi khi tải PDF {f.name}: {e}")
            st.error(f"Không thể tải PDF: {f.name}. Vui lòng kiểm tra định dạng tệp.")
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                    logger.info(f"Đã xóa tệp tạm thời: {temp_file_path}")
                except OSError as e:
                    logger.error(f"Lỗi khi xóa tệp tạm thời {temp_file_path}: {e}")

    if not documents:
        st.error("Không tài liệu nào được tải thành công từ các tệp PDF.")
        return None

    splitter = TokenTextSplitter(chunk_size=200, chunk_overlap=40)
    chunks = splitter.split_documents(documents)
    logger.info(f"Đã chia tài liệu thành {len(chunks)} đoạn")

    try:
        embeddings = AzureOpenAIEmbeddings(
            deployment=os.getenv("OPENAI_ADA_EMBEDDING_DEPLOYMENT_NAME"),
            model=os.getenv("OPENAI_ADA_EMBEDDING_MODEL_NAME"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            openai_api_type="azure",
            chunk_size=100
        )
        logger.info("Đã khởi tạo AzureOpenAIEmbeddings thành công.")
    except Exception as e:
        logger.error(f"Lỗi khi khởi tạo AzureOpenAIEmbeddings: {e}")
        st.error("Không thể khởi tạo Embeddings từ Azure OpenAI. Vui lòng kiểm tra biến môi trường.")
        st.error(f"Lỗi cụ thể: {e}")
        return None

    try:
        logger.info("Đang tạo Chroma vector store...")
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name="pdf_collection",
            persist_directory="./chroma_db"
        )
        logger.info("Đã tạo và lưu trữ Chroma vector store.")
        return vector_store
    except Exception as e:
        logger.error(f"Lỗi khi tạo vector store: {e}")
        st.error("Không thể tạo vector store. Vui lòng kiểm tra cấu hình embeddings và lưu trữ.")
        st.error(f"Lỗi cụ thể: {e}")
        return None

def main():
    st.set_page_config(page_title="Trò chuyện với PDF", page_icon="📚")
    st.title("📘 Trò chuyện với tài liệu bạn đã tải lên.")
    init_session_state()

    st.sidebar.header("Tải lên & Tùy chọn")
    uploaded_files = st.sidebar.file_uploader("Tải lên tệp PDF", type="pdf", accept_multiple_files=True)

    if st.sidebar.button("🔄 Đặt lại Trò chuyện"):
        reset_chat()
        st.rerun()

    process_button = st.sidebar.button("🚀 Xử lý tệp đã tải lên")

    if uploaded_files and (process_button or "chain" not in st.session_state or st.session_state['chain'] is None):
        st.session_state.pop('chain', None)
        st.session_state.pop('vector_store', None)
        reset_chat()

        with st.spinner("Đang xử lý tài liệu... Quá trình này có thể mất vài phút đối với các tệp lớn."):
            vector_store = process_pdfs(uploaded_files)
            if vector_store:
                st.session_state['vector_store'] = vector_store
                st.session_state['chain'] = create_chain(vector_store)
                st.success("Tài liệu đã được xử lý thành công! Bây giờ bạn có thể đặt câu hỏi.")

    if "chain" in st.session_state and st.session_state['chain'] is not None:
        display_chat(st.session_state["chain"])
    else:
        st.info("Vui lòng tải lên tài liệu PDF và nhấn 'Xử lý tệp đã tải lên' để bắt đầu.")

if __name__ == "__main__":
    main()
