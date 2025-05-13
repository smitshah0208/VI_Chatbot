"""
Complete vi_main.py file with robust error handling for cross-environment compatibility
"""
import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from dotenv import load_dotenv
import os
import logging
import tempfile
import shutil
import re
import sys
import uuid

# Configure logging to show more detailed information
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import PDF processing function
try:
    from process_pdf import get_md_text
    logger.info("Đã nhập hàm get_md_text từ process_pdf.py thành công.")
except ImportError:
    logger.error("Không tìm thấy process_pdf.py hoặc hàm get_md_text. Vui lòng đảm bảo tệp tồn tại.")
    
    # Define a dummy function to prevent errors if import fails
    def get_md_text(pdf_path, output_dir_base="md_output", poppler_path=None):
        logger.error("Hàm get_md_text không khả dụng do lỗi nhập.")
        return ""

load_dotenv()

# Configuration constants
MARKDOWN_OUTPUT_FILENAME = "document.md"
MARKDOWN_OUTPUT_DIR = "md_output"
MARKDOWN_OUTPUT_FILE_PATH = os.path.join(MARKDOWN_OUTPUT_DIR, MARKDOWN_OUTPUT_FILENAME)

# Multiple potential DB directories to try
DB_DIRECTORIES = [
    "./md_chroma_db",                              # Original location
    os.path.join(tempfile.gettempdir(), "md_chroma_db"),  # Temp directory
    os.path.expanduser("~/md_chroma_db"),          # Home directory
    "/tmp/md_chroma_db"                            # Linux/Unix tmp directory
]

# Default to first option, but will be updated if needed
CHROMA_DB_DIR = DB_DIRECTORIES[0]

# Define the fixed Poppler path - adapt as needed for different environments
if sys.platform.startswith('win'):
    FIXED_POPPLER_PATH = r".\poppler-24.08.0\Library\bin"
else:
    FIXED_POPPLER_PATH = None  # Let system find it on Linux/Mac

# Regex to identify markdown image lines
IMAGE_LINE_REGEX = re.compile(r'^!\[.*?\]\(.*?\)$')


def init_session_state():
    """Initializes Streamlit session state variables."""
    st.session_state.setdefault('history', [])
    st.session_state.setdefault('generated', ["Các phản hồi của bạn sẽ được hiển thị ở đây."])
    st.session_state.setdefault('past', ["Xin chào! 👋 Hãy hỏi tôi bất cứ điều gì về tài liệu PDF đã tải lên 📄"])
    st.session_state.setdefault('processed_pdf_name', None)
    st.session_state.setdefault('vector_store_ready', False)
    st.session_state.setdefault('current_db_path', CHROMA_DB_DIR)
    st.session_state.setdefault('unique_session_id', str(uuid.uuid4()))


def safely_remove_path(path, is_dir=False):
    """Safely removes a file or directory with robust error handling."""
    if not os.path.exists(path):
        return True
        
    try:
        if is_dir:
            shutil.rmtree(path)
            logger.info(f"Đã xóa thư mục: {path}")
        else:
            os.remove(path)
            logger.info(f"Đã xóa tệp: {path}")
        return True
    except (OSError, PermissionError) as e:
        logger.warning(f"Không thể xóa {'thư mục' if is_dir else 'tệp'} {path}: {e}")
        
        # If we can't remove it, try renaming it with a unique identifier
        try:
            unique_suffix = st.session_state.get('unique_session_id', str(uuid.uuid4()))[:8]
            new_path = f"{path}_{unique_suffix}_old"
            os.rename(path, new_path)
            logger.info(f"Không thể xóa, đã đổi tên thành: {new_path}")
            return True
        except Exception as rename_err:
            logger.error(f"Cả xóa và đổi tên đều thất bại cho {path}: {rename_err}")
            return False


def reset_chat():
    """Resets the chat history and processing state with improved cleanup."""
    # Reset session state
    st.session_state['history'] = []
    st.session_state['past'] = ["Xin chào! 👋 Hãy hỏi tôi bất cứ điều gì về tài liệu PDF đã tải lên 📄"]
    st.session_state.pop('chain', None)
    st.session_state.pop('vector_store', None)
    st.session_state['processed_pdf_name'] = None
    st.session_state['vector_store_ready'] = False
    
    # Generate new session ID for unique filenames if needed
    st.session_state['unique_session_id'] = str(uuid.uuid4())
    
    # Clean up files with robust error handling
    safely_remove_path(MARKDOWN_OUTPUT_FILE_PATH, is_dir=False)
    safely_remove_path(MARKDOWN_OUTPUT_DIR, is_dir=True)
    
    # Clean up all potential DB directories
    for db_dir in DB_DIRECTORIES:
        if os.path.exists(db_dir):
            safely_remove_path(db_dir, is_dir=True)
    
    # Reset current DB path to default
    st.session_state['current_db_path'] = CHROMA_DB_DIR
    
    logger.info("Chat và trạng thái xử lý đã được đặt lại.")


def conversation_chat(query, chain, history):
    """Handles the conversational chat interaction."""
    try:
        logger.info(f"Nhận câu hỏi người dùng: {query}")
        result = chain.invoke({"question": query, "chat_history": history})
        answer = result["answer"].strip()
        history.append((query, answer))
        logger.info("Đã tạo phản hồi thành công.")
        return answer
    except Exception as e:
        logger.error(f"Lỗi hội thoại: {e}")
        error_msg = f"Xin lỗi, tôi không thể xử lý yêu cầu này ngay bây giờ. Lỗi: {e}"
        history.append((query, error_msg))
        return error_msg


def display_chat(chain):
    """Displays the chat interface."""
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input("Câu hỏi:", placeholder=f"Hãy hỏi về nội dung của tài liệu {st.session_state.get('processed_pdf_name', 'PDF')}", key='input')
        if st.form_submit_button("Gửi") and user_input:
            with st.spinner("Đang tạo phản hồi..."):
                output = conversation_chat(user_input, chain, st.session_state['history'])
                st.session_state['past'].append(user_input)
                st.session_state['generated'].append(output)
            # Rerun to update the chat display
            st.rerun()

    # Display chat messages
    if st.session_state['generated']:
        for i in reversed(range(len(st.session_state['generated']))):
            if i < len(st.session_state["past"]):
                message(st.session_state["past"][i], is_user=True, key=f"{i}_user")
            message(st.session_state["generated"][i], key=str(i))


def create_chain(vector_store):
    """Creates a conversational retrieval chain."""
    try:
        logger.info("Đang tạo chuỗi hội thoại...")
        llm = AzureChatOpenAI(
            deployment_name=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
            model_name=os.getenv("AZURE_OPENAI_CHAT_MODEL_NAME"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            openai_api_version=os.getenv("OPENAI_API_CHAT_MODEL_VERSION"),
            openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            openai_api_type="azure",
            temperature=0
        )

        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        # Adjust search_kwargs if needed, e.g., number of relevant documents to retrieve
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

        # Prompt for markdown context
        template = """Sử dụng các đoạn văn bản sau từ tài liệu để trả lời câu hỏi. Trả lời **chỉ bằng tiếng Việt**.
Nếu bạn không biết câu trả lời, chỉ cần nói rằng bạn không biết, đừng cố tạo ra câu trả lời.

Ngữ cảnh:
{context}

Câu hỏi: {question}

Trả lời:"""
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])

        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            combine_docs_chain_kwargs={'prompt': prompt},
            return_source_documents=False
        )
        logger.info("Đã tạo chuỗi hội thoại thành công.")
        return chain
    except Exception as e:
        logger.error(f"Lỗi khi tạo chuỗi hội thoại: {e}")
        st.error("Không thể tạo chuỗi hội thoại. Vui lòng kiểm tra cấu hình Azure OpenAI.")
        st.error(f"Lỗi cụ thể: {e}")
        return None


def process_markdown_file_lines(markdown_file_path: str):
    """
    Reads a markdown file and creates document chunks based on content
    following image lines.
    """
    chunks = []
    current_chunk_lines = []
    if not os.path.exists(markdown_file_path):
        logger.error(f"Lỗi: Tệp markdown không tìm thấy tại {markdown_file_path}")
        st.error(f"Lỗi: Tệp markdown không tìm thấy tại {markdown_file_path}. Quá trình chuyển đổi PDF sang Markdown có thể đã thất bại hoặc tệp không được lưu đúng cách.")
        return None

    try:
        logger.info(f"Đang đọc tệp markdown và tạo chunks dựa trên hình ảnh: {markdown_file_path}")
        with open(markdown_file_path, 'r', encoding='utf-8') as f:
            # Read the file line by line
            for i, line in enumerate(f):
                cleaned_line = line.strip()

                # Check if the cleaned line is an image line
                if IMAGE_LINE_REGEX.match(cleaned_line):
                    # If we encounter an image line, the content accumulated so far
                    # (if any) forms a chunk.
                    if current_chunk_lines:
                        chunk_content = "\n".join(current_chunk_lines)
                        # Create a Document object for the chunk
                        chunks.append(Document(page_content=chunk_content, metadata={"source": os.path.basename(markdown_file_path), "start_line": i - len(current_chunk_lines) + 1}))
                        logger.debug(f"Đã tạo chunk từ dòng {i - len(current_chunk_lines) + 1} đến {i-1}")

                    # Clear the buffer to start accumulating content for the next chunk
                    current_chunk_lines = []
                    # Do NOT add the image line itself to the current chunk buffer

                elif cleaned_line: # If it's not an image line and not an empty line
                    # Add the cleaned line to the current chunk buffer
                    current_chunk_lines.append(cleaned_line)

        # After the loop, add any remaining content in the buffer as the last chunk
        if current_chunk_lines:
            chunk_content = "\n".join(current_chunk_lines)
            chunks.append(Document(page_content=chunk_content, metadata={"source": os.path.basename(markdown_file_path), "start_line": i - len(current_chunk_lines) + 2}))
            logger.debug(f"Đã tạo chunk cuối cùng từ dòng {i - len(current_chunk_lines) + 2} đến {i+1}")

        logger.info(f"Đã tạo {len(chunks)} chunks dựa trên hình ảnh từ {markdown_file_path}")
        return chunks

    except Exception as e:
        logger.error(f"Lỗi khi đọc tệp markdown và tạo chunks {markdown_file_path}: {e}")
        st.error(f"Không thể đọc tệp markdown hoặc tạo chunks: {markdown_file_path}. Vui lòng kiểm tra định dạng tệp hoặc quyền truy cập.")
        st.error(f"Lỗi cụ thể: {e}")
        return None


def find_writable_directory():
    """Find a writable directory for the database from the list of potential directories."""
    for db_path in DB_DIRECTORIES:
        try:
            # Ensure the directory exists
            os.makedirs(db_path, exist_ok=True)
            
            # Test write permissions
            test_file = os.path.join(db_path, "test_write.txt")
            with open(test_file, 'w') as f:
                f.write("Testing write permissions")
            os.remove(test_file)
            
            logger.info(f"Tìm thấy thư mục có thể ghi: {db_path}")
            return db_path
        except Exception as e:
            logger.warning(f"Không thể sử dụng thư mục {db_path}: {e}")
            continue
            
    # If no writable directory found, use temp directory with unique name
    fallback_dir = os.path.join(tempfile.gettempdir(), f"md_chroma_{st.session_state['unique_session_id']}")
    try:
        os.makedirs(fallback_dir, exist_ok=True)
        logger.info(f"Sử dụng thư mục tạm thời duy nhất: {fallback_dir}")
        return fallback_dir
    except Exception as e:
        logger.error(f"Không thể tạo thư mục tạm thời: {e}")
        return None


def create_vector_store_from_documents(documents: list[Document]):
    """
    Generates a Chroma vector store with robust error handling and fallback options.
    """
    if not documents:
        st.error("Không có tài liệu nào để tạo vector store.")
        return None

    # Initialize embeddings
    try:
        logger.info("Đang khởi tạo AzureOpenAIEmbeddings...")
        embeddings = AzureOpenAIEmbeddings(
            deployment=os.getenv("OPENAI_ADA_EMBEDDING_EMBEDDING_DEPLOYMENT_NAME"),
            model=os.getenv("OPENAI_ADA_EMBEDDING_MODEL_NAME"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            openai_api_type="azure",
            chunk_size=100
        )
        logger.info("Đã khởi tạo AzureOpenAIEmbeddings thành công.")
    except Exception as e:
        logger.error(f"Lỗi khi khởi tạo AzureOpenAIEmbeddings: {e}")
        st.error("Không thể khởi tạo Embeddings từ Azure OpenAI. Vui lòng kiểm tra biến môi trường.")
        st.error(f"Lỗi cụ thể: {e}")
        return None

    # Find a writable directory
    db_path = find_writable_directory()
    if not db_path:
        logger.warning("Không tìm thấy thư mục có thể ghi, sử dụng vector store trong bộ nhớ.")
        try:
            # Create in-memory vector store as last resort
            vector_store = Chroma.from_documents(
                documents=documents,
                embedding=embeddings
            )
            logger.info("Đã tạo vector store trong bộ nhớ thành công")
            return vector_store
        except Exception as e:
            logger.critical(f"Không thể tạo vector store ngay cả trong bộ nhớ: {e}")
            st.error("Không thể tạo vector store. Vui lòng kiểm tra cấu hình.")
            st.error(f"Lỗi cụ thể: {e}")
            return None
    
    # Store the current DB path in session state
    st.session_state['current_db_path'] = db_path
    
    # Create vector store in the writable directory
    try:
        logger.info(f"Đang tạo Chroma vector store tại {db_path}...")
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            collection_name="markdown_image_chunk_collection",
            persist_directory=db_path
        )
        logger.info(f"Đã tạo thành công Chroma vector store tại: {db_path}")
        return vector_store
    except Exception as e:
        logger.error(f"Lỗi khi tạo vector store tại {db_path}: {e}")
        
        # Try once more without persistence as fallback
        try:
            logger.info("Thử tạo vector store trong bộ nhớ...")
            vector_store = Chroma.from_documents(
                documents=documents,
                embedding=embeddings
            )
            logger.info("Đã tạo vector store trong bộ nhớ thành công")
            return vector_store
        except Exception as e2:
            logger.critical(f"Không thể tạo vector store ngay cả trong bộ nhớ: {e2}")
            st.error("Không thể tạo vector store. Vui lòng kiểm tra cấu hình.")
            st.error(f"Lỗi cụ thể: {e2}")
            return None


def main():
    """Main function to run the Streamlit application."""
    st.set_page_config(page_title="Trò chuyện với PDF", page_icon="📚")
    st.title("📘 Trò chuyện với tài liệu PDF của bạn.")
    init_session_state()

    st.sidebar.header("Tải lên & Tùy chọn")
    uploaded_files = st.sidebar.file_uploader("Tải lên tệp PDF", type="pdf", accept_multiple_files=False)

    # Determine Poppler path based on platform
    poppler_path = FIXED_POPPLER_PATH
    logger.info(f"Sử dụng đường dẫn Poppler: {poppler_path}")

    # Add a status indicator for the vector store
    if st.session_state.get('vector_store_ready'):
        st.sidebar.success(f"PDF đã sẵn sàng: {st.session_state.get('processed_pdf_name', '')}")
        db_path = st.session_state.get('current_db_path', 'bộ nhớ')
        st.sidebar.info(f"Cơ sở dữ liệu: {db_path}")
    else:
        st.sidebar.warning("")

    if st.sidebar.button("🔄 Đặt lại Trò chuyện"):
        reset_chat()
        st.rerun()

    process_button = st.sidebar.button("🚀 Xử lý tệp PDF đã tải lên")

    # Process the uploaded PDF file
    if uploaded_files and (process_button or not st.session_state.get('vector_store_ready')):
        current_uploaded_file_name = uploaded_files.name if uploaded_files else None
        
        if current_uploaded_file_name != st.session_state.get('processed_pdf_name') or process_button or not st.session_state.get('vector_store_ready'):
            st.session_state.pop('chain', None)
            st.session_state.pop('vector_store', None)
            reset_chat()  # Reset chat before processing new document

            st.session_state['processed_pdf_name'] = current_uploaded_file_name

            with st.spinner(f"Đang xử lý tệp PDF: {uploaded_files.name}..."):
                temp_pdf_path = None
                try:
                    # Create the output directory if it doesn't exist
                    os.makedirs(MARKDOWN_OUTPUT_DIR, exist_ok=True)
                    
                    # Save the uploaded PDF to a temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_files.getvalue())
                        temp_pdf_path = tmp.name
                    logger.info(f"Đã lưu tệp PDF tạm thời tại: {temp_pdf_path}")

                    # Convert PDF to markdown
                    logger.info(f"Đang chuyển đổi PDF sang Markdown: {temp_pdf_path}")
                    markdown_text = get_md_text(temp_pdf_path, output_dir_base=MARKDOWN_OUTPUT_DIR, poppler_path=poppler_path)
                    logger.info(f"Quá trình chuyển đổi PDF sang Markdown hoàn tất. Kết quả được lưu tại {MARKDOWN_OUTPUT_FILE_PATH}")

                    # Create chunks based on image lines
                    chunks = process_markdown_file_lines(MARKDOWN_OUTPUT_FILE_PATH)

                    if chunks and len(chunks) > 0:
                        # Create vector store with robust error handling
                        vector_store = create_vector_store_from_documents(chunks)
                        
                        if vector_store:
                            st.session_state['vector_store'] = vector_store
                            st.session_state['chain'] = create_chain(vector_store)
                            st.session_state['vector_store_ready'] = True
                            st.success(f"Tài liệu PDF '{uploaded_files.name}' đã được xử lý thành công! Bây giờ bạn có thể đặt câu hỏi.")
                        else:
                            st.error("Không thể tạo vector store từ các chunks.")
                    else:
                        st.error("Không có nội dung nào được trích xuất hoặc chunked từ tệp Markdown.")

                except Exception as e:
                    logger.error(f"Lỗi trong quá trình xử lý PDF hoặc Markdown: {e}")
                    st.error("Đã xảy ra lỗi trong quá trình xử lý tài liệu. Vui lòng thử lại.")
                    st.error(f"Lỗi cụ thể: {e}")
                finally:
                    # Clean up the temporary PDF file
                    if temp_pdf_path and os.path.exists(temp_pdf_path):
                        try:
                            os.unlink(temp_pdf_path)
                            logger.info(f"Đã xóa tệp PDF tạm thời: {temp_pdf_path}")
                        except OSError as e:
                            logger.warning(f"Lỗi khi xóa tệp PDF tạm thời {temp_pdf_path}: {e}")

    # Display chat interface if ready
    if st.session_state.get('chain') is not None and st.session_state.get('vector_store_ready'):
        display_chat(st.session_state["chain"])
    elif uploaded_files and not st.session_state.get('vector_store_ready'):
        st.info("Đang chờ xử lý tệp PDF. Vui lòng nhấn 'Xử lý tệp PDF đã tải lên'.")
    else:
        st.info("Vui lòng tải lên tài liệu PDF và nhấn 'Xử lý tệp PDF đã tải lên' để bắt đầu.")


if __name__ == "__main__":
    main()