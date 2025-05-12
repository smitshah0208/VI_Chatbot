import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings # Assuming correct import paths
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document # Needed to create Document objects for lines
from dotenv import load_dotenv
import os
import logging
import tempfile
import shutil # Import shutil for directory removal
import re # Import regex for checking image lines

# Assuming process_pdf.py is in the same directory or accessible
try:
    from process_pdf import get_md_text
    logger = logging.getLogger(__name__)
    logger.info("Đã nhập hàm get_md_text từ process_pdf.py thành công.")
except ImportError:
    logger = logging.getLogger(__name__)
    logger.error("Không tìm thấy process_pdf.py hoặc hàm get_md_text. Vui lòng đảm bảo tệp tồn tại.")
    st.error("Lỗi: Không tìm thấy tệp process_pdf.py hoặc hàm get_md_text. Vui lòng kiểm tra cài đặt.")
    # Define a dummy function to prevent errors if import fails, though the app won't work correctly
    def get_md_text(pdf_path, output_dir_base="md_output", poppler_path=None):
        logger.error("Hàm get_md_text không khả dụng do lỗi nhập.")
        return ""


# Cài đặt logging
logging.basicConfig(level=logging.INFO)
load_dotenv()

# Define the fixed markdown output file name
MARKDOWN_OUTPUT_FILENAME = "document.md"
# Define the directory for markdown output
MARKDOWN_OUTPUT_DIR = "md_output"
# Define the full path to the markdown output file
MARKDOWN_OUTPUT_FILE_PATH = os.path.join(MARKDOWN_OUTPUT_DIR, MARKDOWN_OUTPUT_FILENAME)
# Define the directory for Chroma DB
CHROMA_DB_DIR = "./chroma_db"

# Define the fixed Poppler path
FIXED_POPPLER_PATH = r".\poppler-24.08.0\Library\bin"

# Regex to identify markdown image lines
IMAGE_LINE_REGEX = re.compile(r'^!\[.*?\]\(.*?\)$')


def init_session_state():
    """Initializes Streamlit session state variables."""
    st.session_state.setdefault('history', [])
    st.session_state.setdefault('generated', ["Các phản hồi của bạn sẽ được hiển thị ở đây."])
    st.session_state.setdefault('past', ["Xin chào! 👋 Hãy hỏi tôi bất cứ điều gì về tài liệu PDF đã tải lên 📄"])
    st.session_state.setdefault('processed_pdf_name', None) # To store the name of the processed PDF
    st.session_state.setdefault('vector_store_ready', False) # Flag to indicate if vector store is ready


def reset_chat():
    """Resets the chat history and processing state."""
    st.session_state['history'] = []
    st.session_state['past'] = ["Xin chào! 👋 Hãy hỏi tôi bất cứ điều gì về tài liệu PDF đã tải lên 📄"]
    st.session_state.pop('chain', None)
    st.session_state.pop('vector_store', None)
    st.session_state['processed_pdf_name'] = None
    st.session_state['vector_store_ready'] = False
    # Clean up generated markdown file and chroma db directory
    if os.path.exists(MARKDOWN_OUTPUT_FILE_PATH):
        try:
            os.remove(MARKDOWN_OUTPUT_FILE_PATH)
            logger.info(f"Đã xóa tệp markdown cũ: {MARKDOWN_OUTPUT_FILE_PATH}")
        except OSError as e:
            logger.error(f"Lỗi khi xóa tệp markdown cũ {MARKDOWN_OUTPUT_FILE_PATH}: {e}")
    if os.path.exists(MARKDOWN_OUTPUT_DIR):
         try:
             shutil.rmtree(MARKDOWN_OUTPUT_DIR)
             logger.info(f"Đã xóa thư mục markdown cũ: {MARKDOWN_OUTPUT_DIR}")
         except OSError as e:
             logger.error(f"Lỗi khi xóa thư mục markdown cũ {MARKDOWN_OUTPUT_DIR}: {e}")
    if os.path.exists(CHROMA_DB_DIR):
        try:
            shutil.rmtree(CHROMA_DB_DIR)
            logger.info(f"Đã xóa thư mục Chroma DB cũ: {CHROMA_DB_DIR}")
        except OSError as e:
            logger.error(f"Lỗi khi xóa thư mục Chroma DB cũ {CHROMA_DB_DIR}: {e}")

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
            temperature=0 # Set temperature to 0 for more deterministic responses
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
            return_source_documents=False # Set to True if you want to see source chunks
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
                        # Metadata will reflect the source file and the starting line of the chunk
                        chunks.append(Document(page_content=chunk_content, metadata={"source": os.path.basename(markdown_file_path), "start_line": i - len(current_chunk_lines) + 1}))
                        logger.debug(f"Đã tạo chunk từ dòng {i - len(current_chunk_lines) + 1} đến {i-1}")

                    # Clear the buffer to start accumulating content for the next chunk
                    current_chunk_lines = []
                    # Do NOT add the image line itself to the current chunk buffer,
                    # as the chunk is defined by the content *after* the image.

                elif cleaned_line: # If it's not an image line and not an empty line
                    # Add the cleaned line to the current chunk buffer
                    current_chunk_lines.append(cleaned_line)

                # If it's an empty line and not an image line, it's skipped by the elif condition

        # After the loop, add any remaining content in the buffer as the last chunk
        if current_chunk_lines:
            chunk_content = "\n".join(current_chunk_lines)
            chunks.append(Document(page_content=chunk_content, metadata={"source": os.path.basename(markdown_file_path), "start_line": i - len(current_chunk_lines) + 2})) # Adjust start_line for the last chunk
            logger.debug(f"Đã tạo chunk cuối cùng từ dòng {i - len(current_chunk_lines) + 2} đến {i+1}")


        logger.info(f"Đã tạo {len(chunks)} chunks dựa trên hình ảnh từ {markdown_file_path}")
        return chunks

    except Exception as e:
        logger.error(f"Lỗi khi đọc tệp markdown và tạo chunks {markdown_file_path}: {e}")
        st.error(f"Không thể đọc tệp markdown hoặc tạo chunks: {markdown_file_path}. Vui lòng kiểm tra định dạng tệp hoặc quyền truy cập.")
        st.error(f"Lỗi cụ thể: {e}")
        return None


def create_vector_store_from_documents(documents: list[Document]):
    """
    Generates a Chroma vector store using Azure OpenAI embeddings from a list of documents.
    """
    if not documents:
        st.error("Không có tài liệu nào để tạo vector store.")
        return None

    try:
        logger.info("Đang khởi tạo AzureOpenAIEmbeddings...")
        # Ensure these environment variables are set
        embeddings = AzureOpenAIEmbeddings(
            deployment=os.getenv("OPENAI_ADA_EMBEDDING_EMBEDDING_DEPLOYMENT_NAME"), # Corrected env var name
            model=os.getenv("OPENAI_ADA_EMBEDDING_MODEL_NAME"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            openai_api_type="azure",
            chunk_size=100 # This chunk_size is for the embedding model's internal batching
        )
        logger.info("Đã khởi tạo AzureOpenAIEmbeddings thành công.")
    except Exception as e:
        logger.error(f"Lỗi khi khởi tạo AzureOpenAIEmbeddings: {e}")
        st.error("Không thể khởi tạo Embeddings từ Azure OpenAI. Vui lòng kiểm tra biến môi trường.")
        st.error(f"Lỗi cụ thể: {e}")
        return None

    try:
        logger.info(f"Đang tạo Chroma vector store từ các dòng vào {CHROMA_DB_DIR}...")
        # Use the list of line_documents directly
        vector_store = Chroma.from_documents(
            documents=documents, # Use the list of Document objects
            embedding=embeddings,
            collection_name="markdown_image_chunk_collection", # Changed collection name
            persist_directory=CHROMA_DB_DIR # Use the defined persist directory
        )
        logger.info("Đã tạo và lưu trữ Chroma vector store từ các dòng.")
        return vector_store
    except Exception as e:
        logger.error(f"Lỗi khi tạo vector store từ các dòng: {e}")
        st.error("Không thể tạo vector store từ các dòng markdown. Vui lòng kiểm tra cấu hình embeddings và lưu trữ.")
        st.error(f"Lỗi cụ thể: {e}")
        return None


def main():
    """Main function to run the Streamlit application."""
    st.set_page_config(page_title="Trò chuyện với PDF", page_icon="📚")
    st.title("📘 Trò chuyện với tài liệu PDF của bạn.")
    init_session_state()

    st.sidebar.header("Tải lên & Tùy chọn")
    # Using file uploader for PDF files
    uploaded_files = st.sidebar.file_uploader("Tải lên tệp PDF", type="pdf", accept_multiple_files=False) # Changed to False for single file processing

    # Poppler path is now fixed
    poppler_path = FIXED_POPPLER_PATH
    logger.info(f"Sử dụng đường dẫn Poppler cố định: {poppler_path}")


    if st.sidebar.button("🔄 Đặt lại Trò chuyện"):
        reset_chat()
        st.rerun()

    process_button = st.sidebar.button("🚀 Xử lý tệp PDF đã tải lên")

    # Process the uploaded PDF file when the button is clicked and a file is uploaded
    # Also process if a file is uploaded and no vector store is ready yet
    if uploaded_files and (process_button or not st.session_state.get('vector_store_ready')):

        # Check if a new file is uploaded or if we need to re-process the current one
        current_uploaded_file_name = uploaded_files.name if uploaded_files else None
        if current_uploaded_file_name != st.session_state.get('processed_pdf_name') or process_button or not st.session_state.get('vector_store_ready'):

            st.session_state.pop('chain', None)
            st.session_state.pop('vector_store', None)
            reset_chat() # Reset chat before processing new document

            st.session_state['processed_pdf_name'] = current_uploaded_file_name # Store the name of the currently processed PDF

            with st.spinner(f"Đang xử lý tệp PDF: {uploaded_files.name}..."):
                temp_pdf_path = None
                try:
                    # Save the uploaded PDF to a temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_files.getvalue())
                        temp_pdf_path = tmp.name
                    logger.info(f"Đã lưu tệp PDF tạm thời tại: {temp_pdf_path}")

                    # Call the get_md_text function to convert PDF to markdown
                    # This function is assumed to save the output to MARKDOWN_OUTPUT_FILE_PATH
                    logger.info(f"Đang chuyển đổi PDF sang Markdown: {temp_pdf_path}")
                    markdown_text = get_md_text(temp_pdf_path, output_dir_base=MARKDOWN_OUTPUT_DIR, poppler_path=poppler_path)
                    logger.info(f"Quá trình chuyển đổi PDF sang Markdown hoàn tất. Kết quả được lưu tại {MARKDOWN_OUTPUT_FILE_PATH}")

                    # Now read the generated markdown file and create chunks based on image lines
                    chunks = process_markdown_file_lines(MARKDOWN_OUTPUT_FILE_PATH)

                    if chunks:
                        vector_store = create_vector_store_from_documents(chunks)
                        if vector_store:
                            st.session_state['vector_store'] = vector_store
                            st.session_state['chain'] = create_chain(vector_store)
                            st.session_state['vector_store_ready'] = True # Set flag to True
                            st.success(f"Tài liệu PDF '{uploaded_files.name}' đã được xử lý thành công! Bây giờ bạn có thể đặt câu hỏi.")
                        else:
                            st.error("Không thể tạo vector store từ các chunks.")
                    else:
                         st.error("Không có nội dung nào được trích xuất hoặc chunked từ tệp Markdown.")

                except Exception as e:
                    logger.error(f"Lỗi trong quá trình xử lý PDF hoặc Markdown: {e}")
                    st.error(f"Đã xảy ra lỗi trong quá trình xử lý tài liệu. Vui lòng thử lại.")
                    st.error(f"Lỗi cụ thể: {e}")
                finally:
                    # Clean up the temporary PDF file
                    if temp_pdf_path and os.path.exists(temp_pdf_path):
                        try:
                            os.unlink(temp_pdf_path)
                            logger.info(f"Đã xóa tệp PDF tạm thời: {temp_pdf_path}")
                        except OSError as e:
                            logger.error(f"Lỗi khi xóa tệp PDF tạm thời {temp_pdf_path}: {e}")
                    # Note: The generated markdown file and directory are cleaned up in reset_chat

    # Display chat interface if the chain is ready
    if st.session_state.get('chain') is not None and st.session_state.get('vector_store_ready'):
        display_chat(st.session_state["chain"])
    elif uploaded_files and not st.session_state.get('vector_store_ready'):
         st.info("Đang chờ xử lý tệp PDF. Vui lòng nhấn 'Xử lý tệp PDF đã tải lên'.")
    else:
        st.info("Vui lòng tải lên tài liệu PDF và nhấn 'Xử lý tệp PDF đã tải lên' để bắt đầu.")

if __name__ == "__main__":
    main()
