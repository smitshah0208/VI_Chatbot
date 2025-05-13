"""
vi_main.py: Streamlit application for chatting with a PDF document.
Processes PDF files into markdown, chunks the markdown based on image lines,
creates a vector store, and uses an Azure OpenAI model to answer questions
about the document content. Includes robust error handling and cross-environment
compatibility features.
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
import traceback # Import traceback for detailed error logging

# Configure logging to show more detailed information
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import PDF processing function from a separate file
try:
    # Assuming process_pdf.py and get_md_text function exist in the same directory or are importable
    from process_pdf import get_md_text
    logger.info("Successfully imported get_md_text function from process_pdf.py.")
except ImportError:
    logger.error("Could not import process_pdf.py or get_md_text function. Please ensure the file exists and is accessible.")

    # Define a dummy function to prevent errors if the import fails
    # This allows the rest of the application to run, albeit without PDF processing capability.
    def get_md_text(pdf_path, output_dir_base="md_output", poppler_path=None):
        logger.error("get_md_text function is not available due to import error.")
        st.error("Could not find the PDF processing module (process_pdf.py). Please ensure it's in the correct location.")
        return "" # Return empty string to indicate failure

# Load environment variables from .env file
load_dotenv()

# --- Configuration Constants ---
MARKDOWN_OUTPUT_FILENAME = "document.md"
MARKDOWN_OUTPUT_DIR = "md_output"
# Use os.path.join for cross-platform compatibility
MARKDOWN_OUTPUT_FILE_PATH = os.path.join(MARKDOWN_OUTPUT_DIR, MARKDOWN_OUTPUT_FILENAME)

# Multiple potential directories to try for the Chroma DB persistence
# This increases the chance of finding a writable location across different OS/environments
DB_DIRECTORIES = [
    "./md_chroma_db",                       # Current directory
    os.path.join(tempfile.gettempdir(), "md_chroma_db"), # Standard temp directory
    os.path.expanduser("~/md_chroma_db"),   # User's home directory
    "/tmp/md_chroma_db"                     # Linux/Unix specific temp directory
]

# Default Chroma DB directory path. This will be updated by find_writable_directory.
# Initialize with the first option as a starting point.
CHROMA_DB_DIR = DB_DIRECTORIES[0]

# Define the fixed Poppler path for Windows. Set to None for other platforms
# where poppler_utils should be in the system's PATH.
if sys.platform.startswith('win'):
    # IMPORTANT: Update this path to where poppler-*-win64/Library/bin is located on your system
    FIXED_POPPLER_PATH = r".\poppler-24.08.0\Library\bin"
    # Consider adding checks if this path exists in a real application
    if not os.path.exists(FIXED_POPPLER_PATH):
         logger.warning(f"Configured Poppler path not found: {FIXED_POPPLER_PATH}. PDF processing may fail.")
         st.warning(f"Đường dẫn Poppler đã cấu hình không tìm thấy: {FIXED_POPPLER_PATH}. Xử lý PDF có thể thất bại.")
else:
    FIXED_POPPLER_PATH = None  # Let system find it on Linux/Mac

# Regex to identify markdown image lines (e.g., ![alt text](path/to/image.png))
IMAGE_LINE_REGEX = re.compile(r'^!\[.*?\]\(.*?\)$')


# --- Helper Functions ---

def init_session_state():
    """
    Initializes Streamlit session state variables.
    Sets default values if keys are not already present.
    """
    # Chat history management
    st.session_state.setdefault('history', []) # List of (query, answer) tuples for Langchain
    st.session_state.setdefault('generated', []) # List of AI responses for UI display
    st.session_state.setdefault('past', [])      # List of user inputs for UI display

    # Application state flags and data
    st.session_state.setdefault('processed_pdf_name', None) # Name of the currently processed PDF
    st.session_state.setdefault('vector_store_ready', False) # Flag indicating if vector store is built
    st.session_state.setdefault('chain', None) # Langchain ConversationalRetrievalChain instance
    st.session_state.setdefault('vector_store', None) # Chroma vector store instance

    # Directory management
    # Use a default path initially, find_writable_directory will update this
    st.session_state.setdefault('current_db_path', CHROMA_DB_DIR)
    # Unique ID for temporary file/directory naming, useful for parallel runs or preventing conflicts
    st.session_state.setdefault('unique_session_id', str(uuid.uuid4()))

    logger.info("Đã khởi tạo trạng thái phiên.")
    # Add initial messages only on the first load
    if not st.session_state['past'] and not st.session_state['generated']:
         st.session_state['past'] = ["Xin chào! 👋 Hãy hỏi tôi bất cứ điều gì về tài liệu PDF đã tải lên 📄"]
         st.session_state['generated'] = ["Các phản hồi của bạn sẽ được hiển thị ở đây."]
         logger.info("Đã thêm tin nhắn chào mừng ban đầu.")


def safely_remove_path(path, is_dir=False):
    """
    Safely removes a file or directory with robust error handling.
    Includes logging and a fallback to renaming if deletion fails due to permissions or in-use issues.
    """
    if not os.path.exists(path):
        logger.debug(f"Đường dẫn không tồn tại, không cần xóa: {path}")
        return True # Path doesn't exist, nothing to remove, consider successful

    try:
        if is_dir:
            # Use shutil.rmtree for directories
            shutil.rmtree(path)
            logger.info(f"Đã xóa thư mục thành công: {path}")
        else:
            # Use os.remove for files
            os.remove(path)
            logger.info(f"Đã xóa tệp thành công: {path}")
        return True
    except (OSError, PermissionError) as e:
        logger.warning(f"Không thể xóa {'thư mục' if is_dir else 'tệp'} {path} do lỗi hệ thống/quyền: {e}", exc_info=True)

        # Fallback: Try renaming the path if deletion failed
        try:
            # Use a unique suffix derived from session ID and timestamp for clarity
            unique_suffix = f"{st.session_state.get('unique_session_id', str(uuid.uuid4()))[:6]}_{int(os.times().elapsed)}"
            new_path = f"{path}_old_{unique_suffix}"
            os.rename(path, new_path)
            logger.warning(f"Không thể xóa, đã đổi tên thành: {new_path}")
            # Note: The renamed file/directory still exists and might need manual cleanup later.
            return True # Renaming was successful, effectively moving the problematic path
        except Exception as rename_err:
            logger.error(f"Cả xóa và đổi tên đều thất bại cho {'thư mục' if is_dir else 'tệp'} {path}: {rename_err}", exc_info=True)
            st.error(f"Lỗi nghiêm trọng: Không thể xóa hoặc đổi tên {'thư mục' if is_dir else 'tệp'} cũ tại {path}. Vui lòng xóa thủ công để tiếp tục.")
            return False # Both attempts failed


def reset_chat():
    """
    Resets the chat history and application processing state.
    Cleans up temporary files and vector store directories.
    Explicitly sets the initial chat messages for display after reset.
    """
    logger.info("Đang đặt lại trò chuyện và trạng thái xử lý.")

    # --- Cleanup Files and Directories ---
    # Clean up the markdown output file and directory
    if os.path.exists(MARKDOWN_OUTPUT_DIR):
         safely_remove_path(MARKDOWN_OUTPUT_FILE_PATH, is_dir=False) # Try removing the file first
         safely_remove_path(MARKDOWN_OUTPUT_DIR, is_dir=True) # Then remove the directory

    # Clean up all potential DB directories where the vector store might have been created
    # It's safer to iterate through all candidates for cleanup
    logger.info("Đang cố gắng xóa các thư mục cơ sở dữ liệu Chroma tiềm năng.")
    for db_dir in DB_DIRECTORIES:
        if os.path.exists(db_dir):
            safely_remove_path(db_dir, is_dir=True)

    # --- Reset Session State ---
    st.session_state['history'] = [] # Langchain history
    # Explicitly set the UI display lists back to initial messages
    st.session_state['past'] = ["Xin chào! 👋 Hãy hỏi tôi bất cứ điều gì về tài liệu PDF đã tải lên 📄"]
    st.session_state['generated'] = ["Các phản hồi của bạn sẽ được hiển thị ở đây."]

    # Remove chain and vector store instances
    st.session_state.pop('chain', None)
    st.session_state.pop('vector_store', None)

    # Reset processing flags and info
    st.session_state['processed_pdf_name'] = None
    st.session_state['vector_store_ready'] = False

    # Generate a *new* unique session ID for the *next* processing cycle
    st.session_state['unique_session_id'] = str(uuid.uuid4())

    # Reset current DB path state to default, find_writable_directory will re-evaluate later
    st.session_state['current_db_path'] = CHROMA_DB_DIR

    logger.info("Chat và trạng thái xử lý đã được đặt lại.")
    # Streamlit automatically reruns when a button is clicked,
    # so the UI will update to show only the initial messages.


def conversation_chat(query, chain, history):
    """
    Handles the conversational chat interaction with the retrieval chain.
    Sends the user query and chat history to the chain and returns the response.
    Includes error handling for the chain invocation.
    """
    if not chain:
        logger.warning("Chuỗi hội thoại chưa được tạo hoặc không khả dụng.")
        return "Lỗi: Chuỗi hội thoại không khả dụng. Vui lòng xử lý tài liệu PDF trước."

    try:
        logger.info(f"Nhận câu hỏi người dùng: '{query}'")
        # The chain expects 'question' and 'chat_history' as input keys
        result = chain.invoke({"question": query, "chat_history": history})
        answer = result.get("answer", "Không có câu trả lời nào được tạo.").strip()
        logger.info(f"Đã tạo phản hồi thành công. Độ dài: {len(answer)}")

        # Append the interaction to the history for the next turn
        # History is used by Langchain's ConversationBufferMemory
        history.append((query, answer))

        return answer
    except Exception as e:
        logger.error(f"Lỗi trong quá trình hội thoại: {e}", exc_info=True)
        error_msg = f"Xin lỗi, tôi không thể xử lý yêu cầu này ngay bây giờ do lỗi kỹ thuật. Lỗi: {e}"
        # Still append to history to acknowledge the user's input, even if failed
        history.append((query, error_msg))
        st.error(error_msg) # Display error in the UI
        return error_msg


def display_chat(chain):
    """
    Displays the chat interface using streamlit_chat components.
    """
    # Use a form for the chat input to allow submitting with Enter key
    with st.form("chat_form", clear_on_submit=True):
        # Input field for user query
        user_input = st.text_input("Câu hỏi của bạn:", placeholder=f"Hãy hỏi về nội dung của tài liệu {st.session_state.get('processed_pdf_name', 'PDF')}", key='chat_input_text')
        # Submit button within the form
        submit_button = st.form_submit_button("Gửi")

    # Process the query if the form is submitted and there's input
    if submit_button and user_input:
        # Use st.spinner to indicate processing
        with st.spinner("Đang tạo phản hồi..."):
            # Call the conversation function
            output = conversation_chat(user_input, chain, st.session_state['history'])

            # Append user input and AI output to the lists used for UI display
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

        # Rerun the script to update the chat display immediately
        st.rerun()

    # Display the chat messages
    # Iterate in reverse to show the latest messages at the bottom
    if st.session_state['generated']:
        # Use a container to manage the scrollbar better for chat history
        chat_container = st.container()
        with chat_container:
            # Display messages from latest to oldest
            for i in range(len(st.session_state['generated']) -1, -1, -1):
                # Display user message if available for this turn
                if i < len(st.session_state["past"]):
                    message(st.session_state["past"][i], is_user=True, key=f"{i}_user")
                # Display AI message
                message(st.session_state["generated"][i], key=f"{i}_ai") # Using f-string keys ensures uniqueness


def create_chain(vector_store):
    """
    Creates and configures the Langchain Conversational Retrieval Chain
    using Azure OpenAI models and the provided vector store.
    Includes error handling for model initialization and chain creation.
    """
    if not vector_store:
        logger.error("Không thể tạo chuỗi hội thoại: Vector store không khả dụng.")
        st.error("Không thể tạo chuỗi hội thoại vì không có vector store.")
        return None

    try:
        logger.info("Đang khởi tạo AzureChatOpenAI (LLM)...")
        # Initialize the Language Model (LLM)
        llm = AzureChatOpenAI(
            deployment_name=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
            model_name=os.getenv("AZURE_OPENAI_CHAT_MODEL_NAME"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            openai_api_version=os.getenv("OPENAI_API_CHAT_MODEL_VERSION"),
            openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            openai_api_type="azure", # Ensure this is set for Azure
            temperature=0 # Keep temperature low for factual retrieval
        )
        logger.info("Đã khởi tạo AzureChatOpenAI thành công.")
    except Exception as e:
        logger.error(f"Lỗi khi khởi tạo AzureChatOpenAI: {e}", exc_info=True)
        st.error("Không thể khởi tạo mô hình LLM từ Azure OpenAI. Vui lòng kiểm tra biến môi trường hoặc cấu hình Azure.")
        st.error(f"Lỗi cụ thể: {e}")
        return None

    try:
        logger.info("Đang tạo ConversationBufferMemory và Retriever...")
        # Setup memory for chat history
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        # Setup retriever from the vector store
        # search_kwargs={"k": 3} means retrieve top 3 relevant documents
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        logger.info("Đã tạo ConversationBufferMemory và Retriever thành công.")

        logger.info("Đang tạo ConversationalRetrievalChain...")
        # Define the prompt template for the LLM
        # This template instructs the LLM on how to use the context
        template = """Sử dụng các đoạn văn bản sau từ tài liệu để trả lời câu hỏi. Trả lời **chỉ bằng tiếng Việt**.
Nếu bạn không biết câu trả lời, chỉ cần nói rằng bạn không biết, đừng cố tạo ra câu trả lời.
Hãy cố gắng trả lời một cách toàn diện nhất có thể dựa trên ngữ cảnh được cung cấp.

Ngữ cảnh:
{context}

Câu hỏi: {question}

Trả lời:"""
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])

        # Create the Conversational Retrieval Chain
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            combine_docs_chain_kwargs={'prompt': prompt}, # Pass the custom prompt here
            return_source_documents=False # Set to True if you want to return source documents
        )
        logger.info("Đã tạo ConversationalRetrievalChain thành công.")
        return chain
    except Exception as e:
        logger.error(f"Lỗi khi tạo chuỗi hội thoại: {e}", exc_info=True)
        st.error("Không thể tạo chuỗi hội thoại. Vui lòng kiểm tra cấu hình hoặc tài liệu PDF.")
        st.error(f"Lỗi cụ thể: {e}")
        return None


def process_markdown_file_lines(markdown_file_path: str):
    """
    Reads a markdown file and creates document chunks based on content
    following image lines. This custom chunking strategy is designed for
    markdown converted from PDFs where images might separate logical sections.
    Empty lines are ignored.
    """
    chunks = []
    current_chunk_lines = [] # Buffer to accumulate lines for the current chunk
    start_line_for_chunk = 0 # To track the starting line number for metadata

    # Check if the markdown file exists
    if not os.path.exists(markdown_file_path):
        logger.error(f"Lỗi: Tệp markdown không tìm thấy tại {markdown_file_path}")
        st.error(f"Lỗi: Tệp markdown không tìm thấy tại {markdown_file_path}. Quá trình chuyển đổi PDF sang Markdown có thể đã thất bại hoặc tệp không được lưu đúng cách.")
        return None # Return None to indicate failure

    try:
        logger.info(f"Đang đọc tệp markdown và tạo chunks dựa trên hình ảnh: {markdown_file_path}")
        with open(markdown_file_path, 'r', encoding='utf-8') as f:
            # Read the file line by line, keeping track of the line number
            for i, line in enumerate(f, 1): # Start line numbering from 1
                cleaned_line = line.strip()

                # Check if the cleaned line is an image line
                if IMAGE_LINE_REGEX.match(cleaned_line):
                    # If we encounter an image line, the content accumulated so far
                    # (if any and not just whitespace) forms a chunk.
                    if current_chunk_lines:
                        # Join buffered lines to form the chunk content
                        chunk_content = "\n".join(current_chunk_lines).strip()
                        if chunk_content: # Only create a chunk if there is actual content
                             # Create a Document object for the chunk with metadata
                            chunks.append(Document(page_content=chunk_content, metadata={"source": os.path.basename(markdown_file_path), "start_line": start_line_for_chunk}))
                            logger.debug(f"Đã tạo chunk từ dòng {start_line_for_chunk} đến {i-1}")

                    # Clear the buffer to start accumulating content for the next chunk
                    current_chunk_lines = []
                    # The start line for the next chunk will be the line *after* the image line
                    start_line_for_chunk = i + 1
                    # Do NOT add the image line itself to the current chunk buffer

                elif cleaned_line: # If it's not an image line and not an empty line
                    # Add the cleaned line to the current chunk buffer
                    # If the buffer was just cleared by an image, this line is the start of the next chunk
                    if not current_chunk_lines:
                        start_line_for_chunk = i
                    current_chunk_lines.append(cleaned_line)
                # If it's an empty line (after strip), just skip it.

        # After the loop, add any remaining content in the buffer as the last chunk
        if current_chunk_lines:
            chunk_content = "\n".join(current_chunk_lines).strip()
            if chunk_content:
                 chunks.append(Document(page_content=chunk_content, metadata={"source": os.path.basename(markdown_file_path), "start_line": start_line_for_chunk if not chunks else start_line_for_chunk}))
                 logger.debug(f"Đã tạo chunk cuối cùng từ dòng {start_line_for_chunk} đến {i if 'i' in locals() else 0}")


        logger.info(f"Đã tạo {len(chunks)} chunks dựa trên hình ảnh từ {markdown_file_path}")
        if not chunks:
            logger.warning("Không có chunks nào được tạo từ tệp markdown. Tệp có thể trống hoặc định dạng không đúng.")
            st.warning("Không có nội dung nào được trích xuất hoặc chunked từ tệp Markdown. Vui lòng kiểm tra nội dung hoặc định dạng tệp PDF gốc.")
        return chunks

    except Exception as e:
        logger.error(f"Lỗi khi đọc tệp markdown và tạo chunks từ {markdown_file_path}: {e}", exc_info=True)
        st.error(f"Không thể đọc tệp markdown hoặc tạo chunks: {markdown_file_path}. Vui lòng kiểm tra định dạng tệp hoặc quyền truy cập.")
        st.error(f"Lỗi cụ thể: {e}")
        return None


def find_writable_directory():
    """
    Iterates through a list of potential directory paths to find one that is
    writable. Creates the directory if it doesn't exist. Falls back to a
    unique subdirectory in the system's temporary directory if no preferred
    path is writable.
    Returns the path of the found writable directory, or None if creation
    of even the fallback directory fails.
    """
    logger.info("Đang tìm kiếm thư mục có thể ghi cho cơ sở dữ liệu Chroma.")
    for db_path in DB_DIRECTORIES:
        try:
            # Ensure the directory exists. exist_ok=True prevents error if dir already exists.
            os.makedirs(db_path, exist_ok=True)

            # Test write permissions by creating and deleting a temporary file
            test_file = os.path.join(db_path, f"test_write_{st.session_state['unique_session_id']}.txt")
            with open(test_file, 'w') as f:
                f.write("Testing write permissions")
            os.remove(test_file) # Clean up the test file

            logger.info(f"Tìm thấy và xác minh thư mục có thể ghi: {db_path}")
            return db_path # Found a usable directory

        except Exception as e:
            logger.warning(f"Không thể sử dụng thư mục {db_path} do lỗi: {e}. Thử đường dẫn tiếp theo.", exc_info=True)
            continue # Try the next directory in the list

    # If no directory from the predefined list was writable, fall back to a unique temp directory
    fallback_dir = os.path.join(tempfile.gettempdir(), f"md_chroma_db_{st.session_state['unique_session_id']}")
    logger.warning(f"Không tìm thấy thư mục có thể ghi từ danh sách. Thử sử dụng thư mục tạm thời duy nhất: {fallback_dir}")
    try:
        os.makedirs(fallback_dir, exist_ok=True)
        logger.info(f"Đã tạo và sẽ sử dụng thư mục tạm thời duy nhất: {fallback_dir}")
        return fallback_dir # Fallback directory is usable
    except Exception as e:
        logger.critical(f"Không thể tạo thư mục tạm thời ngay cả tại {fallback_dir}: {e}", exc_info=True)
        st.error("Lỗi nghiêm trọng: Không thể tìm hoặc tạo thư mục để lưu cơ sở dữ liệu. Vui lòng kiểm tra quyền truy cập hệ thống tệp.")
        st.error(f"Lỗi cụ thể: {e}")
        return None # Cannot find or create any writable directory


def create_vector_store_from_documents(documents: list[Document]):
    """
    Generates a Chroma vector store from a list of Document objects.
    Initializes Azure OpenAI Embeddings and handles finding a writable directory
    for persistence. Falls back to an in-memory store if persistence fails.
    Includes extensive error handling.
    """
    if not documents:
        st.error("Không có tài liệu nào được cung cấp để tạo vector store.")
        logger.warning("Không có tài liệu nào để tạo vector store.")
        return None

    # --- Initialize Embeddings ---
    try:
        logger.info("Đang khởi tạo AzureOpenAIEmbeddings...")
        # Initialize the Embedding Model
        embeddings = AzureOpenAIEmbeddings(
            deployment=os.getenv("OPENAI_ADA_EMBEDDING_EMBEDDING_DEPLOYMENT_NAME"), # Should be your Azure deployment name for embeddings
            model=os.getenv("OPENAI_ADA_EMBEDDING_MODEL_NAME"), # Should be the model name (e.g., text-embedding-ada-002)
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            openai_api_type="azure", # Ensure this is set for Azure
            chunk_size=100 # May need tuning based on model limits and performance
        )
        logger.info("Đã khởi tạo AzureOpenAIEmbeddings thành công.")
    except Exception as e:
        logger.error(f"Lỗi khi khởi tạo AzureOpenAIEmbeddings: {e}", exc_info=True)
        st.error("Không thể khởi tạo Embeddings từ Azure OpenAI. Vui lòng kiểm tra biến môi trường và cấu hình Azure cho mô hình Embedding.")
        st.error(f"Lỗi cụ thể: {e}")
        return None

    # --- Find Writable Directory ---
    db_path = find_writable_directory()

    # --- Create Vector Store ---
    # Always try persistence first if a path was found
    if db_path:
        try:
            logger.info(f"Đang tạo Chroma vector store với persistence tại {db_path}...")
            vector_store = Chroma.from_documents(
                documents=documents,
                embedding=embeddings,
                collection_name="markdown_image_chunk_collection", # A name for the collection within Chroma
                persist_directory=db_path # Directory to save the database
            )
            # Trigger persistence (optional, as from_documents with persist_directory usually does it)
            # vector_store.persist() # Call persist() if automatic saving is not guaranteed
            logger.info(f"Đã tạo thành công Chroma vector store tại: {db_path}")
            # Store the path that was actually used
            st.session_state['current_db_path'] = db_path
            return vector_store
        except Exception as e:
            logger.error(f"Lỗi khi tạo vector store với persistence tại {db_path}: {e}", exc_info=True)
            st.warning(f"Không thể lưu trữ vector store trên đĩa tại {db_path}. Thử sử dụng vector store trong bộ nhớ.")
            # Fall through to try in-memory store

    # If persistence failed or no writable path was found, try in-memory
    logger.warning("Không thể sử dụng vector store persistent. Thử tạo vector store trong bộ nhớ.")
    try:
        # Create in-memory vector store as a fallback
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings
            # No persist_directory means it's in-memory
        )
        logger.info("Đã tạo vector store trong bộ nhớ thành công.")
        # Indicate that an in-memory store is being used
        st.session_state['current_db_path'] = "bộ nhớ (trong RAM)"
        return vector_store
    except Exception as e2:
        logger.critical(f"Không thể tạo vector store ngay cả trong bộ nhớ: {e2}", exc_info=True)
        st.error("Không thể tạo vector store (cả persistent và in-memory). Vui lòng kiểm tra cấu hình Azure OpenAI hoặc tài liệu PDF.")
        st.error(f"Lỗi cụ thể: {e2}")
        return None


# --- Main Application Logic ---

def main():
    """
    Main function to run the Streamlit application for PDF Q&A.
    Sets up the UI, handles file uploads, processing, and chat interaction.
    """
    # Configure the Streamlit page
    st.set_page_config(page_title="Trò chuyện với PDF", page_icon="📚", layout="wide")
    st.title("📘 Trò chuyện với tài liệu PDF của bạn.")

    # Initialize session state variables on first run or after a full reset
    init_session_state()

    # --- Sidebar UI ---
    st.sidebar.header("Tải lên & Tùy chọn")

    # File uploader widget
    uploaded_file = st.sidebar.file_uploader("Tải lên tệp PDF", type="pdf", accept_multiple_files=False)

    # Determine Poppler path based on platform
    # get_md_text function needs this for PDF-to-image conversion components
    poppler_path = FIXED_POPPLER_PATH
    logger.info(f"Sử dụng đường dẫn Poppler: {poppler_path}")

    # Add a status indicator for the processing and vector store state
    processing_status = st.sidebar.empty() # Placeholder for dynamic status messages
    if st.session_state.get('vector_store_ready'):
         processing_status.success(f"PDF đã sẵn sàng: {st.session_state.get('processed_pdf_name', '')}")
         db_path_display = st.session_state.get('current_db_path', 'bộ nhớ (trong RAM)')
         st.sidebar.info(f"Cơ sở dữ liệu: {db_path_display}")
    elif uploaded_file:
         processing_status.warning("Đang chờ xử lý...")
    else:
         processing_status.info("")


    # Button to explicitly trigger processing (useful if auto-processing is off or fails)
    process_button = st.sidebar.button("🚀 Xử lý tệp PDF đã tải lên")

    # Button to reset the chat and state
    if st.sidebar.button("🔄 Đặt lại Trò chuyện"):
        # Call the reset function and then rerun the app
        reset_chat()
        st.rerun() # Force a rerun to clear the UI immediately

    # --- Main Content Area Logic ---

    # Determine if processing should happen
    # Process if a file is uploaded AND (the process button is clicked OR no file has been processed yet)
    # This logic prevents reprocessing the same file unless the button is clicked or state is reset
    if uploaded_file and (process_button or st.session_state.get('processed_pdf_name') != uploaded_file.name or not st.session_state.get('vector_store_ready')):

        # --- Processing Flow ---
        logger.info(f"Bắt đầu xử lý tệp mới: {uploaded_file.name}")
        # Reset state if processing a new file or forcing reprocessing
        # (reset_chat was already called by the button or will be called if processing starts)
        if st.session_state.get('processed_pdf_name') != uploaded_file.name or not st.session_state.get('vector_store_ready'):
             # If a new file is uploaded while a previous one was processed, reset first
             if st.session_state.get('processed_pdf_name') is not None:
                  logger.info("Đang xử lý tệp PDF mới. Đặt lại trạng thái trước đó.")
                  reset_chat() # Reset only if there was a previously processed file or state is not ready

             st.session_state['processed_pdf_name'] = uploaded_file.name # Store the name of the file being processed
             st.session_state['vector_store_ready'] = False # Set flag to False during processing

        with st.spinner(f"Đang xử lý tệp PDF: {uploaded_file.name}..."):
            temp_pdf_path = None # Initialize variable outside try block

            try:
                # Ensure the markdown output directory exists
                os.makedirs(MARKDOWN_OUTPUT_DIR, exist_ok=True)
                logger.info(f"Đã đảm bảo thư mục đầu ra markdown tồn tại: {MARKDOWN_OUTPUT_DIR}")

                # Save the uploaded PDF to a temporary file
                # Streamlit's file_uploader provides a BytesIO object, save it to a real file for tools like poppler
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    temp_pdf_path = tmp.name # Store the temporary file path
                logger.info(f"Đã lưu tệp PDF tạm thời tại: {temp_pdf_path}")

                # Convert PDF to markdown using the imported function
                if get_md_text: # Check if the import was successful
                    logger.info(f"Đang chuyển đổi PDF sang Markdown: {temp_pdf_path}")
                    # get_md_text function should write the markdown content to MARKDOWN_OUTPUT_FILE_PATH
                    markdown_content = get_md_text(temp_pdf_path, output_dir_base=MARKDOWN_OUTPUT_DIR, poppler_path=poppler_path)
                    logger.info(f"Quá trình chuyển đổi PDF sang Markdown hoàn tất. Kết quả dự kiến tại {MARKDOWN_OUTPUT_FILE_PATH}")

                    # Process the generated markdown file into document chunks
                    if os.path.exists(MARKDOWN_OUTPUT_FILE_PATH):
                         chunks = process_markdown_file_lines(MARKDOWN_OUTPUT_FILE_PATH)
                    else:
                         chunks = None
                         logger.error(f"Tệp markdown dự kiến không được tạo: {MARKDOWN_OUTPUT_FILE_PATH}")
                         st.error("Quá trình chuyển đổi PDF sang Markdown không tạo ra tệp output dự kiến.")


                    # Create vector store from the chunks
                    if chunks and len(chunks) > 0:
                        vector_store = create_vector_store_from_documents(chunks)

                        if vector_store:
                            # Store the vector store and create the chain
                            st.session_state['vector_store'] = vector_store
                            st.session_state['chain'] = create_chain(vector_store)

                            if st.session_state['chain']: # Ensure chain creation was successful
                                st.session_state['vector_store_ready'] = True # Mark processing as complete and successful
                                success_message = f"Tài liệu PDF '{uploaded_file.name}' đã được xử lý thành công! Bây giờ bạn có thể đặt câu hỏi."
                                st.success(success_message)
                                logger.info(success_message)
                                # Force a rerun to display the chat interface
                                st.rerun()
                            else:
                                 logger.error("Không thể tạo chuỗi hội thoại sau khi tạo vector store.")
                                 st.error("Không thể tạo chuỗi hội thoại. Vui lòng kiểm tra cấu hình Azure OpenAI.")
                        else:
                            logger.error("Không thể tạo vector store từ các chunks đã xử lý.")
                            st.error("Không thể tạo vector store từ các chunks.")
                    else:
                         if chunks is not None: # If chunks is an empty list but not None
                              st.error("Không có nội dung nào được trích xuất hoặc chunked từ tệp Markdown.")
                         # If chunks is None, an error message was likely shown by process_markdown_file_lines or get_md_text


                else: # If get_md_text import failed
                    st.error("Không thể xử lý PDF vì hàm xử lý không được tải.")

            except Exception as e:
                logger.error(f"Lỗi nghiêm trọng trong quá trình xử lý PDF hoặc Markdown: {e}", exc_info=True)
                st.error("Đã xảy ra lỗi nghiêm trọng trong quá trình xử lý tài liệu. Vui lòng thử lại.")
                # Display traceback in the UI for debugging (optional, maybe remove in production)
                st.exception(e)
                st.error(f"Lỗi cụ thể: {e}")

            finally:
                # Clean up the temporary PDF file regardless of success or failure
                if temp_pdf_path and os.path.exists(temp_pdf_path):
                    logger.info(f"Đang cố gắng xóa tệp PDF tạm thời: {temp_pdf_path}")
                    try:
                        os.unlink(temp_pdf_path) # Use os.unlink for reliability
                        logger.info(f"Đã xóa tệp PDF tạm thời: {temp_pdf_path}")
                    except OSError as e:
                        logger.warning(f"Lỗi khi xóa tệp PDF tạm thời {temp_pdf_path}: {e}")
                        # If deletion fails, try renaming as a fallback cleanup
                        safely_remove_path(temp_pdf_path, is_dir=False)


    # --- Display Chat Interface ---
    # Only display the chat interface if the vector store and chain are ready
    if st.session_state.get('chain') is not None and st.session_state.get('vector_store_ready'):
        display_chat(st.session_state["chain"])
    # Provide instructions based on the current state
    elif uploaded_file and not st.session_state.get('vector_store_ready'):
        st.info("Đang chờ xử lý tệp PDF. Vui lòng nhấn '🚀 Xử lý tệp PDF đã tải lên'.")
    elif not uploaded_file:
         st.info("Vui lòng tải lên tài liệu PDF và nhấn '🚀 Xử lý tệp PDF đã tải lên' để bắt đầu.")
    # If none of the above, perhaps an error occurred during processing, the error message will be displayed by st.error

# --- Script Entry Point ---
if __name__ == "__main__":
    main()