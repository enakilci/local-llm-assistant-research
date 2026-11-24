import streamlit as st
import PyPDF2
import os
import logging
import tempfile
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import time
from typing import List, Optional

# Page configuration
st.set_page_config(
    page_title="PDF Q&A Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .stChat > div {
        padding-top: 2rem;
    }
    .stSelectbox > div > div > select {
        background-color: #f0f2f6;
    }
    .upload-text {
        text-align: center;
        padding: 2rem;
        border: 2px dashed #ccc;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Configuration
OLLAMA_MODEL_NAME = 'gpt-oss:20b'
OLLAMA_API_BASE = 'http://localhost:11434'
REQUESTED_CONTEXT_SIZE = 1024

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = None
if "pdf_filename" not in st.session_state:
    st.session_state.pdf_filename = None
if "total_pages" not in st.session_state:
    st.session_state.total_pages = 0
if "llm_initialized" not in st.session_state:
    st.session_state.llm_initialized = False
if "llm" not in st.session_state:
    st.session_state.llm = None

# Configure logging for Streamlit
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@st.cache_resource
def initialize_llm():
    """Initialize the LangChain LLM with caching"""
    try:
        llm = ChatOllama(
            model=OLLAMA_MODEL_NAME,
            base_url=OLLAMA_API_BASE,
            num_ctx=REQUESTED_CONTEXT_SIZE,
            temperature=0.1,
        )
        return llm, None
    except Exception as e:
        return None, str(e)

def extract_text_from_pdf(pdf_file, page_numbers: List[int]) -> tuple[str, int]:
    """
    Extract text from specified pages of a PDF file.
    Returns: (extracted_text, total_pages)
    """
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_file.read())
            tmp_path = tmp_file.name

        with open(tmp_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            num_total_pages = len(reader.pages)
            extracted_text = []

            valid_pages = []
            for page_num in page_numbers:
                if 0 <= page_num < num_total_pages:
                    valid_pages.append(page_num)

            if not valid_pages:
                return "Error: No valid page numbers specified or pages out of range.", num_total_pages

            valid_pages.sort()

            for page_num in valid_pages:
                page = reader.pages[page_num]
                page_text = page.extract_text()
                if page_text:
                    extracted_text.append(f"--- Page {page_num + 1} ---\n{page_text}")
                else:
                    extracted_text.append(f"--- Page {page_num + 1} (No text extracted) ---")

            full_text = "\n\n".join(extracted_text)
            
        # Clean up temporary file
        os.unlink(tmp_path)
        return full_text, num_total_pages

    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        return f"Error reading PDF file: {e}", 0

def parse_page_numbers(pages_str: str, max_pages: int) -> Optional[List[int]]:
    """Parse page number string and return 0-indexed list"""
    pages = set()
    try:
        parts = pages_str.split(',')
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if '-' in part:
                start_str, end_str = part.split('-', 1)
                start = int(start_str)
                end = int(end_str)
                if start <= 0 or end <= 0 or end < start or start > max_pages or end > max_pages:
                    st.warning(f"Invalid page range '{part}'. Skipping.")
                    continue
                pages.update(range(start - 1, end))
            else:
                page_num = int(part)
                if page_num <= 0 or page_num > max_pages:
                    st.warning(f"Invalid page number '{part}'. Must be between 1 and {max_pages}. Skipping.")
                    continue
                pages.add(page_num - 1)
        return sorted(list(pages))
    except ValueError:
        st.error(f"Invalid page number format in '{pages_str}'. Use comma-separated numbers or ranges (e.g., '1,3,5-7').")
        return None
    except Exception as e:
        st.error(f"Error parsing page numbers: {e}")
        return None

def get_pdf_qa_response(question: str, context: str) -> str:
    """Get response from the PDF Q&A model"""
    if not st.session_state.llm:
        return "Error: Language model not initialized. Please check your Ollama server."
    
    # Create prompt template
    pdf_qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert AI assistant that answers questions based strictly on provided context."),
        ("human", """
Answer the question based on the provided context. Think step-by-step and provide your reasoning BEFORE giving the final answer.

Context:
{context}

Question: {question}

Format your response exactly as follows:
REASONING:
[Think through this step-by-step, showing your analysis of the context]

FINAL ANSWER:
[Provide your concise final answer based on the reasoning above]
""")
    ])

    # Create chain
    pdf_qa_chain = (
        {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
        | pdf_qa_prompt
        | st.session_state.llm
        | StrOutputParser()
    )

    try:
        input_data = {
            "context": context,
            "question": question
        }
        response_text = pdf_qa_chain.invoke(input_data)
        return response_text
    except Exception as e:
        logger.error(f"Error getting LLM response: {e}")
        return f"Error getting response from model: {e}"

# Main app layout
st.title("📚 PDF Q&A Assistant")
st.markdown("Upload a PDF and ask questions about its content using AI!")

# Sidebar for configuration
with st.sidebar:
    st.header("🔧 Configuration")
    
    # Model status
    st.subheader("Model Status")
    if not st.session_state.llm_initialized:
        with st.spinner("Initializing language model..."):
            llm, error = initialize_llm()
            if llm:
                st.session_state.llm = llm
                st.session_state.llm_initialized = True
                st.success("✅ Model initialized successfully!")
            else:
                st.error(f"❌ Failed to initialize model: {error}")
                st.info("Please ensure Ollama server is running and the model is available.")
    else:
        st.success("✅ Model ready")
    
    st.markdown("---")
    
    # Model information
    st.subheader("Model Information")
    st.info(f"""
    **Model:** {OLLAMA_MODEL_NAME}  
    **API Base:** {OLLAMA_API_BASE}  
    **Context Size:** {REQUESTED_CONTEXT_SIZE}
    """)

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📄 PDF Upload")
    
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type="pdf",
        help="Drag and drop your PDF file here or click to browse"
    )
    
    if uploaded_file is not None:
        if st.session_state.pdf_filename != uploaded_file.name:
            # New file uploaded
            st.session_state.pdf_filename = uploaded_file.name
            
            # Extract text from all pages initially to get total page count
            with st.spinner("Processing PDF..."):
                _, total_pages = extract_text_from_pdf(uploaded_file, [0])
                st.session_state.total_pages = total_pages
                
            st.success(f"✅ PDF uploaded: **{uploaded_file.name}** ({total_pages} pages)")
        
        # Page selection
        st.subheader("Page Selection")
        page_selection_method = st.radio(
            "Select pages to analyze:",
            ["All pages", "Specific pages", "Page range"]
        )
        
        selected_pages = []
        if page_selection_method == "All pages":
            selected_pages = list(range(st.session_state.total_pages))
        elif page_selection_method == "Specific pages":
            page_input = st.text_input(
                "Enter page numbers (comma-separated, 1-based):",
                placeholder="e.g., 1, 3, 5",
                help=f"Enter page numbers between 1 and {st.session_state.total_pages}"
            )
            if page_input:
                parsed_pages = parse_page_numbers(page_input, st.session_state.total_pages)
                if parsed_pages:
                    selected_pages = parsed_pages
        else:  # Page range
            col_start, col_end = st.columns(2)
            with col_start:
                start_page = st.number_input(
                    "Start page", 
                    min_value=1, 
                    max_value=st.session_state.total_pages, 
                    value=1
                )
            with col_end:
                end_page = st.number_input(
                    "End page", 
                    min_value=start_page, 
                    max_value=st.session_state.total_pages, 
                    value=min(start_page + 4, st.session_state.total_pages)
                )
            selected_pages = list(range(start_page - 1, end_page))
        
        # Extract text button
        if selected_pages and st.button("📖 Extract Text from Selected Pages", type="primary"):
            with st.spinner("Extracting text from selected pages..."):
                extracted_text, _ = extract_text_from_pdf(uploaded_file, selected_pages)
                if not extracted_text.startswith("Error:"):
                    st.session_state.pdf_text = extracted_text
                    page_nums = [p + 1 for p in selected_pages]  # Convert to 1-based for display
                    st.success(f"✅ Text extracted from pages: {page_nums}")
                    
                    # Show preview
                    with st.expander("📋 Text Preview", expanded=False):
                        preview_text = extracted_text[:1000] + "..." if len(extracted_text) > 1000 else extracted_text
                        st.text_area("Extracted text preview:", preview_text, height=200, disabled=True)
                else:
                    st.error(extracted_text)

with col2:
    st.header("💬 Chat Interface")
    
    if st.session_state.pdf_text:
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask a question about the PDF content..."):
            if not st.session_state.llm_initialized:
                st.error("Please wait for the model to initialize before asking questions.")
            else:
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # Get AI response
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response = get_pdf_qa_response(prompt, st.session_state.pdf_text)
                    
                    st.markdown(response)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
            
    else:
        st.info("👆 Please upload a PDF and extract text to start chatting!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <small>PDF Q&A Assistant powered by LangChain and Ollama</small>
</div>
""", unsafe_allow_html=True)