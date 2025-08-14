import PyPDF2
import argparse
import os
import logging 
from langchain_ollama import ChatOllama  # Updated import
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- Logger Configuration ---
log_formatter = logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log_filename = 'pdf_qa_agent_cot.log'

logger = logging.getLogger()
logger.setLevel(logging.INFO)

try:
    file_handler = logging.FileHandler(log_filename, mode='a', encoding='utf-8')
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
except Exception as e:
    print(f"Warning: Could not set up log file '{log_filename}'. Logging to console only. Error: {e}")

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(log_formatter)
logger.addHandler(stream_handler)

# --- Configuration ---
OLLAMA_MODEL_NAME = 'gpt-oss:20b'
OLLAMA_API_BASE = 'http://localhost:11434'
REQUESTED_CONTEXT_SIZE = 1024

# --- LangChain Language Model Setup ---
logging.info(f"Configuring LangChain to use Ollama model: {OLLAMA_MODEL_NAME}")
logging.info(f"API Base: {OLLAMA_API_BASE}")
logging.info(f"Attempting to configure context window size (num_ctx): {REQUESTED_CONTEXT_SIZE}")

llm = ChatOllama(
    model=OLLAMA_MODEL_NAME,
    base_url=OLLAMA_API_BASE,
    num_ctx=REQUESTED_CONTEXT_SIZE,
    temperature=0.1,
)
logging.info("LangChain configuration complete.")

# --- PDF Text Extraction ---
def extract_text_from_pdf(pdf_path: str, page_numbers: list[int]) -> str:
    """
    Extracts text from specified pages (0-indexed) of a PDF file.

    Args:
        pdf_path: Path to the PDF file.
        page_numbers: A list of 0-indexed page numbers to extract text from.

    Returns:
        A string containing the concatenated text from the specified pages,
        or an error message if extraction fails.
    """
    if not os.path.exists(pdf_path):
        return f"Error: PDF file not found at {pdf_path}"

    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            num_total_pages = len(reader.pages)
            extracted_text = []

            valid_pages = []
            for page_num in page_numbers:
                if 0 <= page_num < num_total_pages:
                    valid_pages.append(page_num)
                else:
                    logging.warning(f"Page number {page_num + 1} is out of range (1-{num_total_pages}). Skipping.")

            if not valid_pages:
                return "Error: No valid page numbers specified or pages out of range."

            valid_pages.sort()

            logging.info(f"Extracting text from pages (0-indexed): {valid_pages}")
            for page_num in valid_pages:
                page = reader.pages[page_num]
                page_text = page.extract_text()
                if page_text:
                    extracted_text.append(f"--- Page {page_num + 1} ---\n{page_text}")
                else:
                    logging.debug(f"Page {page_num + 1} yielded no text.")
                    extracted_text.append(f"--- Page {page_num + 1} (No text extracted) ---")

            full_text = "\n\n".join(extracted_text) 
            logging.info(f"Successfully extracted ~{len(full_text)} characters.")
            return full_text

    except FileNotFoundError:
        return f"Error: PDF file not found at {pdf_path}"
    except PyPDF2.errors.PdfReadError as e:
        logging.error(f"Error reading PDF file '{pdf_path}': {e}. The file might be corrupted or encrypted.")
        return f"Error reading PDF file: {e}. The file might be corrupted or encrypted."
    except Exception as e:
        logging.exception(f"An unexpected error occurred during PDF processing for '{pdf_path}'.")
        return f"An unexpected error occurred during PDF processing: {e}"

# --- Function to save extracted text ---
def save_text_to_file(text: str, output_path: str):
    """Saves the given text content to a file."""
    try:
        os.makedirs("context", exist_ok=True)
        output_path_full = os.path.join("context", output_path)
        with open(output_path_full, 'w', encoding='utf-8') as f:
            f.write(text)
        logging.info(f"Successfully saved extracted text to: {output_path_full}")
    except IOError as e:
        logging.error(f"Could not write extracted text to file '{output_path_full}': {e}")
    except Exception as e:
        logging.exception(f"An unexpected error occurred while saving text to '{output_path_full}'.")

# --- LangChain Prompt Template ---
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

# --- Create chain ---
pdf_qa_chain = (
    {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
    | pdf_qa_prompt
    | llm
    | StrOutputParser()
)

# --- Helper function to parse page numbers ---
def parse_page_numbers(pages_str: str) -> list[int] | None:
    pages = set()
    try:
        parts = pages_str.split(',')
        for part in parts:
            part = part.strip()
            if not part: continue
            if '-' in part:
                start_str, end_str = part.split('-', 1)
                start = int(start_str)
                end = int(end_str)
                if start <= 0 or end <= 0 or end < start:
                    logging.warning(f"Invalid page range '{part}'. Skipping.")
                    continue
                pages.update(range(start - 1, end))
            else:
                page_num = int(part)
                if page_num <= 0:
                    logging.warning(f"Invalid page number '{part}'. Page numbers must be positive. Skipping.")
                    continue
                pages.add(page_num - 1)
        return sorted(list(pages))
    except ValueError:
        logging.error(f"Invalid page number format in '{pages_str}'. Use comma-separated numbers or ranges (e.g., '1,3,5-7').")
        return None
    except Exception as e:
        logging.exception(f"An unexpected error occurred parsing page numbers '{pages_str}'.")
        return None

# --- Main Execution Logic ---
def main():
    parser = argparse.ArgumentParser(description="Ask questions about specific pages of a PDF file using LangChain and Ollama.")
    parser.add_argument("pdf_path", help="Path to the PDF file.")
    parser.add_argument("-q", "--question", required=True, help="The question to ask about the PDF content.")
    parser.add_argument("-p", "--pages", required=True, help="Comma-separated page numbers or ranges (1-based index, e.g., '1', '1,2', '3-4').")
    parser.add_argument("-o", "--output-txt", help="Optional path to save the extracted text (e.g., 'extracted_content.txt').")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.debug("Debug logging enabled.")

    page_numbers_0_indexed = parse_page_numbers(args.pages)
    if page_numbers_0_indexed is None:
        return

    if not page_numbers_0_indexed:
        logging.error("No valid page numbers provided after parsing.")
        return

    logging.info("--- Starting PDF Q&A ---")
    logging.info(f"File: {args.pdf_path}")
    logging.info(f"Question: {args.question}")
    logging.info(f"Target Pages (1-based): {args.pages} -> (0-based): {page_numbers_0_indexed}")
    if args.output_txt:
        logging.info(f"Will save extracted text to: {args.output_txt}")

    context = extract_text_from_pdf(args.pdf_path, page_numbers_0_indexed)

    if context.startswith("Error:"):
        logging.error(f"Stopping due to error during text extraction: {context}")
        return

    if not context.strip():
        logging.warning("Extracted text content is empty. The specified pages might be blank or contain only images/scans without OCR.")

    logging.debug(f"--- Context (Preview) ---\n{context[:500].strip()}...")

    if args.output_txt:
        save_text_to_file(context, args.output_txt)

    logging.info("Asking the LLM using LangChain...")
    try:
        # Prepare input for the chain
        input_data = {
            "context": context,
            "question": args.question
        }
        
        # Execute the chain
        response_text = pdf_qa_chain.invoke(input_data)
        
        logging.debug(f"--- Model Output ---\n{response_text}")
        
        print("=" * 60)
        print("LLM RESPONSE:")
        print("=" * 60)
        print(response_text)
        print("=" * 60)
        
        # Extract and log reasoning and final answer sections
        lines = response_text.strip().split('\n')
        reasoning_lines = []
        answer_lines = []
        current_section = None
        
        for line in lines:
            line_stripped = line.strip()
            if line_stripped.upper().startswith('REASONING:'):
                current_section = 'reasoning'
                # Add the rest of the line after "REASONING:" if it exists
                if len(line_stripped) > 10:
                    reasoning_lines.append(line_stripped[10:].strip())
                continue
            elif line_stripped.upper().startswith('FINAL ANSWER:') or line_stripped.upper().startswith('ANSWER:'):
                current_section = 'answer'
                # Add the rest of the line after the header if it exists
                colon_pos = line_stripped.find(':')
                if colon_pos != -1 and len(line_stripped) > colon_pos + 1:
                    answer_lines.append(line_stripped[colon_pos + 1:].strip())
                continue
            
            if current_section == 'reasoning':
                reasoning_lines.append(line)
            elif current_section == 'answer':
                answer_lines.append(line)
        
        # Log reasoning section
        if reasoning_lines:
            logging.info("--- LLM Reasoning ---")
            for line in reasoning_lines:
                if line.strip():
                    logging.info(f"{line}")
        
        # Log answer section
        if answer_lines:
            logging.info("--- LLM Final Answer ---")
            for line in answer_lines:
                if line.strip():
                    logging.info(f"{line}")
        else:
            # If we couldn't parse sections, log the full response under answer
            logging.info("--- LLM Response ---")
            for line in lines:
                logging.info(f"{line}")

    except Exception as e:
        logging.exception("An error occurred during the LLM query.")
        logging.error("Please ensure the Ollama server is running, the model is available, and the configuration is correct.")

if __name__ == "__main__":
    try:
        main()
        logging.info("Script execution completed.")
    except Exception as e:
        logging.exception("Script terminated due to an unhandled exception in main.")
    finally:
        logging.shutdown()