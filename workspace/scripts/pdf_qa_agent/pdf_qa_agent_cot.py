import dspy
import PyPDF2
import argparse
import os
import logging 

# --- Logger Configuration ---
log_formatter = logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log_filename = 'pdf_qa_agent_cot.log' # Define the log file name

logger = logging.getLogger()
logger.setLevel(logging.INFO) # Set the default logging level for the root logger

try:
    file_handler = logging.FileHandler(log_filename, mode='a', encoding='utf-8') # 'a' for append mode
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
except Exception as e:
    # Fallback to console if file handler fails
    print(f"Warning: Could not set up log file '{log_filename}'. Logging to console only. Error: {e}")

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(log_formatter)
logger.addHandler(stream_handler)

# --- Configuration ---
OLLAMA_MODEL_NAME = 'ollama_chat/deepseek-r1:14b'
OLLAMA_API_BASE = 'http://localhost:11434'
# Context window size requested (16384 tokens).
# Ensure your Ollama setup and the model version can handle this.
# This might affect performance or memory usage.
REQUESTED_CONTEXT_SIZE = 1024

# --- DSPy Language Model Setup ---
logging.info(f"Configuring DSPy to use Ollama model: {OLLAMA_MODEL_NAME}")
logging.info(f"API Base: {OLLAMA_API_BASE}")
logging.info(f"Attempting to configure context window size (num_ctx): {REQUESTED_CONTEXT_SIZE}")

# Use the dspy.Ollama client specifically designed for Ollama
# The 'model' parameter is the tag Ollama uses for the model
llm = dspy.LM(model=OLLAMA_MODEL_NAME, api_base=OLLAMA_API_BASE)
dspy.configure(lm=llm, num_ctx=REQUESTED_CONTEXT_SIZE)
logging.info("DSPy configuration complete.")

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
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
        logging.info(f"Successfully saved extracted text to: {output_path}")
    except IOError as e:
        logging.error(f"Could not write extracted text to file '{output_path}': {e}")
    except Exception as e:
        logging.exception(f"An unexpected error occurred while saving text to '{output_path}'.")


# --- DSPy Signature Definition ---
class PdfQA(dspy.Signature):
    """Answer questions based *only* on the provided context from a PDF document.
    Reason step-by-step before providing the final answer."""
    
    context = dspy.InputField(desc="Relevant text extracted from specific pages of a PDF document.")
    question = dspy.InputField(desc="The question to answer based on the provided PDF context.")
    # ChainOfThought implicitly adds a 'rationale' or 'reasoning' step
    answer = dspy.OutputField(desc="A concise answer derived strictly from the provided context.")

# --- DSPy Module (Agent) ---
# Using ChainOfThought to encourage step-by-step reasoning before the answer.
pdf_qa_agent = dspy.ChainOfThought(PdfQA)

# --- Helper function to parse page numbers ---
def parse_page_numbers(pages_str: str) -> list[int] | None:
    """Parses a comma-separated string of pages/ranges into a list of 0-indexed integers."""
    pages = set()
    try:
        parts = pages_str.split(',')
        for part in parts:
            part = part.strip()
            if not part: continue
            if '-' in part:
                # Handle range like "3-5" (inclusive, 1-based) -> pages 2, 3, 4 (0-based)
                start_str, end_str = part.split('-', 1)
                start = int(start_str)
                end = int(end_str)
                if start <= 0 or end <= 0 or end < start:
                    logging.warning(f"Invalid page range '{part}'. Skipping.")
                    continue
                pages.update(range(start - 1, end)) # end is exclusive in Python range, so 'end' works directly
            else:
                # Handle single page like "1" -> page 0
                page_num = int(part)
                if page_num <= 0:
                    logging.warning(f"Invalid page number '{part}'. Page numbers must be positive. Skipping.")
                    continue
                pages.add(page_num - 1) # Convert to 0-indexed
        return sorted(list(pages))
    except ValueError:
        logging.error(f"Invalid page number format in '{pages_str}'. Use comma-separated numbers or ranges (e.g., '1,3,5-7').")
        return None
    except Exception as e:
        logging.exception(f"An unexpected error occurred parsing page numbers '{pages_str}'.")
        return None

# --- Main Execution Logic ---
def main():
    parser = argparse.ArgumentParser(description="Ask questions about specific pages of a PDF file using DSPy and Ollama.")
    parser.add_argument("pdf_path", help="Path to the PDF file.")
    parser.add_argument("-q", "--question", required=True, help="The question to ask about the PDF content.")
    parser.add_argument("-p", "--pages", required=True, help="Comma-separated page numbers or ranges (1-based index, e.g., '1', '1,2', '3-4').")
    parser.add_argument("-o", "--output-txt", help="Optional path to save the extracted text (e.g., 'extracted_content.txt').")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.") # Optional debug flag

    args = parser.parse_args()

    # Adjust log level if debug flag is set
    # Note: We get the root logger again here, which is fine.
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.debug("Debug logging enabled.") # This will now go to file and console

    # Parse page numbers (convert from 1-based user input to 0-based internal)
    page_numbers_0_indexed = parse_page_numbers(args.pages)
    if page_numbers_0_indexed is None:
        # Error already logged by parse_page_numbers
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

    # 1. Extract Context
    context = extract_text_from_pdf(args.pdf_path, page_numbers_0_indexed)

    if context.startswith("Error:"):
        # Error details already logged by extract_text_from_pdf or os.path.exists
        logging.error(f"Stopping due to error during text extraction: {context}")
        return

    if not context.strip():
        logging.warning("Extracted text content is empty. The specified pages might be blank or contain only images/scans without OCR.")
        # Decide if you want to proceed or stop. Currently proceeds.

    # Log context preview at DEBUG level to avoid cluttering INFO logs
    logging.debug(f"--- Context (Preview) ---\n{context[:500].strip()}...")

    # 2. Save Extracted Text (if requested)
    if args.output_txt:
        save_text_to_file(context, args.output_txt) # Errors logged within function

    # 3. Ask the LLM using the DSPy agent
    logging.info("Asking the LLM (using ChainOfThought)...")
    try:
        # Execute the ChainOfThought prediction
        result = pdf_qa_agent(context=context, question=args.question)
        
        print(result)

        # --- Log Rationale (Thinking Process) ---
        if hasattr(result, 'reasoning') and result.reasoning and result.reasoning.strip():
            logging.info("--- LLM Rationale (Thinking) ---")
            # Log rationale multi-line for readability
            for line in result.reasoning.strip().split('\n'):
                logging.info(f"Rationale: {line}")
        else:
            logging.info("--- No explicit rationale provided by LLM ---")

        # --- Log Final Answer ---
        logging.info("--- LLM Final Answer ---")
        # Log answer multi-line for readability
        for line in result.answer.strip().split('\n'):
            logging.info(f"Answer: {line}")

        # Optional: Inspect the full interaction history (might be verbose)
        logging.debug("--- LLM Interaction History (Debug) ---")
        try:
            # Use inspect_history(n=1) to see the last interaction
            history = llm.inspect_history(n=1) # Get prompt and response
            logging.debug(f"Last LLM Interaction:\n{history}")
        except Exception as e:
            logging.debug(f"Could not inspect history: {e}")

    except Exception as e:
        logging.exception("An error occurred during the LLM query.") # Log full traceback
        logging.error("Please ensure the Ollama server is running, the model is available, and the configuration is correct.")

if __name__ == "__main__":
    # Add a final log message indicating script completion or termination
    try:
        main()
        logging.info("Script execution completed.")
    except Exception as e:
        logging.exception("Script terminated due to an unhandled exception in main.")
        # Optionally re-raise or exit with error code
        # raise e
        # import sys
        # sys.exit(1)
    finally:
        # Ensure handlers are closed properly, especially file handlers
        logging.shutdown()