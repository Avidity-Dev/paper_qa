import os
from typing import List
from dotenv import load_dotenv
from document_processor import DocumentProcessor
import warnings

def load_environment_variables(env_file: str = '.env') -> None:
    """
    Load environment variables from a specified .env file.
    
    Args:
        env_file (str): Path to the .env file
    """
    load_dotenv(env_file)

def get_pdf_files(directory: str) -> List[str]:
    """
    Retrieve a list of PDF file paths from the specified directory.
    
    Args:
        directory (str): Directory path to search for PDFs
        
    Returns:
        List[str]: List of PDF file paths
    """
    return [
        os.path.join(directory, filename)
        for filename in os.listdir(directory)
        if filename.lower().endswith('.pdf')
    ]

def main() -> None:
    """Main function to orchestrate the document processing and querying."""
    # Suppress all warnings
    warnings.filterwarnings('ignore')
    
    # Load environment variables
    load_environment_variables()
    
    # Initialize the document processor
    processor = DocumentProcessor()
    
    # Get the project root directory (one level up from src)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Specify the directory containing PDF files
    papers_directory = os.path.join(project_root, 'data', 'RAW')
    
    print(f"Looking for PDFs in: {papers_directory}")
    
    # Check if the directory exists and has PDF files
    if not os.path.exists(papers_directory):
        print(f"Creating directory: {papers_directory}")
        os.makedirs(papers_directory)
        print("Please add your PDF files to the 'data/RAW' directory and run the script again.")
        return
    
    # Retrieve PDF files from the directory
    pdf_files = get_pdf_files(papers_directory)
    
    if not pdf_files:
        print("No PDF files found in the 'data/RAW' directory.")
        print("Please add your PDF files and run the script again.")
        return
    
    # Process each document
    print("\nProcessing documents...")
    for filepath in pdf_files:
        metadata = processor.process_document(filepath)
        if metadata:
            print(f"Processed: {metadata.filename}")
    
    print("\nAll documents processed successfully!")
    
    # Interactive query loop
    print("\nEnter your questions (or 'quit' to exit):")
    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() in ['quit', 'exit', 'q']:
            break
        
        if question:
            answer = processor.query_documents(question)
            print(f"\nAnswer: {answer}")

if __name__ == '__main__':
    main()