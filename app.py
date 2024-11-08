import streamlit as st
import openai
import os
from src.document_processor import DocumentProcessor
import tempfile
from dotenv import load_dotenv
from config import config 

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="Paper RAG",
    page_icon="📚",
    layout="wide"
)

# Initialize session state for document processor
if 'doc_processor' not in st.session_state:
    st.session_state.doc_processor = DocumentProcessor()

def save_uploaded_file(uploaded_file):
    """Save uploaded file to temp directory and return the path."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None

def main():
    # Sidebar
    with st.sidebar:
        st.title("📚 Paper RAG")
        st.markdown("---")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Upload PDF Documents",
            type=['pdf'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            process_button = st.button("Process Documents")
            if process_button:
                with st.spinner("Processing documents..."):
                    for uploaded_file in uploaded_files:
                        # Save uploaded file and get path
                        temp_path = save_uploaded_file(uploaded_file)
                        if temp_path:
                            # Process the document
                            metadata = st.session_state.doc_processor.process_document(temp_path)
                            if metadata:
                                st.success(f"Processed: {metadata.filename}")
                            else:
                                st.error(f"Failed to process: {uploaded_file.name}")
                            
                            # Clean up temp file
                            try:
                                os.unlink(temp_path)
                            except Exception as e:
                                st.warning(f"Error removing temporary file: {e}")
        
        # Show processed files
        st.markdown("### Processed Documents")
        processed_files = st.session_state.doc_processor.get_processed_files()
        if processed_files:
            for file in processed_files:
                st.text(f"📄 {file}")
        else:
            st.info("No documents processed yet")

    # Main content
    st.title("Query Your Documents")
    
    # Query input
    query = st.text_input("Enter your question:")
    
    if query:
        if not st.session_state.doc_processor.get_processed_files():
            st.warning("Please upload and process some documents first!")
        else:
            with st.spinner("Generating answer..."):
                answer = st.session_state.doc_processor.query_documents(query)
                
                # Display answer in a nice format
                st.markdown("### Answer")
                st.markdown(answer)
    
    # Instructions
    if not query and not st.session_state.doc_processor.get_processed_files():
        st.markdown("""
        ### Instructions
        1. Upload your PDF documents using the sidebar
        2. Click 'Process Documents' to analyze them
        3. Enter your question in the text box above
        4. Get AI-powered answers based on your documents
        
        ### Features
        - Process multiple PDF documents
        - Natural language querying
        - AI-powered document analysis
        - Interactive interface
        """)

if __name__ == "__main__":
    main()