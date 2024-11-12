# PAPER_QA RAG

A RAG (Retrieval-Augmented Generation) application for processing and querying research papers.

## Installation

```bash
# Clone the repository
git clone https://github.com/ahmurill0217/paper_qa.git
cd PAPER_QA

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install the requirements
pip install -r requirements.txt
```

## Usage

### 1. Environment Setup
Create a `.env` file with your OpenAI API key:
```
OPENAI_API_KEY=your_openai_api_key_here
```

### 2. Choose Your Interface

#### Command Line Interface
1. Add your PDF files to the `data/raw` directory
2. Run the application:
```bash
python -m src.main
```
3. Follow the interactive prompts to query your documents

#### Streamlit Web Interface
1. Run the Streamlit app:
```bash
streamlit run app.py
```
2. Use the web interface to:
   - Upload PDF documents directly through the browser
   - Process documents with a single click
   - Query your documents using the interactive interface
   - View processing status and results in real-time

## Project Structure
```
PAPER_QA/
├── data/
│   ├── processed/
│   └── RAW/
├── src/
│   ├── __init__.py
│   ├── document_processor.py
│   └── main.py          # CLI interface
├── tests/
│   ├── __init__.py
│   └── test_document_processor.py
├── .env
├── .gitignore
├── .python-version      # Specifies Python 3.11.8 for pyenv
├── app.py              # Streamlit interface
├── config.py           # Configuration settings
├── README.md
└── requirements.txt
```

## Features

- Process multiple PDF documents
- Query documents using natural language
- Interactive question-answering interface
- LLM-powered document analysis
- Two interface options:
  - Command-line interface for direct file processing
  - Web interface for interactive document management

## Notes

- The application requires an OpenAI API key to function
- For CLI usage: Place your research papers in PDF format in the `data/raw` directory
- For Web usage: Upload PDFs directly through the Streamlit interface
- The application will process all provided PDF files