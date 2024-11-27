from typing import List, Optional, Dict
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain.docstore.document import Document as LangchainDocument
from paperqa import Doc, Text, Docs
from dotenv import load_dotenv
import warnings
import os

class DocumentQuerier:
    def __init__(self, pinecone_api_key: str, pinecone_environment: str, index_name: str):
        self.embeddings = OpenAIEmbeddings()
        self.docs = Docs()
        self.llm = ChatOpenAI(temperature=0)
        
        self.index = LangchainPinecone.from_existing_index(
            index_name=index_name,
            embedding=self.embeddings
        )
    
    def get_summary_and_score(self, text: str, citation: str, question: str) -> Dict:
        """Get summary and relevance score for a chunk of text."""
        print(f"\nGenerating summary for chunk...")
        print(f"Text preview: {text[:200]}...")
        
        prompt = (
            "Summarize the excerpt below to help answer a question.\n\n"
            f"Excerpt from {citation}\n\n----\n\n{text}\n\n----\n\n"
            f"Question: {question}\n\n"
            "Do not directly answer the question, instead summarize to give evidence "
            "to help answer the question. Stay detailed; report specific numbers, "
            'equations, or direct quotes (marked with quotation marks). '
            'Reply "Not applicable" if the excerpt is irrelevant. '
            "At the end of your response, provide an integer score from 1-10 on a "
            "newline indicating relevance to question. Do not explain your score."
        )
        
        response = self.llm.invoke(prompt).content
        
        print(f"Summary response: {response}")
        
        # Split response into summary and score
        parts = response.strip().split('\n')
        try:
            score = int(parts[-1])
            summary = '\n'.join(parts[:-1])
        except:
            score = 0
            summary = "Not applicable"
            
        return {"summary": summary, "score": score}

    def convert_to_paperqa_text(self, langchain_doc: LangchainDocument, 
                              summary_info: Dict) -> Text:
        """Convert LangChain Document to paper-qa Text object with context."""
        doc = Doc(
            docname=langchain_doc.metadata.get('doc_name', 'Unknown'),
            citation=f"Source: {langchain_doc.metadata.get('doc_name', 'Unknown')} - {langchain_doc.metadata.get('citation', 'Unknown')}",
            dockey=langchain_doc.metadata.get('doc_id', 'Unknown')
        )
        
        print("\nCreating Text object:")
        print(f"Doc name: {doc.docname}")
        print(f"Citation: {doc.citation}")
        print(f"Summary: {summary_info['summary'][:200]}...")
        
        text = Text(
            text=langchain_doc.page_content,  # Use full text instead of summary
            name=doc.docname,
            doc=doc
        )
        
        return text

    def query_documents(self, question: str, k: int = 10) -> str:
        """Query the documents with a specific question."""
        try:
            # Clear existing texts
            self.docs.texts = []
            
            print("\nRetrieving relevant chunks...")
            results = self.index.similarity_search(
                query=question,
                k=k
            )
            
            print(f"\nFound {len(results)} relevant chunks")
            print("\nExamining chunks:")
            for i, doc in enumerate(results):
                print(f"\nChunk {i+1}:")
                print(f"Content preview: {doc.page_content[:200]}...")
            
            # Process chunks with summaries
            processed_chunks = []
            for doc in results:
                summary_info = self.get_summary_and_score(
                    doc.page_content,
                    doc.metadata.get('citation', 'Unknown'),
                    question
                )
                
                if summary_info['score'] > 3:  # Only keep relevant chunks
                    text = self.convert_to_paperqa_text(doc, summary_info)
                    processed_chunks.append((text, summary_info['score']))
            
            processed_chunks.sort(key=lambda x: x[1], reverse=True)
            
            print(f"\nProcessing {len(processed_chunks)} relevant chunks...")
            
            if not processed_chunks:
                return "No relevant information found to answer the question."
            
            # Add to docs with context
            for text, score in processed_chunks:
                self.docs.texts.append(text)
            
            # Debug print the context paper-qa will use
            print("\nContext being sent to paper-qa:")
            for text in self.docs.texts:
                print(f"\nDocument: {text.name}")
                print(f"Citation: {text.doc.citation}")
                print(f"Text preview: {text.text[:200]}...")
            
            # Query using paper-qa
            print("\nGenerating final answer...")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                answer = self.docs.query(question)  
                
            return answer.formatted_answer
            
        except Exception as e:
            print(f"Error querying documents: {str(e)}")
            print(f"Error type: {type(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return "Sorry, I encountered an error while processing your query."

# In the main section:
if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    env_vars = {
        'PINECONE_API_KEY': os.getenv('PINECONE_API_KEY'),
        'PINECONE_ENVIRONMENT': os.getenv('PINECONE_ENVIRONMENT'),
        'PINECONE_INDEX_NAME': os.getenv('PINECONE_INDEX_NAME')
    }
    print("Environment variables loaded successfully")
    
    # Initialize querier
    querier = DocumentQuerier(
        pinecone_api_key=env_vars['PINECONE_API_KEY'],
        pinecone_environment=env_vars['PINECONE_ENVIRONMENT'],
        index_name=env_vars['PINECONE_INDEX_NAME']
    )
    print("DocumentQuerier initialized")
    
    # Example query with increased chunk retrieval
    question = "What methods are used for siRNA efficiency prediction?"
    print(f"\nQuestion: {question}")
    
    answer = querier.query_documents(question, k=10)  # Increased k to 10
    print("\nFinal Answer:")
    print(answer)