import os
from dotenv import load_dotenv
from pinecone import Pinecone
from paperqa import Docs
from paperqa.llms import LiteLLMEmbeddingModel

load_dotenv()

async def query_pinecone(query: str, k: int = 4):
    """Query Pinecone and return relevant text chunks and generated answer."""
    # Initialize Pinecone and embedding model
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))
    embedding_model = LiteLLMEmbeddingModel()
    
    # Get query embedding
    query_embedding = (await embedding_model.embed_documents([query]))[0]
    
    # Query Pinecone
    results = index.query(
        vector=query_embedding,
        top_k=k,
        include_metadata=True
    )
    
    # Print results
    print("\nRetrieved documents:")
    for match in results.matches:
        print(f"\nDocument: {match.metadata['docname']}")
        print(f"Citation: {match.metadata['citation']}")
        print(f"Text preview: {match.metadata['text'][:200]}...")
        print(f"Score: {match.score}")
    
    # Create Docs object with retrieved results
    docs = Docs()
    docs.docs = {match.metadata['docname']: match.metadata['doc'] for match in results.matches}
    
    # Generate answer
    print("\nGenerating answer...")
    answer = await docs.aget_answer(query)
    print("\nAnswer:")
    print(answer)
    
    return results, answer

async def main():
    # Query example
    query = "How are CNNs used in deep learning for siRNA prediction?"
    print(f"Querying: {query}")
    
    # Get relevant documents from Pinecone
    _, _ = await query_pinecone(query)  # Ignore both results and answer since I am just printing

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())