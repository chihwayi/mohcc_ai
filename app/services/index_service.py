import os
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import BM25Retriever, FARMReader
import fitz  # PyMuPDF

# Initialize an in-memory document store
document_store = InMemoryDocumentStore(use_bm25=True)
retriever = BM25Retriever(document_store=document_store)

def index_document(file_path: str):
    """Index a document for retrieval."""
    # Read the document
    with fitz.open(file_path) as doc:
        text = ""
        for page in doc:
            text += page.get_text()
    
    # Create a document object
    document = {"content": text, "meta": {"name": os.path.basename(file_path)}}
    
    # Write document to the store
    document_store.write_documents([document])
    retriever = BM25Retriever(document_store=document_store)  # Update retriever after writing
    return document

def list_documents():
    """List all indexed documents."""
    return document_store.get_all_documents()

def search_documents(query: str):
    """Search for documents matching the query."""
    return retriever.retrieve(query)
