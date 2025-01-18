import os
from typing import List
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import PreProcessor, BM25Retriever, FARMReader
import fitz  # PyMuPDF
import json

# Initialize an in-memory document store with BM25 enabled
document_store = InMemoryDocumentStore(use_bm25=True)

# Folder where uploaded documents are stored
UPLOAD_FOLDER = "uploaded_docs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def list_documents() -> List[str]:
    """List all uploaded documents."""
    try:
        return os.listdir(UPLOAD_FOLDER)
    except FileNotFoundError:
        raise RuntimeError(f"Upload folder '{UPLOAD_FOLDER}' not found. Please create it.")

def read_document(file_path: str) -> str:
    """Read the content of a document."""
    if file_path.endswith('.pdf'):
        return read_pdf(file_path)
    else:
        raise RuntimeError(f"Unsupported file type: '{file_path}'")

def read_pdf(file_path: str) -> str:
    """Extract text from a PDF file."""
    try:
        document = fitz.open(file_path)
        content = ""
        for page_num in range(document.page_count):
            page = document.load_page(page_num)
            content += page.get_text()
        return content
    except Exception as e:
        raise RuntimeError(f"Error reading PDF file '{file_path}': {str(e)}")

def preprocess_document(content: str) -> List[dict]:
    """Preprocess the document content into smaller chunks for indexing."""
    preprocessor = PreProcessor(
        split_by="word",
        split_length=200,
        split_overlap=20,
        split_respect_sentence_boundary=True,
    )
    documents = [{"content": content}]
    processed_documents = preprocessor.process(documents)
    return processed_documents

def index_document(file_name: str):
    """Read, preprocess, and index a document."""
    print("Starting indexing process...")
    
    file_path = os.path.join(UPLOAD_FOLDER, file_name)
    if not os.path.isfile(file_path):
        raise RuntimeError(f"File '{file_name}' not found in '{UPLOAD_FOLDER}'.")

    print(f"Reading document: {file_name}")
    content = read_document(file_path)
    print("Document read successfully.")

    print("Preprocessing document...")
    chunks = preprocess_document(content)
    print(f"Document preprocessing completed. Total chunks: {len(chunks)}")

    print("Formatting chunks for indexing...")
    # Prepare chunks in Haystack-compatible format
    formatted_chunks = [{"content": chunk.content} for chunk in chunks]
    document_store.write_documents(formatted_chunks)
    
    # Save indexed documents
    save_documents()

    print("Document indexed successfully.")

    return {
        "message": f"Document '{file_name}' indexed successfully.",
        "chunks_count": len(formatted_chunks),
    }

def search_documents(query: str):
    """Search indexed documents using a query."""
    retriever = BM25Retriever(document_store=document_store)
    reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=False)
    
    # Retrieve documents
    retrieved_docs = retriever.retrieve(query)
    if not retrieved_docs:
        return {"message": "No relevant documents found."}

    # Reader to predict answers from retrieved documents
    answers = reader.predict(query=query, documents=retrieved_docs, top_k=3)

    # Serialize the results to avoid unsupported types
    results = [
        {
            "answer": ans.answer,
            "score": ans.score,
            "context": ans.context,
            "document_id": ans.document_ids[0] if ans.document_ids else None,  # Update here
        }
        for ans in answers["answers"]
    ]

    return {"results": results}

def list_indexed_documents():
    """List all documents currently indexed in the document store."""
    docs = document_store.get_all_documents()
    print(f"Total indexed documents: {len(docs)}")
    for doc in docs:
        print(f"Document ID: {doc.id}, Content: {doc.content[:100]}...")  # Print the first 100 characters of the content
    return docs

def save_documents(file_path="app/data/documents.json"):
    """Save indexed documents to a JSON file."""
    all_docs = document_store.get_all_documents()
    with open(file_path, "w") as f:
        json.dump([doc.to_dict() for doc in all_docs], f)
    print(f"Documents saved to {file_path}")
    
def load_documents(file_path="app/data/documents.json"):
    """Load documents from a JSON file into the document store."""
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            docs = json.load(f)
        document_store.write_documents(docs)
        print(f"Loaded {len(docs)} documents from {file_path}")
    else:
        print(f"No saved documents found at {file_path}. Please index documents first.")
        

if __name__ == "__main__":
    # Load saved documents into the document store
    load_documents()
    
    # For testing purposes, list all indexed documents
    list_indexed_documents()


