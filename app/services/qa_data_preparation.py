import os
import json
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import BM25Retriever

# Initialize document store
document_store = InMemoryDocumentStore(use_bm25=True)

def load_documents(file_path="app/data/documents.json"):
    """Load documents from a JSON file into the document store."""
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            documents = json.load(f)
        document_store.write_documents(documents)
        print(f"Documents loaded from {file_path}")
    else:
        print(f"No saved documents found at {file_path}")

def prepare_squad_data(output_path="app/data/squad_data.json"):
    # Fetch all documents from the document store
    all_docs = document_store.get_all_documents()
    
    squad_data = {"data": []}
    for doc in all_docs:
        if doc.content:  # Ensure the document has content
            paragraphs = [{"context": doc.content, "qas": []}]
            squad_data["data"].append({"title": doc.meta.get("name", "Document"), "paragraphs": paragraphs})
    
    # Save the SQuAD data to a file
    with open(output_path, "w") as f:
        json.dump(squad_data, f, indent=2)

    print(f"SQuAD data prepared and saved to {output_path}")
    print(f"Total documents processed: {len(all_docs)}")
    print(f"Total documents added to SQuAD data: {len(squad_data['data'])}")

if __name__ == "__main__":
    load_documents()  # Ensure documents are loaded
    prepare_squad_data()
