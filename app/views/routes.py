from fastapi import APIRouter, File, UploadFile, HTTPException
from pydantic import BaseModel
from app.services.document_service import list_documents, index_document, search_documents
from app.services.qa_service import get_answer
import os

router = APIRouter()

# Folder to store uploaded documents
UPLOAD_FOLDER = "uploaded_docs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

class FileName(BaseModel):
    file_name: str

@router.get("/")
async def root():
    return {"message": "Welcome to MOHCC AI Tools"}

@router.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())
    return {"message": f"File '{file.filename}' uploaded successfully.", "path": file_path}

@router.get("/files/")
async def list_files():
    files = os.listdir(UPLOAD_FOLDER)
    return {"uploaded_files": files}

@router.get("/list-documents/")
async def get_uploaded_documents():
    """List all uploaded documents."""
    documents = list_documents()
    return {"documents": documents}

@router.post("/index-document/")
async def index_uploaded_document(file: FileName):
    """Index an uploaded document."""
    documents = list_documents()
    if file.file_name not in documents:
        raise HTTPException(status_code=404, detail="Document not found.")
    return index_document(file.file_name)

@router.get("/search/")
async def search_query(query: str):
    """Search indexed documents."""
    results = search_documents(query)
    return {"results": results}

@router.post("/index-all/")
async def index_all_documents():
    """Index all uploaded documents in the upload folder."""
    documents = list_documents()
    if not documents:
        return {"message": "No documents found in the upload folder."}

    indexed_files = []
    skipped_files = []

    for file_name in documents:
        try:
            index_document(file_name)
            indexed_files.append(file_name)
        except Exception as e:
            print(f"Error indexing document '{file_name}': {str(e)}")
            skipped_files.append(file_name)

    return {
        "message": "Indexing completed.",
        "indexed_files": indexed_files,
        "skipped_files": skipped_files,
    }

@router.get("/ask/")
async def ask_question(query: str):
    """Ask a question and get an answer from the indexed documents."""
    documents = search_documents(query)
    if not documents:
        raise HTTPException(status_code=404, detail="No relevant documents found.")
    
    context = documents[0].content  # Using the most relevant document as context
    answer = get_answer(query, context)
    return {"question": query, "answer": answer}
