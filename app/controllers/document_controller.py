import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF
import os
import re
import spacy
import json

nltk.download('punkt', quiet=True)
nlp = spacy.load("en_core_web_sm")

# Load a pre-trained model for sentence embeddings
model = SentenceTransformer('all-MiniLM-L6-v2') 

def extract_text_from_pdf(pdf_path):
    document = fitz.open(pdf_path)
    text = ""
    try:
        for page_num in range(len(document)):
            page = document.load_page(page_num)
            text += page.get_text()
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return ""  # or you could return None or log the error
    finally:
        document.close()  # Ensure the document is closed even if an error occurs
    return text

def process_pdfs(directory_path):
    pdf_texts = {}
    for filename in os.listdir(directory_path):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(directory_path, filename)
            try:
                text = extract_text_from_pdf(pdf_path)
                if text:  # Only add to dictionary if text was successfully extracted
                    pdf_texts[os.path.splitext(filename)[0]] = text
            except Exception as e:
                print(f"Failed to process {filename}: {e}")
    return pdf_texts

def clean_text(text):
    # Remove headers, footers, and page numbers if they follow a certain pattern
    text = re.sub(r'ZIMBABWE HEALTH SECTOR HIV AND STI STRATEGY • \d{4}-\d{4}\s*a\n', '', text, flags=re.IGNORECASE)  # Example pattern for one document
    text = re.sub(r'Comprehensive National HIV Communications Strategy for Zimbabwe:\s*\d{4}-\d{4}\s*', '', text, flags=re.IGNORECASE)  # Example for another
    
    # Remove page numbers and other noise if present and identifiable
    text = re.sub(r'Page \d+\s*', '', text)  # Remove 'Page X' if found
    
    # Convert to lowercase for consistency
    text = text.lower()
    
    # Remove special characters or normalize them
    text = re.sub(r'[^\w\s]', '', text)  # Remove all special characters
    
    # Remove extra whitespaces
    text = ' '.join(text.split())
    
    return text

def chunk_text(text, chunk_size=500):  # 500 characters per chunk as an example
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += " " + sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def add_metadata(chunks, doc_name):
    metadata_chunks = []
    for i, chunk in enumerate(chunks, 1):
        metadata = {
            'document': doc_name,
            'section': f"Section_{i}",  # This is a placeholder. If your document has actual sections, parse them.
            'text': chunk
        }
        metadata_chunks.append(metadata)
    return metadata_chunks

def detect_questions(text):
    doc = nlp(text)
    questions = [sent.text for sent in doc.sents if any(token.pos_ == "AUX" for token in sent) or sent.text.endswith('?')]
    return questions

# Generate embeddings and save to JSONL
def generate_and_save_embeddings(potential_answers):
    embeddings_data = []
    for item in potential_answers:
        question_embedding = model.encode(item['question'])
        answer_embedding = model.encode(item['answer'])
        embeddings_data.append({
            'question': item['question'],
            'question_embedding': question_embedding.tolist(),
            'answer': item['answer'],
            'answer_embedding': answer_embedding.tolist(),
            'document': item['document'],
            'section': item['section']
        })
    
    # Ensure the directory exists
    os.makedirs('app/data/', exist_ok=True)
    
    # Writing embeddings to JSONL file
    with open('app/data/qna_embeddings.jsonl', 'w') as file:
        for item in embeddings_data:
            json.dump(item, file)
            file.write('\n')
    
    print("JSONL file with embeddings created: app/data/qna_embeddings.jsonl")

# Example usage:
directory = 'uploaded_docs'
pdf_contents = process_pdfs(directory)
clean_texts = {name: clean_text(text) for name, text in pdf_contents.items()}

# Now chunk each cleaned document
chunked_texts = {}
for doc_name, doc_text in clean_texts.items():
    chunks = chunk_text(doc_text)
    chunked_texts[doc_name] = chunks

# Add metadata to chunks
metadata_texts = {}
for doc_name, chunks in chunked_texts.items():
    metadata_texts[doc_name] = add_metadata(chunks, doc_name)

# Example usage
potential_questions = []
potential_answers = []
for doc_name, chunks in metadata_texts.items():
    for chunk in chunks:
        questions = detect_questions(chunk['text'])
        potential_questions.extend([{'text': q, 'document': doc_name, 'section': chunk['section']} for q in questions])
        
        # Here, we assume the text following a question in the same chunk might be an answer
        for q in questions:
            answer_start = chunk['text'].find(q) + len(q)
            potential_answers.append({
                'question': q,
                'answer': chunk['text'][answer_start:].strip().split('\n')[0],  # Very rough answer extraction
                'document': doc_name,
                'section': chunk['section']
            })

# Generate embeddings and save to JSONL
generate_and_save_embeddings(potential_answers)