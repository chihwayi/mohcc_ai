import os
from typing import List, Dict
import json
import torch
from transformers import (
    Pipeline,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    pipeline
)
from pdf2image import convert_from_path
import pytesseract
from PyPDF2 import PdfReader

class PDFQuestionAnswerGenerator:
    def __init__(self):
        # Initialize the question generation model using T5
        self.qg_tokenizer = AutoTokenizer.from_pretrained("t5-base")
        self.qg_model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
        
        # Initialize the question answering pipeline
        self.qa_pipeline = pipeline(
            "question-answering",
            model="deepset/roberta-base-squad2"
        )
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF using OCR if needed."""
        try:
            # First try direct text extraction
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            
            # If no text was extracted, use OCR
            if not text.strip():
                images = convert_from_path(pdf_path)
                text = ""
                for image in images:
                    text += pytesseract.image_to_string(image) + "\n"
            
            return text.strip()
        except Exception as e:
            print(f"Error processing {pdf_path}: {str(e)}")
            return ""

    def split_text_into_chunks(self, text: str, chunk_size: int = 1000) -> List[str]:
        """Split text into manageable chunks."""
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) > chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_length = len(word)
            else:
                current_chunk.append(word)
                current_length += len(word) + 1  # +1 for space
                
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        return chunks

    def generate_questions(self, text: str, num_questions: int = 5) -> List[Dict[str, str]]:
        """Generate questions and answers from text."""
        qa_pairs = []
        chunks = self.split_text_into_chunks(text)
        
        for chunk in chunks:
            # Prepare input for question generation
            input_text = f"generate question: {chunk}"
            inputs = self.qg_tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
            
            # Generate questions
            outputs = self.qg_model.generate(
                inputs.input_ids,
                max_length=64,
                num_return_sequences=min(3, num_questions),
                num_beams=4,
                no_repeat_ngram_size=2
            )
            
            questions = [self.qg_tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
            
            for question in questions:
                # Get answer for generated question
                answer = self.qa_pipeline(
                    question=question,
                    context=chunk
                )
                
                if answer['score'] > 0.7:  # Only keep high-confidence answers
                    qa_pairs.append({
                        "prompt": question,
                        "completion": answer['answer']
                    })
                
                if len(qa_pairs) >= num_questions:
                    return qa_pairs
                    
        return qa_pairs

    def process_pdfs(self, pdf_dir: str, output_file: str, questions_per_pdf: int = 5):
        """Process all PDFs in directory and save Q&A pairs to JSON."""
        all_qa_pairs = []
        
        for filename in os.listdir(pdf_dir):
            if filename.lower().endswith('.pdf'):
                pdf_path = os.path.join(pdf_dir, filename)
                print(f"Processing {filename}...")
                
                # Extract text from PDF
                text = self.extract_text_from_pdf(pdf_path)
                
                if text:
                    # Generate Q&A pairs
                    qa_pairs = self.generate_questions(text, questions_per_pdf)
                    all_qa_pairs.extend(qa_pairs)
        
        # Save to JSON file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_qa_pairs, f, indent=2, ensure_ascii=False)
        
        print(f"Generated {len(all_qa_pairs)} Q&A pairs saved to {output_file}")

def main():
    # Initialize the generator
    generator = PDFQuestionAnswerGenerator()
    
    # Set your input and output paths
    pdf_directory = "uploaded_docs"
    output_json = "app/data/qa_pairs.json"
    
    # Process PDFs and generate Q&A pairs
    generator.process_pdfs(pdf_directory, output_json, questions_per_pdf=5)

if __name__ == "__main__":
    main()