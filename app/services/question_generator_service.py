import os
import json
import logging
import spacy
import re
from haystack.nodes import QuestionGenerator
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load spaCy model for NER
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.error("spaCy model not found. Please install en_core_web_sm by running: python -m spacy download en_core_web_sm")
    raise

def extract_answer(context, question):
    """
    Advanced answer extraction method with multiple strategies.
    
    Args:
        context (str): The full text context
        question (str): The generated question
    
    Returns:
        dict: A dictionary with 'text' and 'answer_start'
    """
    # Normalize context and question
    context_lower = context.lower()
    question_lower = question.lower()

    # Preprocessing question types
    wh_words = ['what', 'when', 'where', 'who', 'why', 'how']
    
    # Process the context with spaCy
    doc = nlp(context)
    
    # Identify question type
    question_type = next((word for word in wh_words if word in question_lower), None)
    
    # Strategies for different question types
    potential_answers = []
    
    # Named Entity Extraction
    entities = [
        (ent.text, ent.start_char, ent.end_char, ent.label_) 
        for ent in doc.ents 
        if len(ent.text) > 2
    ]
    
    # Keyword-based context extraction
    def extract_keyword_context(keyword, context, window=50):
        matches = list(re.finditer(r'\b' + re.escape(keyword.lower()) + r'\b', context.lower()))
        return [
            context[max(0, match.start() - window):min(len(context), match.end() + window)].strip()
            for match in matches
        ]
    
    # Strategy based on question type
    if question_type == 'who':
        # Prioritize person entities
        person_entities = [ent for ent in entities if ent[3] in ['PERSON', 'ORG']]
        potential_answers.extend([ent[0] for ent in person_entities])
    
    elif question_type == 'when':
        # Prioritize date and time entities
        time_entities = [ent for ent in entities if ent[3] in ['DATE', 'TIME']]
        potential_answers.extend([ent[0] for ent in time_entities])
    
    elif question_type == 'where':
        # Prioritize location entities
        location_entities = [ent for ent in entities if ent[3] in ['GPE', 'LOC']]
        potential_answers.extend([ent[0] for ent in location_entities])
    
    # Extract keywords from question
    question_keywords = [
        word.lower() for word in re.findall(r'\b\w+\b', question_lower) 
        if word not in wh_words and len(word) > 3
    ]
    
    # Find context around keywords
    for keyword in question_keywords:
        keyword_contexts = extract_keyword_context(keyword, context)
        potential_answers.extend(keyword_contexts)
    
    # Sentence-based extraction
    sentences = [sent.text.strip() for sent in doc.sents]
    
    # Filter and process potential answers
    valid_answers = []
    for answer in potential_answers:
        # Criteria for a good answer
        if (5 < len(answer) < 200 and 
            answer.lower() != context_lower and 
            answer not in valid_answers):
            valid_answers.append(answer)
    
    # Final answer selection
    if valid_answers:
        best_answer = valid_answers[0]
        return {
            "text": best_answer,
            "answer_start": context.lower().find(best_answer.lower())
        }
    
    # Fallback strategies
    if sentences:
        return {
            "text": sentences[0],
            "answer_start": context.find(sentences[0])
        }
    
    # Absolute last resort
    return {
        "text": context[:200],
        "answer_start": 0
    }

def generate_questions_for_context(data, max_questions_per_context=10):
    """
    Generate questions for each context in the SQuAD data structure.
    
    Args:
        data (dict): SQuAD data structure
        max_questions_per_context (int): Maximum number of questions to generate per context
    
    Returns:
        dict: Updated SQuAD data structure
    """
    logger.info("Starting question generation process")
    
    # Initialize the question generator
    question_generator = QuestionGenerator()

    # Track total documents and paragraphs
    total_docs = len(data["data"])
    logger.info(f"Total documents to process: {total_docs}")

    for doc_index, doc in enumerate(data["data"], 1):
        logger.info(f"Processing document {doc_index}/{total_docs}")
        
        for para_index, paragraph in enumerate(doc["paragraphs"], 1):
            context = paragraph["context"]
            logger.info(f"Generating questions for paragraph {para_index}")

            try:
                # Generate questions
                questions = question_generator.generate(context)
                
                # Limit number of questions
                questions = questions[:max_questions_per_context]
                logger.info(f"Number of questions generated: {len(questions)}")

                # Add generated questions to the 'qas' array
                for q_index, question in enumerate(questions, 1):
                    # Extract answer
                    answer_data = extract_answer(context, question)
                    
                    # Only append if answer is meaningful
                    if answer_data['text'] and answer_data['text'].strip():
                        paragraph["qas"].append({
                            "id": f"generated_question_{len(paragraph['qas']) + q_index}",
                            "question": question,
                            "answers": [answer_data]
                        })
                    
                    logger.debug(f"Added question: {question}")
                    logger.debug(f"Extracted answer: {answer_data['text']}")

            except Exception as e:
                logger.error(f"Error generating questions for paragraph: {e}")
                logger.exception("Full error traceback:")

    logger.info("Question generation process completed")
    return data

def save_squad_data(data, file_path):
    """
    Save the updated SQuAD data structure to a JSON file.

    Args:
        data (dict): The SQuAD data structure.
        file_path (str): Path to the file where data will be saved.
    """
    logger.info(f"Saving updated squad data to {file_path}")
    start_time = time.time()
    
    try:
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)
        
        save_time = time.time() - start_time
        logger.info(f"Data saved successfully in {save_time:.2f} seconds")
    except Exception as e:
        logger.error(f"Error saving squad data: {e}")
        logger.exception("Full error traceback:")

def load_squad_data(file_path):
    """
    Load the SQuAD data structure from a JSON file.

    Args:
        file_path (str): Path to the file to load data from.

    Returns:
        dict: The SQuAD data structure.
    """
    logger.info(f"Loading squad data from {file_path}")
    start_time = time.time()
    
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        
        load_time = time.time() - start_time
        logger.info(f"Data loaded successfully in {load_time:.2f} seconds")
        logger.info(f"Total documents in data: {len(data['data'])}")
        return data
    except Exception as e:
        logger.error(f"Error loading squad data: {e}")
        logger.exception("Full error traceback:")
        raise

# Main execution
if __name__ == "__main__":
    logger.info("Starting question generation script")
    
    try:
        file_path = "app/data/squad_data.json"
        
        # Load the existing squad_data.json file
        squad_data = load_squad_data(file_path)
        
        # Generate questions
        updated_squad_data = generate_questions_for_context(squad_data)
        
        # Save the updated data
        save_squad_data(updated_squad_data, file_path)
        
        logger.info("Script completed successfully")
    
    except Exception as e:
        logger.error(f"Unexpected error in main execution: {e}")
        logger.exception("Full error traceback:")