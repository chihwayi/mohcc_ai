import json

def generate_positions(data):
    result = []
    for doc in data:
        # Simulate a context by concatenating all questions and answers from the document
        document_context = ""
        for entry in data:
            if entry['document_number'] == doc['document_number']:
                document_context += f"{entry['question']} {entry['answer']} "
        
        # Find the position of the answer in the context
        answer_start = document_context.find(doc['answer'])
        
        if answer_start == -1:
            # If the exact answer isn't found, we might need to handle this differently, 
            # perhaps by looking for partial matches or skipping this entry
            print(f"Warning: Could not find answer for question: {doc['question']} in document {doc['document_number']}")
            continue
        
        # Calculate positions
        start_position = answer_start
        end_position = start_position + len(doc['answer'])
        
        # Append the new entry with positions
        result.append({
            "question": doc['question'],
            "answer": doc['answer'],
            "start_position": start_position,
            "end_position": end_position
        })
    
    return result

# Load the JSON dataset
with open('app/data/clean.json', 'r') as infile:
    data = json.load(infile)

# Generate positions
new_data = generate_positions(data)

# Write to a JSONL file including the positions
with open('app/data/qa_dataset.jsonl', 'w') as outfile:
    for item in new_data:
        json.dump(item, outfile)
        outfile.write('\n')

print("Conversion to JSONL with start and end positions complete.")