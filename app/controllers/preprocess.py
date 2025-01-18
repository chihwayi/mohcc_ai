import json

# Load the original dataset
with open('app/data/clean.json', 'r') as file:
    data = json.load(file)

# Create a new list with the desired format
preprocessed_data = [
    {"prompt": item["question"], "completion": item["answer"]}
    for item in data
]

# Save the preprocessed data
with open('app/data/preprocessed_data.json', 'w') as file:
    json.dump(preprocessed_data, file, indent=2)
