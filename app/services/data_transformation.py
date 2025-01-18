import json

# Load the original dataset
with open('app/data/clean.json', 'r') as file:
    data = json.load(file)

# Check and transform the data to the required format
transformed_data = []
for item in data:
    if "question" in item and "answer" in item:
        transformed_data.append({
            "prompt": item["question"],
            "completion": item["answer"]
        })
    else:
        print(f"Missing question or answer in item: {item}")

# Save the transformed data
with open('app/data/transformed_data.json', 'w') as file:
    json.dump(transformed_data, file, indent=2)

print("Data transformation complete.")
