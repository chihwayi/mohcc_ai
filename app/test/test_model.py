from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the trained model and tokenizer
model = GPT2LMHeadModel.from_pretrained('app/models/')
tokenizer = GPT2Tokenizer.from_pretrained('app/models/')

def generate_answer(question):
    inputs = tokenizer.encode(question, return_tensors='pt')
    outputs = model.generate(inputs, max_length=150, num_return_sequences=1)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Limit the answer to the first sentence
    answer = answer.split('.')[0] + '.'
    return answer

# Test the model
question = "When is re-testing for HIV recommended?"
print(generate_answer(question))
