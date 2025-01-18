from transformers import RobertaTokenizerFast, RobertaForQuestionAnswering
import torch

# Load the fine-tuned model and tokenizer
model_name = "app/models/qa_model"  # Update this path if necessary
tokenizer = RobertaTokenizerFast.from_pretrained(model_name)
model = RobertaForQuestionAnswering.from_pretrained(model_name)

def get_answer(question: str, context: str):
    """Generate an answer to the question based on the context."""
    inputs = tokenizer(question, context, return_tensors='pt', truncation=True, padding="max_length", max_length=512)
    input_ids = inputs["input_ids"].tolist()[0]

    with torch.no_grad():
        outputs = model(**inputs)

    start_logits = outputs.start_logits
    end_logits = outputs.end_logits

    start_index = torch.argmax(start_logits)
    end_index = torch.argmax(end_logits) + 1

    # Decode the tokens
    answer_tokens = tokenizer.convert_ids_to_tokens(input_ids[start_index:end_index])
    answer = tokenizer.convert_tokens_to_string(answer_tokens)
    
    # Strip special tokens and padding
    answer = answer.replace('<s>', '').replace('</s>', '').replace('<pad>', '').strip()
    
    return answer
