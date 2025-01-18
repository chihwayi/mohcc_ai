import logging
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling, TrainerCallback
from datasets import load_dataset, DatasetDict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the transformed dataset
logger.info("Loading the dataset...")
dataset = load_dataset('json', data_files='app/data/transformed_data.json', split='train')

# Split the dataset into training and evaluation sets
train_test_split = dataset.train_test_split(test_size=0.1)
train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']

# Load the model and tokenizer
logger.info("Loading the model and tokenizer...")
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Add a padding token to the tokenizer
logger.info("Adding padding token to the tokenizer...")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model.resize_token_embeddings(len(tokenizer))

# Tokenize the datasets
logger.info("Tokenizing the datasets...")
def tokenize_function(example):
    return tokenizer(example['prompt'], truncation=True, padding='max_length', max_length=512)

train_dataset = train_dataset.map(tokenize_function, batched=True)
eval_dataset = eval_dataset.map(tokenize_function, batched=True)

# Data collator that handles batch generation
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# Define training arguments
training_args = TrainingArguments(
    output_dir='app/models/',
    evaluation_strategy='epoch',
    logging_dir='app/models/logs',  # Directory for storing logs
    logging_steps=10,  # Log every 10 steps
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    save_steps=10_000,
    save_total_limit=2,
    report_to="none"  # Disable the default reporting to enable custom logging
)

# Custom Callback for Logging Training Metrics
class LoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        _ = logs.pop("total_flos", None)
        if state.is_local_process_zero:
            logger.info(f"Step: {state.global_step}, Logs: {logs}")

# Create Trainer instance with logging callback
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    callbacks=[LoggingCallback]
)

# Train the model
logger.info("Starting training...")
trainer.train()

# Save the model
logger.info("Saving the model...")
model.save_pretrained('app/models/')
tokenizer.save_pretrained('app/models/')

logger.info("Training complete. Model saved to 'app/models/'")
