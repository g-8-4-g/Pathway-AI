import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

# Load model and tokenizer (FIXED)
model_name = "ai4bharat/indic-bert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)  # ✅ Corrected Model Type

# Load tokenized dataset
tokenized_data = torch.load("data/tokenized_chatbot_data.pt")

# Split dataset: 80% train, 20% validation
train_size = 0.8
train_data, eval_data = train_test_split(tokenized_data, train_size=train_size, random_state=42)

# Custom Dataset Class (Fixed)
class ChatbotDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.data[idx]["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(self.data[idx]["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(self.data[idx]["labels"], dtype=torch.long),
        }

# Convert to dataset
train_dataset = ChatbotDataset(train_data)
eval_dataset = ChatbotDataset(eval_data)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",  # Ensures evaluation every epoch
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    save_steps=500,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=200,
    report_to="none",
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

# Train the model
trainer.train()

# Save trained model
model.save_pretrained("models/chatbot_model")
tokenizer.save_pretrained("models/chatbot_model")

print("✅ Training complete! Model saved in 'models/chatbot_model'.")
