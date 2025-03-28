import json
import torch
from transformers import AutoTokenizer

# Load IndicBERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-bert")

# Function to load cleaned chatbot data
def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

# Load cleaned data
medical_data = load_data("data/medical_chatbot_cleaned.json")
internship_data = load_data("data/internship_chatbot_cleaned.json")

# Combine both datasets
combined_data = medical_data + internship_data

# Prepare tokenized dataset
tokenized_data = []
for convo in combined_data:
    for msg in convo["messages"]:
        if msg["role"] == "user":
            input_text = msg["text"]
        elif msg["role"] == "bot":
            output_text = msg["text"]

            # Tokenize input and output
            encoded_input = tokenizer(input_text, padding="max_length", truncation=True, max_length=128)
            encoded_output = tokenizer(output_text, padding="max_length", truncation=True, max_length=128)

            # Store as a dictionary
            tokenized_data.append({
                "input_ids": encoded_input["input_ids"],
                "attention_mask": encoded_input["attention_mask"],
                "labels": encoded_output["input_ids"]
            })

# Save processed dataset
torch.save(tokenized_data, "data/tokenized_chatbot_data.pt")
print("✅ Tokenized dataset saved as data/tokenized_chatbot_data.pt")
