from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load Pretrained Hinglish BERT Model
model_name = "ai4bharat/indic-bert"
from transformers import AlbertTokenizer
tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")

model = AutoModelForSequenceClassification.from_pretrained(model_name)

def process_text(text):
    tokens = tokenizer(text, return_tensors="pt")
    return tokens

# Example Test
sentence = "Mujhe weight loss ke liye diet suggest karo"
tokens = process_text(sentence)
print(tokens)
