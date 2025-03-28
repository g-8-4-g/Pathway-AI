import json

# Load and check the medical chatbot dataset
with open("data/medical_chatbot_conversations.json", "r", encoding="utf-8") as f:
    medical_data = json.load(f)

# Load and check the internship chatbot dataset
with open("data/internship_chatbot_conversations_only.json", "r", encoding="utf-8") as f:
    internship_data = json.load(f)

# Print sample conversations
print("Medical Chatbot Sample:", medical_data[:2])  # Print first 2 conversations
print("Internship Chatbot Sample:", internship_data[:2])  # Print first 2 conversations
