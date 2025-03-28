import json
import requests

API_URL = "http://127.0.0.1:5000/fix_response"  # Replace with your actual API URL if needed

def clean_data_with_api(file_path, save_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    fixed_data = []
    
    for convo in data:
        messages = convo.get("messages", [])
        if len(messages) == 2 and messages[0]["role"] == "user" and messages[1]["role"] == "bot":
            user_msg = messages[0]["text"]
            bot_msg = messages[1]["text"]

            # Send request to the API for correction
            payload = {"user_message": user_msg, "bot_response": bot_msg}
            response = requests.post(API_URL, json=payload)
            
            if response.status_code == 200:
                fixed_response = response.json().get("fixed_response", bot_msg)  # Get corrected text
                messages[1]["text"] = fixed_response
            else:
                print(f"API Error: {response.status_code}")
            
            fixed_data.append(convo)

    # Save cleaned data
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(fixed_data, f, indent=4)

    print(f"Cleaned data saved to {save_path}")

# Run cleaning for both datasets
clean_data_with_api("data/medical_chatbot_conversations.json", "data/medical_chatbot_cleaned.json")
clean_data_with_api("data/internship_chatbot_conversations_only.json", "data/internship_chatbot_cleaned.json")
