from flask import Flask, request, jsonify
import openai  # Use GPT-4 or any other model (you can replace this with your local AI model)

app = Flask(__name__)

# Replace with your OpenAI API Key (or another AI model)
openai.api_key = "your-api-key"

@app.route("/fix_response", methods=["POST"])
def fix_response():
    data = request.json
    user_message = data.get("user_message", "")
    bot_response = data.get("bot_response", "")

    # Use GPT-4 to verify and correct chatbot response
    prompt = f"""User: {user_message}
    Chatbot: {bot_response}
    If the response is incorrect or doesn't match the user's question, provide a better response.
    Otherwise, return the same response.
    Fixed Response:"""

    try:
        ai_response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "system", "content": "You are an expert chatbot trainer."},
                      {"role": "user", "content": prompt}]
        )
        fixed_response = ai_response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        fixed_response = bot_response  # Keep original if API fails

    return jsonify({"fixed_response": fixed_response})

if __name__ == "__main__":
    app.run(debug=True)
