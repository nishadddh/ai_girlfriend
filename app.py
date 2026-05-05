import os
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

HF_TOKEN = os.environ.get("HF_TOKEN")

API_URL = "https://api-inference.huggingface.co/models/sercancelenk/ai-girlfriend-v2"

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

@app.route("/")
def home():
    return jsonify({"status": "API running 🚀"})


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_input = data.get("message", "").strip()

    if not user_input:
        return jsonify({"reply": "Say something 💖"})

    try:
        prompt = f"""
You are Aylin 💖, a romantic girlfriend.
Speak sweetly and emotionally.

User: {user_input}
Aylin:
"""

        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 100,
                "temperature": 0.8
            }
        }

        response = requests.post(API_URL, headers=headers, json=payload)
        result = response.json()

        print("HF RESPONSE:", result)

        # ✅ Extract text safely
        if isinstance(result, list):
            reply = result[0]["generated_text"]
            reply = reply.split("Aylin:")[-1].strip()
        else:
            reply = "Try again 💖"

        return jsonify({"reply": reply})

    except Exception as e:
        print("🔥 ERROR:", e)
        return jsonify({"reply": "Server issue 😢"}), 500


if __name__ == "__main__":
    app.run()
