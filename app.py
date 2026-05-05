import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from huggingface_hub import InferenceClient

app = Flask(__name__)

# ✅ Allow your InfinityFree domain (or allow all)
CORS(app, resources={r"/*": {"origins": "*"}})

HF_TOKEN = os.environ.get("HF_TOKEN")

client = InferenceClient(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    token=HF_TOKEN
)

@app.route("/")
def home():
    return jsonify({"status": "API running"})

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_input = data.get("message", "").strip()

    if not user_input:
        return jsonify({"reply": "Say something 💖"})

    try:
        prompt = f"""
You are Aylin 💖, a romantic AI girlfriend.
Speak sweetly, emotionally, and naturally.
Keep replies short.

User: {user_input}
Aylin:
"""

        response = client.text_generation(
            prompt,
            max_new_tokens=120,
            temperature=0.9
        )

        return jsonify({"reply": response.strip()})

    except Exception as e:
        print("ERROR:", e)
        return jsonify({"reply": "Server error 😢"}), 500


if __name__ == "__main__":
    app.run()
