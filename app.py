import os
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

HF_TOKEN = os.environ.get("HF_TOKEN")

# ✅ Try multiple free models
MODELS = [
    "HuggingFaceH4/zephyr-7b-beta",
    "google/flan-t5-large",
    "facebook/blenderbot-400M-distill"
]

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}"
}


def query_model(model, prompt):
    API_URL = f"https://api-inference.huggingface.co/models/{model}"

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 80,
            "temperature": 0.8
        }
    }

    response = requests.post(API_URL, headers=HEADERS, json=payload)
    return response.json()


@app.route("/")
def home():
    return jsonify({"status": "HF Multi-Model API running 🚀"})


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_input = data.get("message", "").strip()

    if not user_input:
        return jsonify({"reply": "Say something 💖"})

    prompt = f"""
You are Aylin 💖, a romantic girlfriend.
Reply sweetly and emotionally.

User: {user_input}
Aylin:
"""

    # 🔥 TRY MULTIPLE MODELS
    for model in MODELS:
        try:
            result = query_model(model, prompt)
            print(f"MODEL {model}:", result)

            # ❌ model loading
            if isinstance(result, dict) and "error" in result:
                continue

            # ✅ success
            if isinstance(result, list):
                text = result[0].get("generated_text", "")
                reply = text.split("Aylin:")[-1].strip()

                if reply:
                    return jsonify({"reply": reply})

        except Exception as e:
            print("ERROR:", e)
            continue

    # ❌ if all fail
    return jsonify({"reply": "I’m here… but a bit sleepy 😴 try again"})


if __name__ == "__main__":
    app.run()
