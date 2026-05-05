import os
import requests
import time
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

HF_TOKEN = os.environ.get("HF_TOKEN")

MODELS = [
    "HuggingFaceH4/zephyr-7b-beta",
    "google/flan-t5-large"
]

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}"
}


def query(model, prompt):
    url = f"https://api-inference.huggingface.co/models/{model}"

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 80,
            "temperature": 0.8
        }
    }

    response = requests.post(url, headers=HEADERS, json=payload)
    return response.json()


@app.route("/")
def home():
    return jsonify({"status": "HF Smart API running 🚀"})


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

    # 🔥 TRY MODELS WITH RETRY
    for model in MODELS:
        for attempt in range(2):  # retry 2 times
            try:
                result = query(model, prompt)
                print(f"{model} →", result)

                # 💤 Model loading → wait and retry
                if isinstance(result, dict) and "error" in result:
                    if "loading" in result["error"].lower():
                        time.sleep(5)
                        continue
                    else:
                        break

                # ✅ Success
                if isinstance(result, list):
                    text = result[0].get("generated_text", "")
                    reply = text.split("Aylin:")[-1].strip()

                    if reply:
                        return jsonify({"reply": reply})

            except Exception as e:
                print("ERROR:", e)
                continue

    return jsonify({"reply": "Still waking up… try again in a few seconds 💖"})


if __name__ == "__main__":
    app.run()
