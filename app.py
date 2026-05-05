import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from huggingface_hub import InferenceClient

app = Flask(__name__)
CORS(app)

HF_TOKEN = os.environ.get("HF_TOKEN")

client = InferenceClient(
    model="HuggingFaceH4/zephyr-7b-beta",
    token=HF_TOKEN
)

@app.route("/")
def home():
    return jsonify({"status": "API running 🚀"})


@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        user_input = data.get("message", "").strip()

        if not user_input:
            return jsonify({"reply": "Say something 💖"})

        messages = [
            {
                "role": "system",
                "content": "You are Aylin 💖, a romantic AI girlfriend. Keep replies short and emotional."
            },
            {
                "role": "user",
                "content": user_input
            }
        ]

        # ✅ CORRECT METHOD
        response = client.chat_completion(
            messages=messages,
            max_tokens=100,
            temperature=0.8
        )

        reply = response.choices[0].message.content

        print("REPLY:", reply)

        return jsonify({"reply": reply})

    except Exception as e:
        print("🔥 ERROR:", e)
        return jsonify({"reply": "AI failed 😢 try again"}), 500


if __name__ == "__main__":
    app.run()
