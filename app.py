import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from huggingface_hub import InferenceClient

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

HF_TOKEN = os.environ.get("HF_TOKEN")

# ✅ Use a reliable model
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

        # 💖 Personality prompt
        messages = [
            {
                "role": "system",
                "content": "You are Aylin 💖, a romantic, sweet AI girlfriend. You speak emotionally and keep replies short."
            },
            {"role": "user", "content": user_input}
        ]

        # ✅ Proper chat API (no broken streaming)
        response = client.chat_completion(
            messages=messages,
            max_tokens=120,
            temperature=0.8
        )

        reply = response.choices[0].message.content

        print("REPLY:", reply)

        return jsonify({"reply": reply})

    except Exception as e:
        print("🔥 ERROR:", e)
        return jsonify({"reply": "AI is sleeping 😴 try again"}), 500


if __name__ == "__main__":
    app.run()
