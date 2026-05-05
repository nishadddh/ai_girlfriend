import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI

app = Flask(__name__)
CORS(app)

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ.get("HF_TOKEN"),
)

# ✅ Working models (fallback system)
MODELS = [
    "mistralai/Mistral-7B-Instruct-v0.2:openai",
    "HuggingFaceH4/zephyr-7b-beta:openai"
]

@app.route("/")
def home():
    return jsonify({"status": "HF Router running 🚀"})


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_input = data.get("message", "").strip()

    if not user_input:
        return jsonify({"reply": "Say something 💖"})

    messages = [
        {
            "role": "system",
            "content": "You are Aylin 💖, a romantic girlfriend. You are sweet, emotional, and slightly flirty."
        },
        {
            "role": "user",
            "content": user_input
        }
    ]

    # 🔥 Try multiple models
    for model in MODELS:
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=100,
                temperature=0.9
            )

            reply = completion.choices[0].message.content

            if reply:
                return jsonify({
                    "reply": reply,
                    "model": model
                })

        except Exception as e:
            print(f"❌ Model failed: {model}")
            print("ERROR:", e)
            continue

    return jsonify({
        "reply": "I'm a bit sleepy 😴 try again...",
        "error": "All models failed"
    }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
