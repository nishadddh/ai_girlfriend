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

@app.route("/")
def home():
    return jsonify({"status": "HF Router API running 🚀"})

@app.route("/chat", methods=["POST"])
def chat():
    try:
        user_input = request.json.get("message", "").strip()

        if not user_input:
            return jsonify({"reply": "Say something 💖"})

        completion = client.chat.completions.create(
            model="HuggingFaceH4/zephyr-7b-beta",  # ✅ FIXED
            messages=[
                {
                    "role": "system",
                    "content": "You are Aylin 💖, a romantic, sweet AI girlfriend. Speak emotionally and keep replies short."
                },
                {
                    "role": "user",
                    "content": user_input
                }
            ],
            max_tokens=100,
            temperature=0.9
        )

        reply = completion.choices[0].message.content

        return jsonify({"reply": reply})

    except Exception as e:
        print("🔥 ERROR:", e)
        return jsonify({"reply": "Hmm… something went wrong 😢"}), 500


if __name__ == "__main__":
    app.run()
