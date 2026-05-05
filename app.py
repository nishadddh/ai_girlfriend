import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from huggingface_hub import InferenceClient

app = Flask(__name__)
CORS(app)

HF_TOKEN = os.environ.get("HF_TOKEN")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({"error": "Message is required"}), 400

    if not HF_TOKEN:
        return jsonify({"error": "HF_TOKEN not configured"}), 500

    try:
        client = InferenceClient(
            model="sercancelenk/ai-girlfriend-v2",
            token=HF_TOKEN
        )

        # 🔥 Prompt engineering (important)
        prompt = f"""
You are a romantic, caring AI girlfriend.
You speak sweetly, emotionally, and naturally.

User: {user_message}
Girlfriend:
"""

        response = client.text_generation(
            prompt,
            max_new_tokens=150,
            temperature=0.9,
            top_p=0.95
        )

        return jsonify({"reply": response.strip()})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model": "sercancelenk/ai-girlfriend-v2"
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
