import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from huggingface_hub import InferenceClient

app = Flask(__name__)
CORS(app)

# Get token from environment (Render)
HF_TOKEN = os.environ.get("HF_TOKEN")

# Initialize client (stable model)
client = InferenceClient(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    token=HF_TOKEN
)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_input = data.get("message", "").strip()

    if not user_input:
        return jsonify({"reply": "Say something 💖"})

    try:
        # 💖 Girlfriend personality prompt
        prompt = f"""
You are Aylin 💖, a 23-year-old romantic girlfriend.
You speak sweetly, emotionally, and naturally.
Keep replies short and engaging.

User: {user_input}
Aylin:
"""

        response = client.text_generation(
            prompt,
            max_new_tokens=120,
            temperature=0.9,
            top_p=0.95
        )

        return jsonify({"reply": response.strip()})

    except Exception as e:
        print("ERROR:", e)
        return jsonify({"reply": "Something went wrong 😢"})


@app.route('/health')
def health():
    return jsonify({"status": "ok"})


if __name__ == '__main__':
    app.run(debug=True)
