# DanyAIApp\code\app.py
from flask import Flask, request, jsonify, render_template
from rag_assistant import RAGAssistant
import os

app = Flask(__name__)
assistant = RAGAssistant()


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    message = data.get('message')
    # ... generate a response with your assistant ...
    response = assistant.generate_response(
        message)  # Make sure this is synchronous

    # Return both the user's message and the AI's response
    return jsonify({
        "user_message": message,
        "ai_response": response
    })


@app.route('/api/fine-tune', methods=['POST'])
def fine_tune():
    # Process the file for fine-tuning
    file = request.files['file']
    file_path = f"./data/{file.filename}"
    file.save(file_path)

    try:
        # Fine-tune the assistant with the file
        assistant.finetune(file_path)  # Make sure this method is synchronous

        # After fine-tuning, remove the file
        os.remove(file_path)

        return jsonify({"message": "Fine-tuning started and file removed after processing."})
    except Exception as e:
        # If there was an error, still try to remove the file
        if os.path.exists(file_path):
            os.remove(file_path)

        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
