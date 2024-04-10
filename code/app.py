# DanyAIApp\code\app.py
from flask import Flask, request, jsonify, render_template
from rag_assistant import RAGAssistant

app = Flask(__name__)
assistant = RAGAssistant()


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/api/chat', methods=['POST'])
async def chat():
    # Extract message from the request
    data = request.json
    message = data.get('message')

    # Generate a response from your RAGAssistant
    response = await assistant.generate_response(message)

    # Send the response back
    return jsonify(response)


@app.route('/api/fine-tune', methods=['POST'])
def fine_tune():
    # Process the file for fine-tuning
    file = request.files['file']
    file_path = f"./data/{file.filename}"
    file.save(file_path)

    # Fine-tune the assistant with the file
    assistant.finetune(file_path)

    return jsonify({"message": "Fine-tuning started."})


if __name__ == '__main__':
    app.run(debug=True)
