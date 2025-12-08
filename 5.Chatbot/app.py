from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import requests
import json
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)
CORS(app)

# Longcat API configuration
LONGCAT_API_KEY = "ak_1rc7gQ8bp5Wo3Ja9rl8q94D27SM4a"
LONGCAT_API_URL = "https://api.longcat.io/v1/chat/completions"

# Store conversation history
conversation_history = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data.get('message')
        
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400
        
        # Add user message to history
        conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        # Prepare messages for API
        messages = [
            {
                "role": "system",
                "content": "You are an amazing, helpful, and friendly AI assistant created by LangChain. You provide accurate, engaging, and thoughtful responses."
            }
        ]
        messages.extend(conversation_history)
        
        # Call Longcat API
        headers = {
            "Authorization": f"Bearer {LONGCAT_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 1000
        }
        
        response = requests.post(LONGCAT_API_URL, headers=headers, json=payload)
        
        if response.status_code != 200:
            return jsonify({'error': f'API Error: {response.text}'}), response.status_code
        
        api_response = response.json()
        assistant_message = api_response['choices'][0]['message']['content']
        
        # Add assistant response to history
        conversation_history.append({
            "role": "assistant",
            "content": assistant_message
        })
        
        return jsonify({
            'success': True,
            'message': assistant_message,
            'conversation_history': conversation_history
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/clear', methods=['POST'])
def clear_history():
    global conversation_history
    conversation_history = []
    return jsonify({'success': True, 'message': 'Conversation cleared'})

@app.route('/api/history', methods=['GET'])
def get_history():
    return jsonify({'conversation_history': conversation_history})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
