from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import requests
import json
from dotenv import load_dotenv
import os
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

load_dotenv()

app = Flask(__name__)
CORS(app)

# Longcat API Key
LONGCAT_API_KEY = "ak_1rc7gQ8bp5Wo3Ja9rl8q94D27SM4a"

# Store conversation history
conversation_history = []

# Mock responses for demo mode
MOCK_RESPONSES = {
    "hello": "Hello! I'm your AI assistant. How can I help you today?",
    "hi": "Hi there! What would you like to talk about?",
    "python": "Python is a versatile programming language great for beginners and experts alike. It's widely used in web development, data science, and AI. Would you like to learn more about a specific topic?",
    "machine learning": "Machine Learning is a subset of AI that enables systems to learn and improve from experience. It involves training models on data to make predictions or decisions. What aspect interests you?",
    "langchain": "LangChain is a framework for developing applications powered by language models. It provides tools for chaining different components together. Very useful for building AI applications!",
    "default": "That's an interesting question! I'm learning and improving my responses. Could you tell me more about what you'd like to know?"
}

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
        
        # Get response - try API first, fall back to mock
        assistant_message = get_ai_response(user_message)
        
        if not assistant_message:
            return jsonify({'error': 'Failed to get response'}), 500
        
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
        logger.error(f"Error in chat: {str(e)}")
        return jsonify({'error': f'Error: {str(e)}'}), 500


def get_ai_response(user_message):
    """Get AI response from Longcat API"""
    
    # Try Longcat API first
    response = call_longcat_api(user_message)
    if response:
        return response
    
    # Fall back to mock responses if API fails
    return get_mock_response(user_message)


def call_longcat_api(user_message):
    """Call Longcat API with proper format"""
    try:
        headers = {
            "Authorization": f"Bearer {LONGCAT_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Prepare payload for Longcat API
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {
                    "role": "system",
                    "content": "You are an amazing, helpful, and friendly AI assistant created by LangChain. You provide accurate, engaging, and thoughtful responses."
                },
                {
                    "role": "user",
                    "content": user_message
                }
            ],
            "temperature": 0.7,
            "max_tokens": 500
        }
        
        print("\n" + "="*80)
        print("📨 USER MESSAGE:")
        print(f"   {user_message}")
        print("="*80)
        print(f"🔄 Calling Longcat API...")
        print(f"   API Key: {LONGCAT_API_KEY[:20]}...")
        print(f"   Endpoint: https://api.longcat.io/v1/chat/completions")
        print("-"*80)
        
        # Try the correct Longcat endpoint
        response = requests.post(
            "https://api.longcat.io/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=15
        )
        
        print(f"📊 API Response Status: {response.status_code}")
        print(f"📝 Response Body: {response.text[:300]}")
        
        if response.status_code == 200:
            api_response = response.json()
            message_content = api_response['choices'][0]['message']['content']
            print(f"\n✅ SUCCESS - Got response from Longcat API!")
            print(f"📣 Assistant Response:")
            print(f"   {message_content}")
            print("="*80 + "\n")
            return message_content
        else:
            print(f"⚠️  API Error: Status {response.status_code}")
            print(f"   {response.text}")
            print("-"*80)
            return None
    except requests.exceptions.Timeout:
        print(f"⏱️  Timeout: Longcat API took too long to respond")
        print("-"*80)
        return None
    except requests.exceptions.ConnectionError as e:
        print(f"🔌 Connection Error: Cannot reach Longcat API")
        print(f"   {str(e)}")
        print("-"*80)
        return None
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("-"*80)
        return None


def get_mock_response(user_message):
    """Generate mock response for demo"""
    user_lower = user_message.lower()
    
    response_text = MOCK_RESPONSES["default"]
    for key, response in MOCK_RESPONSES.items():
        if key in user_lower:
            response_text = response
            break
    
    print("\n" + "="*80)
    print(f"📨 USER MESSAGE:")
    print(f"   {user_message}")
    print("="*80)
    print(f"📚 Using Mock Response (Longcat API unavailable)")
    print(f"   Status: Fallback Mode")
    print("-"*80)
    print(f"✅ Mock Response Generated:")
    print(f"   {response_text}")
    print("="*80 + "\n")
    
    return response_text


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
