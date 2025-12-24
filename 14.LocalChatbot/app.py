"""
Local Chatbot Application - No API Keys Required!
==================================================

Features:
1. YouTube Video Q&A - Ask questions about YouTube videos
2. Document Q&A - Chat with your documents
3. General Chat - Free-form conversation
4. All using local models (Ollama + HuggingFace)

Requirements:
- Ollama installed and running
- Python packages from requirements.txt
"""

from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
import os
import logging
from datetime import datetime
import uuid

# Import our modules
from youtube.youtube_loader import YouTubeLoader
from documents.document_qa import DocumentQA
from chat.general_chat import GeneralChat

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.urandom(24)  # For session management
CORS(app)

# Initialize chatbot modules
youtube_loader = YouTubeLoader()
document_qa = DocumentQA()
general_chat = GeneralChat()

@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')

@app.route('/api/youtube/load', methods=['POST'])
def load_youtube():
    """Load YouTube video transcript."""
    try:
        data = request.json
        video_url = data.get('url')
        
        if not video_url:
            return jsonify({'error': 'YouTube URL is required'}), 400
        
        logger.info(f"Loading YouTube video: {video_url}")
        
        # Extract video ID
        video_id = youtube_loader.extract_video_id(video_url)
        if not video_id:
            return jsonify({'error': 'Invalid YouTube URL'}), 400
        
        # Load transcript
        result = youtube_loader.load_video(video_id)
        
        if result['success']:
            # Store in session
            session['youtube_video_id'] = video_id
            session['youtube_title'] = result.get('title', 'Unknown')
            
            return jsonify({
                'success': True,
                'title': result.get('title', 'Unknown'),
                'transcript_length': result.get('transcript_length', 0),
                'chunks_created': result.get('chunks_created', 0),
                'message': f"Video loaded successfully! {result.get('chunks_created', 0)} chunks created."
            })
        else:
            return jsonify({'error': result.get('error', 'Failed to load video')}), 500
            
    except Exception as e:
        logger.error(f"Error loading YouTube video: {str(e)}")
        return jsonify({'error': f'Error: {str(e)}'}), 500

@app.route('/api/youtube/chat', methods=['POST'])
def youtube_chat():
    """Chat about YouTube video."""
    try:
        data = request.json
        question = data.get('question')
        
        if not question:
            return jsonify({'error': 'Question is required'}), 400
        
        video_id = session.get('youtube_video_id')
        if not video_id:
            return jsonify({'error': 'No video loaded. Please load a YouTube video first.'}), 400
        
        logger.info(f"Question about video {video_id}: {question}")
        
        # Get answer using RAG
        answer = youtube_loader.ask_question(question, video_id)
        
        return jsonify({
            'success': True,
            'answer': answer.get('answer', 'Sorry, I could not generate an answer.'),
            'sources': answer.get('sources', [])
        })
        
    except Exception as e:
        logger.error(f"Error in YouTube chat: {str(e)}")
        return jsonify({'error': f'Error: {str(e)}'}), 500

@app.route('/api/documents/upload', methods=['POST'])
def upload_document():
    """Upload and process document."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save file temporarily
        file_id = str(uuid.uuid4())
        upload_dir = 'uploads'
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, f"{file_id}_{file.filename}")
        file.save(file_path)
        
        logger.info(f"Processing document: {file.filename}")
        
        # Process document
        result = document_qa.load_document(file_path, file_id)
        
        if result['success']:
            # Store in session
            if 'documents' not in session:
                session['documents'] = []
            session['documents'].append({
                'id': file_id,
                'name': file.filename,
                'chunks': result.get('chunks_created', 0)
            })
            
            return jsonify({
                'success': True,
                'file_id': file_id,
                'filename': file.filename,
                'chunks_created': result.get('chunks_created', 0),
                'message': f"Document processed! {result.get('chunks_created', 0)} chunks created."
            })
        else:
            return jsonify({'error': result.get('error', 'Failed to process document')}), 500
            
    except Exception as e:
        logger.error(f"Error uploading document: {str(e)}")
        return jsonify({'error': f'Error: {str(e)}'}), 500

@app.route('/api/documents/chat', methods=['POST'])
def document_chat():
    """Chat about uploaded documents."""
    try:
        data = request.json
        question = data.get('question')
        file_id = data.get('file_id')
        
        if not question:
            return jsonify({'error': 'Question is required'}), 400
        
        if not file_id:
            return jsonify({'error': 'File ID is required'}), 400
        
        logger.info(f"Question about document {file_id}: {question}")
        
        # Get answer
        answer = document_qa.ask_question(question, file_id)
        
        return jsonify({
            'success': True,
            'answer': answer.get('answer', 'Sorry, I could not generate an answer.'),
            'sources': answer.get('sources', [])
        })
        
    except Exception as e:
        logger.error(f"Error in document chat: {str(e)}")
        return jsonify({'error': f'Error: {str(e)}'}), 500

@app.route('/api/chat/general', methods=['POST'])
def general_chat_api():
    """General conversation without context."""
    try:
        data = request.json
        message = data.get('message')
        history = data.get('history', [])
        
        if not message:
            return jsonify({'error': 'Message is required'}), 400
        
        logger.info(f"General chat message: {message[:50]}...")
        
        # Get response
        response = general_chat.chat(message, history)
        
        return jsonify({
            'success': True,
            'response': response
        })
        
    except Exception as e:
        logger.error(f"Error in general chat: {str(e)}")
        return jsonify({'error': f'Error: {str(e)}'}), 500

@app.route('/api/clear', methods=['POST'])
def clear_session():
    """Clear session data."""
    try:
        session.clear()
        return jsonify({'success': True, 'message': 'Session cleared'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/status', methods=['GET'])
def status():
    """Check system status."""
    try:
        # Check if Ollama is running
        ollama_status = general_chat.check_ollama()
        
        return jsonify({
            'success': True,
            'ollama_running': ollama_status,
            'message': 'Ollama is running' if ollama_status else 'Ollama is not running. Please start it with: ollama serve'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*80)
    print("🚀 LOCAL CHATBOT APPLICATION")
    print("="*80)
    print("\n✅ Features:")
    print("   1. YouTube Video Q&A (No API keys needed!)")
    print("   2. Document Q&A (Upload PDFs, TXT, etc.)")
    print("   3. General Chat (Free conversation)")
    print("\n📋 Requirements:")
    print("   - Ollama must be running: ollama serve")
    print("   - Pull a model: ollama pull mistral")
    print("\n🌐 Starting server on http://localhost:5000")
    print("="*80 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)


