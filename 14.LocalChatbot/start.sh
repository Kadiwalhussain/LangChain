#!/bin/bash

# Local Chatbot Quick Start Script

echo "🚀 Starting Local Chatbot Application"
echo "======================================"
echo ""

# Check if Ollama is running
echo "📋 Checking prerequisites..."

if ! command -v ollama &> /dev/null; then
    echo "❌ Ollama is not installed!"
    echo "   Install from: https://ollama.ai"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install requirements
if [ ! -f ".installed" ]; then
    echo "📥 Installing dependencies (this may take a few minutes)..."
    pip install -r requirements.txt
    touch .installed
fi

# Create necessary directories
mkdir -p uploads youtube_db documents_db

# Check Ollama status
echo ""
echo "🔍 Checking Ollama status..."
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "✅ Ollama is running"
else
    echo "⚠️  Ollama is not running!"
    echo "   Please start it in another terminal: ollama serve"
    echo ""
    read -p "Press Enter to continue anyway..."
fi

# Start the application
echo ""
echo "🌐 Starting Flask application..."
echo "   Open http://localhost:5000 in your browser"
echo ""
python app.py


