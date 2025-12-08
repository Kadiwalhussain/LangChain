# LangChain AI Chatbot

An amazing, modern chatbot built with Flask (Python) backend and advanced HTML/CSS/JavaScript frontend, powered by the Longcat API.

## Features

✨ **Modern UI/UX**
- Beautiful gradient design with dark theme
- Smooth animations and transitions
- Responsive design for all devices
- Real-time chat interface

🤖 **Advanced Functionality**
- Powered by Longcat API (GPT-3.5-turbo)
- Conversation history tracking
- Message formatting (bold, italic, code blocks)
- Auto-scrolling chat
- Loading indicators
- Clear chat history
- Quick suggestion buttons

⌨️ **Keyboard Shortcuts**
- `Ctrl/Cmd + Enter` - Send message
- `Escape` - Clear input

## Installation

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Add your API key to `.env`:**
```
LONGCAT_API_KEY=ak_1rc7gQ8bp5Wo3Ja9rl8q94D27SM4a
```

## Usage

1. **Start the Flask server:**
```bash
python app.py
```

2. **Open your browser:**
```
http://localhost:5000
```

3. **Start chatting!**
Type your message and press Enter or click the send button.

## Project Structure

```
5.Chatbot/
├── app.py                 # Flask backend
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── templates/
    └── index.html        # HTML template
└── static/
    ├── css/
    │   └── style.css     # Advanced styling
    └── js/
        └── script.js     # Frontend logic
```

## API Endpoints

- `GET /` - Render main chat page
- `POST /api/chat` - Send message to AI
- `POST /api/clear` - Clear conversation history
- `GET /api/history` - Get conversation history

## Technologies Used

**Backend:**
- Python 3.x
- Flask
- Flask-CORS
- Requests
- python-dotenv

**Frontend:**
- HTML5
- CSS3 (with animations & gradients)
- JavaScript (Vanilla)
- Font Awesome Icons

## Features Breakdown

### Frontend
- **Responsive Design** - Works on mobile, tablet, and desktop
- **Gradient UI** - Modern color scheme with primary/secondary colors
- **Animations** - Bouncing header, floating welcome icon, smooth transitions
- **Message Formatting** - Support for bold, italic, and code blocks
- **Auto-scroll** - Messages automatically scroll to view
- **Typing Indicators** - Shows when AI is thinking

### Backend
- **API Integration** - Seamless integration with Longcat API
- **Conversation Memory** - Maintains chat history for context
- **Error Handling** - Graceful error messages
- **CORS Support** - Cross-origin resource sharing enabled
- **Rate Limiting Ready** - Structure supports rate limiting

## Customization

### Change Colors
Edit `:root` variables in `static/css/style.css`:
```css
--primary-color: #6366f1;
--secondary-color: #ec4899;
```

### Change System Prompt
Edit the system message in `app.py`:
```python
"content": "You are an amazing, helpful, and friendly AI assistant..."
```

### Change Model
Edit the model in `app.py`:
```python
"model": "gpt-3.5-turbo"
```

## Browser Support

- Chrome/Edge: Full support
- Firefox: Full support
- Safari: Full support
- Mobile browsers: Full support

## Security Notes

⚠️ **Important:**
- Never commit `.env` file with API keys
- API key should be added to `.gitignore`
- Use environment variables in production
- Consider adding rate limiting
- Add authentication for production use

## Troubleshooting

**Port already in use:**
```bash
python app.py --port 5001
```

**API Error:**
- Check API key is correct
- Check internet connection
- Check Longcat API status

**CSS not loading:**
- Clear browser cache (Ctrl+Shift+R)
- Check if server is running on correct port

## License

This project is open source and available under the MIT License.

## Author

Created with ❤️ using LangChain

---

Enjoy your amazing chatbot! 🚀
