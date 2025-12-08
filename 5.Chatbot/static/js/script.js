// DOM Elements
const chatMessages = document.getElementById('chatMessages');
const messageInput = document.getElementById('messageInput');
const chatForm = document.getElementById('chatForm');
const loadingIndicator = document.getElementById('loadingIndicator');

// Send message function
async function sendMessage(event) {
    event.preventDefault();
    
    const message = messageInput.value.trim();
    if (!message) return;
    
    // Add user message to chat
    addMessageToChat(message, 'user');
    messageInput.value = '';
    messageInput.focus();
    
    // Show loading indicator
    showLoading(true);
    
    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ message })
        });
        
        if (!response.ok) {
            throw new Error(`API error: ${response.statusText}`);
        }
        
        const data = await response.json();
        
        if (data.success) {
            addMessageToChat(data.message, 'assistant');
        } else {
            addMessageToChat(`Error: ${data.error}`, 'assistant');
        }
    } catch (error) {
        console.error('Error:', error);
        addMessageToChat(`Sorry, I encountered an error: ${error.message}`, 'assistant');
    } finally {
        showLoading(false);
    }
}

// Add message to chat display
function addMessageToChat(message, sender) {
    // Remove welcome message if it exists
    const welcome = chatMessages.querySelector('.welcome-message');
    if (welcome) {
        welcome.remove();
    }
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}`;
    
    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';
    bubble.innerHTML = formatMessage(message);
    
    messageDiv.appendChild(bubble);
    chatMessages.appendChild(messageDiv);
    
    // Scroll to bottom
    setTimeout(() => {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }, 10);
}

// Format message (basic markdown support)
function formatMessage(text) {
    let formatted = text
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        .replace(/`(.*?)`/g, '<code>$1</code>')
        .replace(/\n/g, '<br>');
    
    return formatted;
}

// Show/hide loading indicator
function showLoading(show) {
    if (show) {
        loadingIndicator.classList.remove('hidden');
    } else {
        loadingIndicator.classList.add('hidden');
    }
}

// Clear chat
async function clearChat() {
    if (confirm('Are you sure you want to clear the chat history?')) {
        try {
            const response = await fetch('/api/clear', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            if (response.ok) {
                chatMessages.innerHTML = `
                    <div class="welcome-message">
                        <div class="welcome-icon">
                            <i class="fas fa-sparkles"></i>
                        </div>
                        <h2>Welcome to LangChain AI Assistant</h2>
                        <p>Ask me anything! I'm here to help with coding, learning, and more.</p>
                    </div>
                `;
                messageInput.focus();
            }
        } catch (error) {
            console.error('Error clearing chat:', error);
        }
    }
}

// Set suggestion as message
function setSuggestion(suggestion) {
    messageInput.value = suggestion;
    messageInput.focus();
}

// Load chat history on page load
async function loadChatHistory() {
    try {
        const response = await fetch('/api/history');
        const data = await response.json();
        
        if (data.conversation_history.length > 0) {
            chatMessages.innerHTML = '';
            data.conversation_history.forEach(msg => {
                addMessageToChat(msg.content, msg.role);
            });
        }
    } catch (error) {
        console.error('Error loading history:', error);
    }
}

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
    // Ctrl/Cmd + Enter to send message
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        if (messageInput.value.trim()) {
            chatForm.dispatchEvent(new Event('submit'));
        }
    }
    
    // Escape to clear input
    if (e.key === 'Escape') {
        messageInput.value = '';
    }
});

// Focus input on page load
window.addEventListener('load', () => {
    loadChatHistory();
    messageInput.focus();
});

// Auto-resize textarea as user types
messageInput.addEventListener('input', () => {
    messageInput.style.height = 'auto';
    messageInput.style.height = Math.min(messageInput.scrollHeight, 120) + 'px';
});

// Prevent sending empty messages
chatForm.addEventListener('submit', (e) => {
    if (!messageInput.value.trim()) {
        e.preventDefault();
    }
});
