// Global state
let currentTab = 'youtube';
let chatHistory = {
    youtube: [],
    documents: [],
    general: []
};

// Initialize
document.addEventListener('DOMContentLoaded', function() {
    checkStatus();
    setupTabs();
    setupYouTube();
    setupDocuments();
    setupGeneralChat();
    setupClearButton();
    
    // Check status every 30 seconds
    setInterval(checkStatus, 30000);
});

// Check Ollama status
async function checkStatus() {
    try {
        const response = await fetch('/api/status');
        const data = await response.json();
        
        const indicator = document.querySelector('.status-dot');
        const statusText = document.getElementById('statusText');
        
        if (data.ollama_running) {
            indicator.classList.add('online');
            statusText.textContent = 'Ollama Online';
        } else {
            indicator.classList.remove('online');
            statusText.textContent = 'Ollama Offline - Run: ollama serve';
        }
    } catch (error) {
        console.error('Status check failed:', error);
    }
}

// Tab switching
function setupTabs() {
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const tab = btn.dataset.tab;
            
            // Update buttons
            tabBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            
            // Update content
            tabContents.forEach(c => c.classList.remove('active'));
            document.getElementById(`${tab}-tab`).classList.add('active');
            
            currentTab = tab;
        });
    });
}

// YouTube functionality
function setupYouTube() {
    const loadBtn = document.getElementById('loadVideoBtn');
    const urlInput = document.getElementById('youtubeUrl');
    const sendBtn = document.getElementById('youtubeSendBtn');
    const questionInput = document.getElementById('youtubeQuestion');
    
    loadBtn.addEventListener('click', loadYouTubeVideo);
    urlInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') loadYouTubeVideo();
    });
    
    sendBtn.addEventListener('click', sendYouTubeQuestion);
    questionInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendYouTubeQuestion();
    });
}

async function loadYouTubeVideo() {
    const url = document.getElementById('youtubeUrl').value;
    if (!url) {
        showError('youtube', 'Please enter a YouTube URL');
        return;
    }
    
    const btn = document.getElementById('loadVideoBtn');
    btn.disabled = true;
    btn.innerHTML = '<span class="loading"></span> Loading...';
    
    try {
        const response = await fetch('/api/youtube/load', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url })
        });
        
        const data = await response.json();
        
        if (data.success) {
            document.getElementById('videoTitle').textContent = data.title;
            document.getElementById('videoStats').textContent = 
                `${data.chunks_created} chunks created from ${data.transcript_length} characters`;
            document.getElementById('videoInfo').style.display = 'block';
            showSuccess('youtube', data.message);
        } else {
            showError('youtube', data.error || 'Failed to load video');
        }
    } catch (error) {
        showError('youtube', `Error: ${error.message}`);
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<i class="fas fa-download"></i> Load Video';
    }
}

async function sendYouTubeQuestion() {
    const question = document.getElementById('youtubeQuestion').value;
    if (!question) return;
    
    addMessage('youtube', 'user', question);
    document.getElementById('youtubeQuestion').value = '';
    
    const sendBtn = document.getElementById('youtubeSendBtn');
    sendBtn.disabled = true;
    sendBtn.innerHTML = '<span class="loading"></span>';
    
    try {
        const response = await fetch('/api/youtube/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question })
        });
        
        const data = await response.json();
        
        if (data.success) {
            let answer = data.answer;
            if (data.sources && data.sources.length > 0) {
                answer += '\n\nSources:\n';
                data.sources.forEach((source, i) => {
                    answer += `${i + 1}. ${source.content}\n`;
                });
            }
            addMessage('youtube', 'bot', answer);
        } else {
            addMessage('youtube', 'bot', `Error: ${data.error}`);
        }
    } catch (error) {
        addMessage('youtube', 'bot', `Error: ${error.message}`);
    } finally {
        sendBtn.disabled = false;
        sendBtn.innerHTML = '<i class="fas fa-paper-plane"></i>';
    }
}

// Documents functionality
function setupDocuments() {
    const fileInput = document.getElementById('fileInput');
    const sendBtn = document.getElementById('documentSendBtn');
    const questionInput = document.getElementById('documentQuestion');
    
    fileInput.addEventListener('change', handleFileUpload);
    
    // Drag and drop
    const uploadLabel = document.querySelector('.upload-label');
    uploadLabel.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadLabel.style.borderColor = '#6366f1';
    });
    uploadLabel.addEventListener('dragleave', () => {
        uploadLabel.style.borderColor = '#e5e7eb';
    });
    uploadLabel.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadLabel.style.borderColor = '#e5e7eb';
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            fileInput.files = files;
            handleFileUpload();
        }
    });
    
    sendBtn.addEventListener('click', sendDocumentQuestion);
    questionInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendDocumentQuestion();
    });
}

async function handleFileUpload() {
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];
    if (!file) return;
    
    const formData = new FormData();
    formData.append('file', file);
    
    const uploadLabel = document.querySelector('.upload-label');
    uploadLabel.innerHTML = '<span class="loading"></span> Uploading...';
    
    try {
        const response = await fetch('/api/documents/upload', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            addDocumentToList(data.file_id, data.filename, data.chunks_created);
            showSuccess('documents', data.message);
        } else {
            showError('documents', data.error || 'Failed to upload document');
        }
    } catch (error) {
        showError('documents', `Error: ${error.message}`);
    } finally {
        uploadLabel.innerHTML = `
            <i class="fas fa-cloud-upload-alt"></i>
            <span>Click to upload or drag and drop</span>
            <small>PDF, TXT, MD files supported</small>
        `;
        fileInput.value = '';
    }
}

function addDocumentToList(fileId, filename, chunks) {
    const select = document.getElementById('documentSelect');
    const option = document.createElement('option');
    option.value = fileId;
    option.textContent = `${filename} (${chunks} chunks)`;
    select.appendChild(option);
    
    const list = document.getElementById('documentList');
    const item = document.createElement('div');
    item.className = 'document-item';
    item.innerHTML = `
        <div>
            <strong>${filename}</strong>
            <small style="display: block; color: #6b7280;">${chunks} chunks</small>
        </div>
    `;
    list.appendChild(item);
}

async function sendDocumentQuestion() {
    const question = document.getElementById('documentQuestion').value;
    const fileId = document.getElementById('documentSelect').value;
    
    if (!question) return;
    if (!fileId) {
        showError('documents', 'Please select a document first');
        return;
    }
    
    addMessage('documents', 'user', question);
    document.getElementById('documentQuestion').value = '';
    
    const sendBtn = document.getElementById('documentSendBtn');
    sendBtn.disabled = true;
    sendBtn.innerHTML = '<span class="loading"></span>';
    
    try {
        const response = await fetch('/api/documents/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question, file_id: fileId })
        });
        
        const data = await response.json();
        
        if (data.success) {
            let answer = data.answer;
            if (data.sources && data.sources.length > 0) {
                answer += '\n\nSources:\n';
                data.sources.forEach((source, i) => {
                    answer += `${i + 1}. ${source.content}\n`;
                });
            }
            addMessage('documents', 'bot', answer);
        } else {
            addMessage('documents', 'bot', `Error: ${data.error}`);
        }
    } catch (error) {
        addMessage('documents', 'bot', `Error: ${error.message}`);
    } finally {
        sendBtn.disabled = false;
        sendBtn.innerHTML = '<i class="fas fa-paper-plane"></i>';
    }
}

// General chat functionality
function setupGeneralChat() {
    const sendBtn = document.getElementById('generalSendBtn');
    const messageInput = document.getElementById('generalMessage');
    
    sendBtn.addEventListener('click', sendGeneralMessage);
    messageInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendGeneralMessage();
    });
}

async function sendGeneralMessage() {
    const message = document.getElementById('generalMessage').value;
    if (!message) return;
    
    addMessage('general', 'user', message);
    document.getElementById('generalMessage').value = '';
    
    chatHistory.general.push({ role: 'user', content: message });
    
    const sendBtn = document.getElementById('generalSendBtn');
    sendBtn.disabled = true;
    sendBtn.innerHTML = '<span class="loading"></span>';
    
    try {
        const response = await fetch('/api/chat/general', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                message,
                history: chatHistory.general
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            addMessage('general', 'bot', data.response);
            chatHistory.general.push({ role: 'assistant', content: data.response });
        } else {
            addMessage('general', 'bot', `Error: ${data.error}`);
        }
    } catch (error) {
        addMessage('general', 'bot', `Error: ${error.message}`);
    } finally {
        sendBtn.disabled = false;
        sendBtn.innerHTML = '<i class="fas fa-paper-plane"></i>';
    }
}

// Utility functions
function addMessage(tab, role, content) {
    const messagesDiv = document.getElementById(`${tab}Messages`);
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}-message`;
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    // Format content with line breaks
    const paragraphs = content.split('\n');
    paragraphs.forEach(p => {
        const para = document.createElement('p');
        para.textContent = p;
        contentDiv.appendChild(para);
    });
    
    messageDiv.appendChild(contentDiv);
    messagesDiv.appendChild(messageDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

function showError(tab, message) {
    const container = document.getElementById(`${tab}-tab`);
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error';
    errorDiv.textContent = message;
    container.insertBefore(errorDiv, container.firstChild);
    setTimeout(() => errorDiv.remove(), 5000);
}

function showSuccess(tab, message) {
    const container = document.getElementById(`${tab}-tab`);
    const successDiv = document.createElement('div');
    successDiv.className = 'success';
    successDiv.textContent = message;
    container.insertBefore(successDiv, container.firstChild);
    setTimeout(() => successDiv.remove(), 5000);
}

function setupClearButton() {
    document.getElementById('clearBtn').addEventListener('click', async () => {
        if (confirm('Clear all conversations and data?')) {
            try {
                await fetch('/api/clear', { method: 'POST' });
                location.reload();
            } catch (error) {
                alert('Error clearing data');
            }
        }
    });
}


