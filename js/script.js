// DanyAIApp\code\js\script.js
document.addEventListener("DOMContentLoaded", function() {
  
  const chatInput = document.getElementById('chat-input');
  const sendButton = document.getElementById('send-btn');
  const loading = document.getElementById('loading');

  // Function to show or hide loading animation
  function toggleLoading(show) {
    loading.style.display = show ? 'block' : 'none';
  }

  // Function to display user message in the chat window
  function displayUserMessage(message) {
    const chatWindow = document.getElementById('chat-window');
    const userDiv = document.createElement('div');
    userDiv.classList.add('user-message');
    userDiv.textContent = message;
    chatWindow.appendChild(userDiv);
    chatWindow.scrollTop = chatWindow.scrollHeight;
  }

  // Function to display AI message in the chat window
  function displayAIMessage(message) {
    const chatWindow = document.getElementById('chat-window');
    const aiDiv = document.createElement('div');
    aiDiv.classList.add('ai-message');
    aiDiv.textContent = message;
    chatWindow.appendChild(aiDiv);
    chatWindow.scrollTop = chatWindow.scrollHeight;
  }

  // Unified function to handle sending messages
  function sendMessage() {
    const message = chatInput.value.trim();
    if (message) {
      displayUserMessage(message);
      toggleLoading(true);
      postChatMessage(message).then(response => {
        displayAIMessage(response);
      }).catch(error => {
        console.error('Error:', error);
        displayAIMessage("Failed to get a response.");
      }).finally(() => {
        toggleLoading(false);
      });
      chatInput.value = '';
    }
  }

  // Attach events for send button and Enter key press
  sendButton.addEventListener('click', sendMessage);
  chatInput.addEventListener('keypress', function(e) {
    if (e.key === 'Enter') {
      e.preventDefault();
      sendMessage();
    }
  });

  // Function to post the chat message to the API
  async function postChatMessage(message) {
    try {
      const response = await fetch('http://127.0.0.1:8000/chat/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query: message })
      });
      const data = await response.json();
      return data.response || 'No response generated';
    } catch (error) {
      console.error('Error:', error);
      throw new Error('Failed to send message');
    }
  }

  // Fine-tune button click event
  document.getElementById('fine-tune-btn').addEventListener('click', function() {
    const fileInput = document.getElementById('fine-tune-file');
    const file = fileInput.files[0];
    if (file) {
      toggleLoading(true);
      fineTuneAI(file).finally(() => {
        toggleLoading(false);
      });
    } else {
      alert('Please select a file to fine-tune the AI.');
    }
  });

  // Function to handle fine-tuning with uploaded file
  async function fineTuneAI(file) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('file_type', file.name.split('.').pop());

    try {
      const response = await fetch('http://127.0.0.1:8000/finetune/finetune', {
        method: 'POST',
        body: formData
      });
      const data = await response.json();
      alert(data.message || 'Fine-tuning complete.');
    } catch (error) {
      console.error('Error:', error);
      alert('Error fine-tuning AI.');
    }
  }

});
