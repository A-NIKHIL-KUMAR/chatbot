// frontend/script.js

async function sendMessage() {
    const userMessage = document.getElementById('userInput').value.trim();
    if (!userMessage) return;

    const messagesDiv = document.getElementById('messages');
    messagesDiv.innerHTML += `<div class="message user">You: ${userMessage}</div>`;
    messagesDiv.scrollTop = messagesDiv.scrollHeight;

    // Send the message to the backend
    try {
        const response = await fetch('https://chatbot-backend-8ftb.onrender.com/api/chatbot', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: userMessage })
        });

        const data = await response.json();
        if (response.ok) {
            messagesDiv.innerHTML += `<div class="message bot">Inno: ${data.response}</div>`;
        } else {
            messagesDiv.innerHTML += `<div class="message bot">Inno: ${data.response}</div>`;
        }
    } catch (error) {
        messagesDiv.innerHTML += `<div class="message bot">Inno: Sorry, something went wrong.</div>`;
    }

    messagesDiv.scrollTop = messagesDiv.scrollHeight;
    document.getElementById('userInput').value = '';  // Clear input field
}
