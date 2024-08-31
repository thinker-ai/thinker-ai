document.getElementById('send').addEventListener('click', () => {
  const message = document.getElementById('input').value;
  document.getElementById('input').value = '';
  appendUserMessage(message);

  // 将消息发送到服务器
  chrome.runtime.sendMessage({ action: "sendMessage", content: message }, (response) => {
    appendAIMessage(response);
  });
});

function appendUserMessage(message) {
  const chat = document.getElementById('chat');
  chat.innerHTML += `<div class="message human">${message}</div>`;
  chat.scrollTop = chat.scrollHeight;
}

function appendAIMessage(message) {
  const chat = document.getElementById('chat');
  chat.innerHTML += `<div class="message ai">${message}</div>`;
  chat.scrollTop = chat.scrollHeight;
}