document.getElementById('input').addEventListener('keydown', function (event) {
    // 如果按下的是 Enter 键，并且没有同时按下 Shift 键，Alt 键或 Ctrl 键
    if (event.key === 'Enter' && !event.shiftKey && !event.altKey && !event.ctrlKey) {
        // 阻止默认的 Enter 键行为，如提交表单或换行
        event.preventDefault();
        // 发送消息
        sendMessage();
    }
});

function to_inner_html(message) {
    var htmlMessage = marked.parse(message).slice(0, -1);
    if (htmlMessage.endsWith('\n')) {
        htmlMessage = htmlMessage.slice(0, -1);
    }
    return htmlMessage;
}

function append_human_message(message) {
    const chat = document.getElementById('chat');
    const htmlMessage = to_inner_html(message);
    chat.innerHTML += `<div class="message-container human-container"><pre class="human_message">${htmlMessage}</pre>
                                    <img class="human_avatar" src="../images/human-avatar.jpg" alt="Human Avatar"></div>`;
    chat.scrollTo({
        top: chat.scrollHeight,
        behavior: 'smooth'
    });
        // 为生成的图片添加响应式类
    const images = chat.querySelectorAll('img');
    images.forEach(image => {
        image.classList.add('responsive-image');
    });
}

function append_ai_message(message) {
    message = highlightCode(message); // 对三引号中的代码进行高亮处理
    var htmlMessage = to_inner_html(message);
    const chat = document.getElementById('chat');
    chat.innerHTML += `<div class="message-container ai-container">
                        <img class="ai_avatar" src="../images/ai-avatar.jpg" alt="AI Avatar">
                        <div class="ai_message">${htmlMessage}</div>
                       </div>`;
    chat.scrollTo({
        top: chat.scrollHeight,
        behavior: 'smooth'
    });
    // 重新触发Prism高亮，因为动态添加了代码块
    Prism.highlightAll();
    // 为生成的图片添加响应式类
    const images = chat.querySelectorAll('img');
    images.forEach(image => {
        image.classList.add('responsive-image');
    });
}

function highlightCode(message) {
    const regex = /```(\w+)?\n([\s\S]*?)```/gm; // 匹配三引号里的内容以及可选的语言标识
    let match;

    while ((match = regex.exec(message)) !== null) {
        const language = match[1] || 'javascript';
        const codeBlock = match[2].trim();
        // 使用Prism来高亮代码
        const highlightedCode = Prism.highlight(codeBlock, Prism.languages[language], language);
        const formattedCode = `<pre style="font-size: 12px; background-color: black;"><code class='language-${language}'>${highlightedCode}</code></pre>`;
        message = message.replace(match[0], formattedCode);
    }
    return message;
}
// 使用 loadAxios 发送消息
function sendMessage() {
    const inputField = document.getElementById('input');
    let message = inputField.value;
    inputField.value = '';
    if (message.trim() === '') return;

    append_human_message(message);

    requestSender.makeRequest(
        {
            method: 'post',
            url: 'http://0.0.0.0:8000/chat',
            params: {
                assistant_name: "assistant_1",
                topic: "default",
                content: message
            }
        },
        {
            onError: function (error) {
                alert('Error: ' + error.message);
                console.error('Error:', error);
            },
            clientParams: {},
            responseParamsExtractor: function (response) {
                return response.data;  // 假设响应数据在 response.data 中
            }
        }
    );
}
requestSender.registerCallback(function(clientParams, responseParams) {
    // 处理响应数据，填充界面元素1
    append_ai_message(responseParams)
});

document.getElementById('send').addEventListener('click', sendMessage)

// 检查是否已登录
function checkLoginStatus() {
    chrome.runtime.sendMessage({ action: 'getAuthorization'}, (response) => {
        if (response && response.access_token && response.user_id) {
            // 已登录，显示聊天窗口
            document.getElementById('chat-container').style.display = 'block';
            document.getElementById('login-container').style.display = 'none';
        } else {
            // 未登录，显示登录表单
            document.getElementById('chat-container').style.display = 'none';
            document.getElementById('login-container').style.display = 'block';
        }
    });
}

// 处理登录表单提交
document.getElementById('login-form').addEventListener('submit', function(event) {
    event.preventDefault();
    const username = document.getElementById('username').value;
    const password = document.getElementById('password').value;

    // 发送登录信息到 background.js
    chrome.runtime.sendMessage({
        action: 'login',
        username: username,
        password: password
    }, function(response) {
        const loginStatus = document.getElementById('login-status');
        if (response.status === 'success') {
            checkLoginStatus();  // 重新检查登录状态，显示聊天窗口
        } else {
            alert('登录失败，请重试');
        }
    });
});

// 页面加载时检查登录状态
document.addEventListener('DOMContentLoaded', function() {
    checkLoginStatus();
});