// 检测 Enter 键并发送消息
document.getElementById('input')?.addEventListener('keydown', function (event: KeyboardEvent) {
    if (event.key === 'Enter' && !event.shiftKey && !event.altKey && !event.ctrlKey) {
        event.preventDefault();
        sendMessage();
    }
});

// 将消息转换为 HTML
function to_inner_html(message: string): string {
    let htmlMessage = (window as any).marked.parse(message).slice(0, -1);
    if (htmlMessage.endsWith('\n')) {
        htmlMessage = htmlMessage.slice(0, -1);
    }
    return htmlMessage;
}

// 添加用户消息到聊天窗口
function append_human_message(message: string): void {
    const chat = document.getElementById('chat');
    if (!chat) return;

    const htmlMessage = to_inner_html(message);
    chat.innerHTML += `<div class="message-container human-container"><pre class="human_message">${htmlMessage}</pre>
                                    <img class="human_avatar" src="../images/human-avatar.jpg" alt="Human Avatar"></div>`;
    chat.scrollTo({
        top: chat.scrollHeight,
        behavior: 'smooth'
    });

    // 为图片添加响应式类
    const images = chat.querySelectorAll('img');
    images.forEach(image => {
        image.classList.add('responsive-image');
    });
}

// 添加 AI 消息到聊天窗口
function append_ai_message(message: string): void {
    message = highlightCode(message);
    const htmlMessage = to_inner_html(message);
    const chat = document.getElementById('chat');
    if (!chat) return;

    chat.innerHTML += `<div class="message-container ai-container">
                        <img class="ai_avatar" src="../images/ai-avatar.jpg" alt="AI Avatar">
                        <div class="ai_message">${htmlMessage}</div>
                       </div>`;
    chat.scrollTo({
        top: chat.scrollHeight,
        behavior: 'smooth'
    });

    // 重新触发 Prism 高亮
    (window as any).Prism.highlightAll();

    // 为图片添加响应式类
    const images = chat.querySelectorAll('img');
    images.forEach(image => {
        image.classList.add('responsive-image');
    });
}

// 代码高亮处理
function highlightCode(message: string): string {
    const regex = /```(\w+)?\n([\s\S]*?)```/gm;
    let match: RegExpExecArray | null;

    while ((match = regex.exec(message)) !== null) {
        const language = match[1] || 'javascript';
        const codeBlock = match[2].trim();
        const highlightedCode = (window as any).Prism.highlight(codeBlock, (window as any).Prism.languages[language], language);
        const formattedCode = `<pre style="font-size: 12px; background-color: black;"><code class='language-${language}'>${highlightedCode}</code></pre>`;
        message = message.replace(match[0], formattedCode);
    }

    return message;
}

// 发送消息
function sendMessage(): void {
    const inputField = document.getElementById('input') as HTMLInputElement | null;
    if (!inputField) return;

    const message = inputField.value;
    inputField.value = '';

    if (message.trim() === '') return;

    append_human_message(message);

    // 通过 Background Script 发送消息
    chrome.runtime.sendMessage({
        action: 'sendMessage',
        content: message
    }, function (response: { status: string }) {
        if (response && response.status === 'success') {
            console.log('Message sent successfully.');
        } else {
            console.error('Failed to send message.');
        }
    });
}

// 接收来自 Background Script 的消息
chrome.runtime.onMessage.addListener((message: any, sender: chrome.runtime.MessageSender, sendResponse: Function) => {
    if (message.action === 'aiMessage') {
        append_ai_message(message.content);
    }
});

// 点击按钮发送消息
document.getElementById('send')?.addEventListener('click', sendMessage);

// 检查是否已登录
function checkLoginStatus(): void {
    chrome.runtime.sendMessage({ action: 'getAuthorization' }, (response: { access_token: string; user_id: string }) => {
        if (response && response.access_token && response.user_id) {
            document.getElementById('chat-container')!.style.display = 'block';
            document.getElementById('login-container')!.style.display = 'none';
        } else {
            document.getElementById('chat-container')!.style.display = 'none';
            document.getElementById('login-container')!.style.display = 'block';
        }
    });
}

// 处理登录表单提交
document.getElementById('login-form')?.addEventListener('submit', function (event: Event) {
    event.preventDefault();
    const username = (document.getElementById('username') as HTMLInputElement).value;
    const password = (document.getElementById('password') as HTMLInputElement).value;

    chrome.runtime.sendMessage({
        action: 'login',
        username: username,
        password: password
    }, function (response: { status: string }) {
        const loginStatus = document.getElementById('login-status');
        if (response.status === 'success') {
            checkLoginStatus();
        } else {
            alert('登录失败，请重试');
        }
    });
});

// 页面加载时检查登录状态
document.addEventListener('DOMContentLoaded', function () {
    checkLoginStatus();
});