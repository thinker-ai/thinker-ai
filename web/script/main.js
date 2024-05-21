document.getElementById('input').addEventListener('keydown', function (event) {
    // 如果按下的是 Enter 键，并且没有同时按下 Shift 键，Alt 键或 Ctrl 键
    if (event.key === 'Enter' && !event.shiftKey && !event.altKey && !event.ctrlKey) {
        // 阻止默认的 Enter 键行为，如提交表单或换行
        event.preventDefault();
        // 发送消息
        sendMessage();
    }
});


function append_human_message(message) {
    const chat = document.getElementById('chat');
    chat.innerHTML += `<div class="message-container human-container"><pre class="human_message">${message}</pre>
                                    <img class="human_avatar" src="/static/human-avatar.jpg" alt="Human Avatar"></div>`;
    chat.scrollTo({
        top: chat.scrollHeight,
        behavior: 'smooth'
    });
}

function append_ai_message(message) {
    message = highlightCode(message); // 对三引号中的代码进行高亮处理
    const chat = document.getElementById('chat');
    chat.innerHTML += `<div class="message-container ai-container">
                        <img class="ai_avatar" src="/static/ai-avatar.jpg" alt="AI Avatar">
                        <div class="ai_message">${message}</div>
                       </div>`;
    chat.scrollTo({
        top: chat.scrollHeight,
        behavior: 'smooth'
    });

    // 重新触发Prism高亮，因为动态添加了代码块
    Prism.highlightAll();
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
function sendMessage() {
    const inputField = document.getElementById('input');
    let message = inputField.value;
    inputField.value = '';
    if (message.trim() === '')
        return;
    append_human_message(message);
    // 发送消息到服务器
    fetch('/chat_to', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            text: message,
        }),
    }).then(response => {
        if (!response.ok) {
            alert(`HTTP error! status: ${response.status}`);
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
    }).then(response => {
        // 添加 AI 的消息
        append_ai_message(response.text);
        // 播放音频
        Speaker.playAudio(response.audio_base64);  // Change 'bytes' to 'audio_base64'
    }).catch(e => {
        alert(e)
        console.error('Error:', e);
    });
}

let isPanelOpen = true;
let isDragging = false;
let panelPosition = { x: 0, y: 100 }; // 初始位置
let prevX = 0, prevY = 0;


function toggleFloatingPanel() {
    const panel = document.getElementById("floating-panel");
    const toggleBtn = document.getElementById("toggle-button");
    const panelContent = document.getElementById("panel-content");

    if (isPanelOpen) {
        panel.style.width = "50px";  // 只有足够的空间来显示 "+" 按钮
        toggleBtn.textContent = "+";
        panelContent.style.display = "none";
    } else {
        panel.style.width = "500px";  // 完整的面板宽度
        toggleBtn.textContent = "-";
        panelContent.style.display = "block";
    }
    panel.style.right = "0px";
    panel.style.top = "100px";
    isPanelOpen = !isPanelOpen;
}

// 在页面加载时调用，以确保面板的初始状态与 isPanelOpen 匹配
window.addEventListener("load", function() {

    const panel = document.getElementById("floating-panel");

    panel.addEventListener("mousedown", function(event) {
        isDragging = true;
        prevX = event.clientX;
        prevY = event.clientY;
    });

    window.addEventListener("mousemove", function(event) {
        if (isDragging && isPanelOpen) {
            let newX = event.clientX;
            let newY = event.clientY;

            const dx = newX - prevX;
            const dy = newY - prevY;

            panelPosition.x -= dx;  // 右侧对齐
            panelPosition.y += dy;

            panel.style.right = `${panelPosition.x}px`;
            panel.style.top = `${panelPosition.y}px`;

            prevX = newX;
            prevY = newY;
        }
    });

    window.addEventListener("mouseup", function() {
        isDragging = false;
    });


    toggleFloatingPanel();  // 设置面板为关闭状态
});

function openTab(event, tabId) {
        const tabs = document.querySelectorAll('.tab');
        const tabContents = document.querySelectorAll('.tab-content');

        tabs.forEach(tab => {
            tab.classList.remove('active');
        });

        tabContents.forEach(content => {
            content.classList.remove('active');
        });

        event.currentTarget.classList.add('active');
        document.getElementById(tabId + '-content').classList.add('active');
    }

let tabCount = 0;
function addTab() {
    const url = prompt("请输入网页地址:");
    if (url) {
        addTabWithUrl(url, `Tab ${tabCount + 1}`);
    }
}

function addTabWithUrl(url, title) {
    tabCount++;
    const newTab = document.createElement('div');
    newTab.className = 'tab';
    newTab.id = 'tab' + tabCount;
    newTab.setAttribute('onclick', `openTab(event, 'tab${tabCount}')`);

    const closeButton = document.createElement('span');
    closeButton.textContent = 'X';
    closeButton.className = 'close-tab';
    closeButton.setAttribute('onclick', `closeTab(event, 'tab${tabCount}')`);

    newTab.textContent = title;
    newTab.appendChild(closeButton);
    document.getElementById('tab-container').appendChild(newTab);

    const newTabContent = document.createElement('div');
    newTabContent.className = 'tab-content';
    newTabContent.id = 'tab' + tabCount + '-content';

    const newIframe = document.createElement('iframe');
    newIframe.className = 'tab-frame';
    newIframe.src = url;
    newTabContent.appendChild(newIframe);

    document.getElementById('container').appendChild(newTabContent);
}
function closeTab(event, tabId) {
    event.stopPropagation();
    const confirmClose = confirm("确定要关闭这个标签页吗？");
    if (confirmClose) {
        const tab = document.getElementById(tabId);
        const tabContent = document.getElementById(tabId + '-content');
        tab.remove();
        tabContent.remove();
    }
}

const user_id = 'some_unique_user_id'; // 这里需要替换为实际的用户ID
const socket = new WebSocket(`ws://localhost:8000/ws/${user_id}`);

socket.onopen = () => {
    console.log('Connected to server');
};

socket.onmessage = (event) => {
    const data = JSON.parse(event.data);
    const port = data.port;
    const url = `http://localhost:${port}`;
    addTabWithUrl(url, data.title);
};




