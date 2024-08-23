// import * as design from 'design.js';
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
                                    <img class="human_avatar" src="/static/human-avatar.jpg" alt="Human Avatar"></div>`;
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
                        <img class="ai_avatar" src="/static/ai-avatar.jpg" alt="AI Avatar">
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

function sendMessage() {
    const inputField = document.getElementById('input');
    let message = inputField.value;
    inputField.value = '';
    if (message.trim() === '')
        return;
    append_human_message(message);
        // 从 localStorage 中获取 user_id 和 topic
    // 发送消息到服务器
    axios.post('/chat', {
        assistant_name:"assistant_1",
        topic:"default",
        content: message
    }, {
        headers: {
            'Content-Type': 'application/json'
        }
    }).then(response => {
        // 检查响应状态
        if (response.status === 200) {
            // 添加 AI 的消息
            append_ai_message(response.data);
        } else {
            alert(`HTTP error! status: ${response.status}`);
            throw new Error(`HTTP error! status: ${response.status}`);
        }
    }).catch(e => {
        alert(e)
        console.error('Error:', e);
    });
}

let isPanelOpen = true;
let isDragging = false;


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
    const verticalText = document.querySelector(".vertical-text-wrapper"); // 获取vertical-text-wrapper元素
    const panel = document.getElementById("floating-panel");
    let prevX = 0;
    let prevY = 0;
    let panelPosition = { x: 0, y: 100 }; // 初始位置

    verticalText.addEventListener("mousedown", function(event) {
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

const user_id = localStorage.getItem("user_id");
let reconnectInterval = 1000; // 1 second

function connect() {
    if (user_id) {
        const socket = new WebSocket(`ws://localhost:8000/ws/${user_id}`);
        let heartbeatInterval;

        socket.onopen = () => {
            console.log('Connected to server');
            reconnectInterval = 1000; // Reset the interval on successful connection

            // Start sending heartbeat messages every 10 seconds
            heartbeatInterval = setInterval(() => {
                if (socket.readyState === WebSocket.OPEN) {
                    socket.send(JSON.stringify({ type: 'heartbeat' }));
                }
            }, 10000);
        };

        socket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if (data.type !== 'heartbeat') {
                const url = `http://localhost:${data.port}${data.mount_path}`;
                addTabWithUrl(url, data.name);
            }
        };

        socket.onclose = (event) => {
            console.log('Connection closed', event);
            clearInterval(heartbeatInterval); // Stop heartbeat messages
            // Attempt to reconnect after a delay
            setTimeout(connect, reconnectInterval);
            // Increment the interval for each failed attempt
            reconnectInterval = Math.min(reconnectInterval * 2, 5000); // Max 5 seconds
        };

        socket.onerror = (error) => {
            console.log('WebSocket error', error);
        };
    }
}
connect();

function showContent(page, element) {
    // Update the iframe's src attribute to load the corresponding page
    document.getElementById('content-frame').src = page;

    // Update the active state of the menu items
    var menuItems = document.getElementsByClassName('menu-item');
    for (var i = 0; i < menuItems.length; i++) {
        menuItems[i].classList.remove('active');
    }
    element.classList.add('active');
}


// 添加请求拦截器
axios.interceptors.request.use(config => {
    const token = localStorage.getItem('access_token');
    if (token) {
        config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
}, error => {
    return Promise.reject(error);
});
if(!localStorage.getItem('access_token'))
    login();



