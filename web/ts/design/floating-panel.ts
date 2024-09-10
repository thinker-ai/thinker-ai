import { makeRequest, registerCallbackWithKey, run_after_plugin_checked,send_websocket_message } from "../common";
declare var marked: {
    parse: (markdown: string) => string;
};

declare var Prism: {
    highlight: (code: string, grammar: any, language: string) => string;
    languages: { [key: string]: any };
    highlightAll: () => void;
};
function to_inner_html(message: string): string {
    let htmlMessage = marked.parse(message).slice(0, -1);
    if (htmlMessage.endsWith('\n')) {
        htmlMessage = htmlMessage.slice(0, -1);
    }
    return htmlMessage;
}

function append_human_message(message: string): void {
    const chat = document.getElementById('chat') as HTMLElement;
    const htmlMessage = to_inner_html(message);
    chat.innerHTML += `<div class="message-container human-container"><pre class="human_message">${htmlMessage}</pre>
                                    <img class="human_avatar" src="../../static/design/human-avatar.jpg" alt="Human Avatar"></div>`;
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

function append_ai_message(message: string): void {
    message = highlightCode(message); // 对三引号中的代码进行高亮处理
    const htmlMessage = to_inner_html(message);
    const chat = document.getElementById('chat') as HTMLElement;
    chat.innerHTML += `<div class="message-container ai-container">
                        <img class="ai_avatar" src="../../static/design/ai-avatar.jpg" alt="AI Avatar">
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

function highlightCode(message: string): string {
    const regex = /```(\w+)?\n([\s\S]*?)```/gm; // 匹配三引号里的内容以及可选的语言标识
    let match: RegExpExecArray | null;

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

function sendMessage(): void {
    const inputField = document.getElementById('input') as HTMLTextAreaElement;
    let message = inputField.value;
    inputField.value = '';
    if (message.trim() === '') return;
    append_human_message(message);
    makeRequest(
        'post',
        '/chat',
        undefined,
        {
              assistant_name:"assistant_1",
              topic:"default",
              content:message
            },
        true,
        "application/json",
        (response_data) => append_ai_message(response_data),
        (error) => {
            alert(error);
            console.error(error);
        }
    );
}

let isDragging = false;
let isPanelOpen = true;

function toggleFloatingPanel(): void {
    const panel = document.getElementById("floating-panel") as HTMLElement;
    const toggleBtn = document.getElementById("toggle-button") as HTMLElement;
    const panelContent = document.getElementById("panel-content") as HTMLElement;

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
    panel.style.top = "0px";
    isPanelOpen = !isPanelOpen;
}

function initializeFloatingPanel(contentDiv: HTMLElement): void {
    const prismScript = document.createElement('script');
    prismScript.type = 'text/javascript';
    prismScript.src = 'https://cdn.jsdelivr.net/npm/prismjs@1.25.0/prism.js';
    document.head.appendChild(prismScript);

    const markedScript = document.createElement('script');
    markedScript.type = 'text/javascript';
    markedScript.src = 'https://cdn.jsdelivr.net/npm/marked/marked.min.js';
    document.head.appendChild(markedScript);

    const floatingPanelHTML = `
        <div id="floating-panel">
            <div id="chat-container">
                <button id="toggle-button">-</button>
                <div id="sidebar">
                    <div class="vertical-text-wrapper">
                        <span class="vertical-text">聊天窗口</span>
                    </div>
                </div>
                <div id="panel-content">
                    <div id="chat"></div>
                    <div id="input-container">
                        <textarea id="input"></textarea>
                        <div id="button-container">
                            <button id="send">
                                <i class="fas fa-paper-plane">发送</i>
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `;
    contentDiv.insertAdjacentHTML('beforeend', floatingPanelHTML);

    // 使用 addEventListener 来添加事件处理
    document.getElementById('toggle-button')?.addEventListener('click', toggleFloatingPanel);
    document.getElementById('sidebar')?.addEventListener('click', toggleFloatingPanel);
    document.getElementById('send')?.addEventListener('click', sendMessage);

    const cssLink = document.createElement('link');
    cssLink.rel = 'stylesheet';
    cssLink.type = 'text/css';
    cssLink.href = '/static/design/floating-panel.css';
    document.head.appendChild(cssLink);

    document.getElementById('input')?.addEventListener('keydown', function (event: KeyboardEvent) {
        // 如果按下的是 Enter 键，并且没有同时按下 Shift 键，Alt 键或 Ctrl 键
        if (event.key === 'Enter' && !event.shiftKey && !event.altKey && !event.ctrlKey) {
            // 阻止默认的 Enter 键行为，如提交表单或换行
            event.preventDefault();
            // 发送消息
            sendMessage();
        }
    });

    const verticalText = document.querySelector(".vertical-text-wrapper") as HTMLElement; // 获取 vertical-text-wrapper 元素
    const panel = document.getElementById("floating-panel") as HTMLElement;
    let prevX = 0;
    let prevY = 0;
    let panelPosition = { x: 0, y: 100 }; // 初始位置

    verticalText?.addEventListener("mousedown", function(event: MouseEvent) {
        isDragging = true;
        prevX = event.clientX;
        prevY = event.clientY;
    });

    window.addEventListener("mousemove", function(event: MouseEvent) {
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
}

export function initialize_floating_panel_if_extension_not_install(contentDiv: HTMLElement): void {
    run_after_plugin_checked(
        undefined, // 第一个参数可选，设为 undefined
        () => initializeFloatingPanel(contentDiv)
    );
}