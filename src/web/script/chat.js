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


const Speaker
    = {
    mute:true,
    audio:new Audio(),
    init() {
        if (annyang) {
            this.audio.addEventListener('play', () => {
                if(annyang.isListening()){
                    annyang.abort();//否则会继续处理已经语音
                    annyang.pause();
                }
            });
            this.audio.addEventListener('ended', () => {
                if(!annyang.isListening()){
                    annyang.resume();
                }
            });
        }
    },
    playAudio: function(base64Data){
        if(this.mute)
            return;
        // 将 Base64 数据转为 Blob
        let byteCharacters = atob(base64Data);
        let byteNumbers = new Array(byteCharacters.length);
        for (let i = 0; i < byteCharacters.length; i++) {
            byteNumbers[i] = byteCharacters.charCodeAt(i);
        }
        let byteArray = new Uint8Array(byteNumbers);
        let audioBlob = new Blob([byteArray], {type: 'audio/mpeg'});

        this.audio.pause();
        this.audio.src = URL.createObjectURL(audioBlob);
        this.audio.currentTime = 0;
        this.audio.play();
    }
}
Speaker.init();
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

function toggleMute() {
    const muteIcon = document.getElementById('mute-icon');
    if (muteIcon.classList.contains('fa-volume-up')) {
        muteIcon.classList.remove('fa-volume-up');
        muteIcon.classList.add('fa-volume-mute');
        Speaker.mute = true;
    } else {
        muteIcon.classList.remove('fa-volume-mute');
        muteIcon.classList.add('fa-volume-up');
        Speaker.mute = false;
    }
}

const dictation
    = {
      recognition: null,
      dictateButton: document.getElementById('dictate'),
      dictate_tooltip:document.getElementById('dictate-tooltip'),
      continuousDictationCheckbox: document.getElementById('continuous-dictation-checkbox'), // 获取"持续录音"复选框的元素

      init: function() {
                if ('SpeechRecognition' in window) {
                    this.recognition = new SpeechRecognition();
                } else if ('webkitSpeechRecognition' in window) {
                    this.recognition = new webkitSpeechRecognition();
                } else {
                    alert('你的浏览器不支持语音识别.');
                    this.dictateButton.disabled = true;
                    this.dictate_tooltip.disabled = true;
                    return;
                }
                this.recognition.continuous = true;
                this.recognition.interimResults = true;
                this.recognition.lang = 'zh-CN';
                this.recognition.onresult = this.handleRecognitionResult;
            },

        handleRecognitionResult: function(event) {
            console.log('Recognition result:', event.results);  // Add a log statement here
            const intput=document.getElementById('input');
            let existingText = intput.value;
            let newText = Array.from(event.results)
                .filter(result => result.isFinal)
                .map(result => result[0])
                .map(result => result.transcript)
                .join('');
            // Remove leading and trailing whitespace
            newText = newText.trim();
            // Add a space only if there is existing human_data and new human_data
            if (existingText.length > 0 && newText.length > 0) {
                newText = "," + newText;
            }
            intput.value = existingText + newText;
       },
        start: function () {
            console.log('Recognition started');
            this.recognition.start();
            document.getElementById('dictate-tooltip').innerText = '正在录音';
            this.dictateButton.classList.add('blinking');  // 开始闪烁
        },
        stop: function () {
            console.log('Recognition stopped');
            // 在延迟后执行其他操作
            setTimeout(() => {
                this.recognition.stop();
                document.getElementById('dictate-tooltip').innerText = '按住录音';
                this.dictateButton.classList.remove('blinking');  // 停止闪烁
            }, 2000);
        }
  };
dictation.init();  // 初始化语音识别对象
const continuous_dictation
    ={
      sendButton: document.getElementById('send'),
      dictateButton: document.getElementById('dictate'),
      dictate_tooltip:document.getElementById('dictate-tooltip'),
      continuousDictationCheckbox: document.getElementById('continuous-dictation-checkbox'),
      startContinuous: function() {
          if (annyang) {
              this.sendButton.disabled=true;
              this.dictateButton.disabled=true;
              this.dictateButton.classList.add('active');
              this.dictate_tooltip.innerText = '正在录音，说‘发送’发送消息';
              this.dictate_tooltip.classList.add('blinking');  // 开始闪烁
             // Define commands
              let commands = {
                '*text': function(new_text) {
                    let exist_text = this.document.getElementById("input").value;
                    new_text=new_text.trim();
                    if (exist_text.length > 0 && new_text.length > 0) {
                        new_text = "," + new_text;
                    }
                    let end=false;
                    if (new_text.endsWith('发送')) {
                        new_text = new_text.replace(/发送$/, '').trim();
                        new_text = new_text.replace(/,$/, '');
                        end = true;
                    }
                    exist_text+= new_text.trim();
                    this.document.getElementById("input").value=exist_text;
                    if(end){
                        sendMessage();
                    }
                }
             };
            // Add our commands to annyang
             annyang.addCommands(commands);
             annyang.setLanguage('zh-CN');
            // Start listening
             annyang.start({continuous: true});
        }else{
             alert('你的浏览器不支持语音识别.');
        }
    },
    stopContinuous: function() {
         if (annyang) {
             setTimeout(() => {
                      this.sendButton.disabled=false;
                      this.dictateButton.disabled=false;
                      this.dictateButton.classList.remove('active');
                      this.dictate_tooltip.innerText = '按住录音';
                      this.dictate_tooltip.classList.remove('blinking');  // 停止闪烁
                      annyang.abort();
             }, 2000); // 为了让声音转换完成后退出
          }else{
              alert('你的浏览器不支持语音识别.');
          }
    },
    toggleContinuous: function() {
        const checkbox = document.getElementById("continuous-dictation-checkbox");
        if (checkbox.checked) {
            continuous_dictation.startContinuous();
        } else {
            continuous_dictation.stopContinuous();
        }
    }
}

function updateGraphics() {
    const editorContent = document.getElementById("mermaid-editor").value;
    // 配置 Mermaid
    mermaid.initialize({
        startOnLoad: false,
        theme: 'default'  // 你可以选择其他主题
    });
    // 渲染图形
    mermaid.render('mermaid-rendered', editorContent, (svgCode, bindFunctions) => {
        const graphicsContainer = document.getElementById("mermaid-graphics");
        graphicsContainer.innerHTML = svgCode;

        const svgElement = graphicsContainer.querySelector("svg");

        let isDragging = false;
        let prevX = 0, prevY = 0;
        let scale = 1;

        svgElement.addEventListener("mousedown", function(event) {
            isDragging = true;
            prevX = event.clientX;
            prevY = event.clientY;
        });

        window.addEventListener("mousemove", function(event) {
            if (isDragging) {
                let newX = event.clientX;
                let newY = event.clientY;

                const dx = newX - prevX;
                const dy = newY - prevY;

                let currentTransform = svgElement.style.transform;
                const regex = /translate\(([\d\.\-\+]+)px, ([\d\.\-\+]+)px\)/;
                const match = currentTransform.match(regex);

                let x = parseFloat(match ? match[1] : 0);
                let y = parseFloat(match ? match[2] : 0);

                x += dx;
                y += dy;

                svgElement.style.transform = `translate(${x}px, ${y}px) scale(${scale})`;

                prevX = newX;
                prevY = newY;
            }
        });

        window.addEventListener("mouseup", function() {
            isDragging = false;
        });

        svgElement.addEventListener("wheel", function(event) {
            event.preventDefault();
            scale += event.deltaY * -0.001;

            // Restrict scale
            scale = Math.min(Math.max(.125, scale), 4);

            // 获取当前的平移参数
            let currentTransform = svgElement.style.transform;
            const regex = /translate\(([\d\.\-\+]+)px, ([\d\.\-\+]+)px\)/;
            const match = currentTransform.match(regex);

            let x = parseFloat(match ? match[1] : 0);
            let y = parseFloat(match ? match[2] : 0);

            // Apply scale transform
            svgElement.style.transform = `translate(${x}px, ${y}px) scale(${scale})`;
        });

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





