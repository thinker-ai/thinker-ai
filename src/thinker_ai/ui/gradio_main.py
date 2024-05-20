import gradio as gr
import random
from typing import List

# 模拟AI响应函数
def ai_response(message):
    responses = [
        "Sure, I can help with that.",
        "Let's get started!",
        "Please wait a moment...",
        "Adding a new tab for you..."
    ]
    return random.choice(responses)
def chat_with_ai(history, message):
    history.append(("User", message))
    response = ai_response(message)
    history.append(("AI", response))
    return history, ""

# 创建聊天界面的 Gradio Blocks
chat_blocks = gr.Blocks()
with chat_blocks:
    chatbot = gr.Chatbot([], elem_id="chatbot")
    chat_input = gr.Textbox(placeholder="Enter your message...")
    chat_submit = gr.Button("Send")
    chat_input.submit(chat_with_ai, [chatbot, chat_input], [chatbot, chat_input])
    chat_submit.click(chat_with_ai, [chatbot, chat_input], [chatbot, chat_input])

# 创建主页面的 Gradio Blocks
with gr.Blocks() as demo:
    # 主页面的标签页区域
    tabs = gr.Tabs()

    # 悬浮窗口的 HTML 和 JavaScript
    gr.HTML("""
        <style>
            #floating-window {
                position: fixed;
                bottom: 10px;
                right: 10px;
                width: 300px;
                height: 400px;
                overflow: auto;
                background: white;
                border: 1px solid black;
                padding: 10px;
                resize: both;
                z-index: 1000;
            }
            #floating-header {
                padding: 10px;
                cursor: move;
                background: #f1f1f1;
                border-bottom: 1px solid #d3d3d3;
            }
        </style>
        <div id="floating-window">
            <div id="floating-header">Chat with AI</div>
            <div id="chat-container">
                <!-- 聊天界面的 Gradio Blocks 会被插入到这里 -->
            </div>
        </div>
        <script>
            function dragElement(elmnt) {
                // ... (existing code)
            }
            dragElement(document.getElementById("floating-window"));

            document.addEventListener("DOMContentLoaded", function() {
                const chatContainer = document.getElementById("chat-container");
                const chatbot = document.querySelector(".gradio-container");
                if (chatContainer && chatbot) {
                    chatContainer.appendChild(chatbot);
                }
            });
        </script>
    """)

# 启动 Gradio 应用
demo.launch()
