import os
from typing import List

from openai import OpenAI
import gradio

gr = gradio
api_key = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)


def print_like_dislike(x: gr.LikeData):
    print(x.index, x.value, x.liked)


def add_message(history, message):
    for x in message["files"]:
        history.append(((x,), None))
    if message["text"] is not None:
        history.append((message["text"], None))
    return history, gr.MultimodalTextbox(value=None, interactive=False)


def predict(history: List[List[str]]):
    history_openai_format = []
    if history[-1][0] is not None: history_openai_format.append({"role": "user", "content": history[-1][0]})

    response = client.chat.completions.create(model='gpt-3.5-turbo',
                                              messages=history_openai_format,
                                              temperature=1.0,
                                              stream=True)

    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            if history[-1][1] is None:
                history[-1][1] = ""
            history[-1][1] = history[-1][1] + chunk.choices[0].delta.content
            yield history


with gr.Blocks() as gradio_ui:
    chatbot = gr.Chatbot(
        [],
        elem_id="chatbot",
        bubble_full_width=False
    )

    chat_input = gr.MultimodalTextbox(interactive=True, file_types=["image"],
                                      placeholder="Enter message or upload file...", show_label=False)

    chat_msg = chat_input.submit(add_message, [chatbot, chat_input], [chatbot, chat_input])
    bot_msg = chat_msg.then(predict, chatbot, chatbot, api_name="bot_response")
    bot_msg.then(lambda: gr.MultimodalTextbox(interactive=True), None, [chat_input])

    chatbot.like(print_like_dislike, None, None)

gradio_ui.queue()
gradio_ui.launch()

# gr.ChatInterface(predict).launch()
