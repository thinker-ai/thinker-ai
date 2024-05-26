import gradio as gr

def slow_echo(message, history):
    import time
    for i in range(len(message)):
        time.sleep(0.3)
        yield "You typed: " + message[: i+1]

demo=gr.ChatInterface(slow_echo)