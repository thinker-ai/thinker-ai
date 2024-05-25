import gradio as gr
demo = gr.Interface(lambda s: f"Hello {s}!", "textbox", "textbox")




