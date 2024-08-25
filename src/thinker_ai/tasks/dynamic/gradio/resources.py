import asyncio

asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

import gradio as gr

class ResourcesSelector:
    def __init__(self):
        self.available_tools = ["工具1", "工具2", "工具3", "工具4"]
        self.selected_tools = []

    def move_tools(self, available_tools, selected_tools, direction):
        if available_tools is None:
            available_tools = []
        if selected_tools is None:
            selected_tools = []

        if direction == "to_selected":
            for tool in available_tools:
                if tool not in self.selected_tools:
                    self.selected_tools.append(tool)
            self.available_tools = [tool for tool in self.available_tools if tool not in available_tools]
        elif direction == "to_available":
            for tool in selected_tools:
                if tool not in self.available_tools:
                    self.available_tools.append(tool)
            self.selected_tools = [tool for tool in self.selected_tools if tool not in selected_tools]

        return gr.update(choices=self.available_tools, value=[]), gr.update(choices=self.selected_tools, value=[])

# 创建 ResourcesSelector 实例
selector = ResourcesSelector()

with gr.Blocks() as resources:
    with gr.Row():
        # 左侧 CheckboxGroup
        available = gr.CheckboxGroup(choices=selector.available_tools, label="可用工具")

        # 中间的按钮
        with gr.Column():
            move_to_selected = gr.Button("→")
            move_to_available = gr.Button("←")

        # 右侧 CheckboxGroup
        selected = gr.CheckboxGroup(choices=selector.selected_tools, label="备选工具")

    move_to_selected.click(
        selector.move_tools,
        inputs=[available, selected, gr.Textbox(value="to_selected", visible=False)],
        outputs=[available, selected]
    )
    move_to_available.click(
        selector.move_tools,
        inputs=[available, selected, gr.Textbox(value="to_available", visible=False)],
        outputs=[available, selected]
    )