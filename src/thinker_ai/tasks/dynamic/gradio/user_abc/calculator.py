import gradio as gr

# 定义加法、减法、乘法和除法功能

def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

def multiply(a, b):
    return a * b

def divide(a, b):
    if b == 0:
        return "Error: Division by zero"
    return a / b

with gr.Blocks() as calculator:
    gr.Markdown("# 简单计算器")
    with gr.Row():
        with gr.Column():
            num1 = gr.Number(label="数字1")
            num2 = gr.Number(label="数字2")
            operation = gr.Radio(["加", "减", "乘", "除"], label="操作")
            run = gr.Button("运行")
        with gr.Column():
            result = gr.Textbox(label="结果")

    def calculate(num1, num2, operation):
        if operation == "加":
            return add(num1, num2)
        elif operation == "减":
            return subtract(num1, num2)
        elif operation == "乘":
            return multiply(num1, num2)
        elif operation == "除":
            return divide(num1, num2)

    run.click(calculate, [num1, num2, operation], result)

# calculator.queue()
# calculator.launch()