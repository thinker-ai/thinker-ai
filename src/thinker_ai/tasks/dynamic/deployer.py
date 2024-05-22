import asyncio
import os
from flask import Flask, jsonify
from flask_socketio import SocketIO, emit
import threading

from thinker_ai.main import socket_clients

# 获取当前脚本所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
deploy_directory = os.path.join(current_dir, 'gradio/')
# 确保部署目录存在
if not os.path.exists(deploy_directory):
    os.makedirs(deploy_directory)
# 动态分配端口的起始值
starting_port = 7861
used_ports = set()
def get_free_port():
    global starting_port
    while starting_port in used_ports:
        starting_port += 1
    used_ports.add(starting_port)
    return starting_port


def deploy_service(title, content, user_id):
    file_path = os.path.join(deploy_directory, f"{title}_{user_id}.py")
    with open(file_path, 'w') as file:
        file.write(f"""
import gradio as gr

{content}
""")
    return file_path


def launch_service(title, user_id):
    port = get_free_port()
    file_path = os.path.join(deploy_directory, f"{title}_{user_id}.py")

    def start_service():
        # 读取文件内容并插入启动代码
        with open(file_path, 'r') as file:
            content = file.read()
        content += f"\niface.launch(server_name='0.0.0.0', server_port={port})"

        # 写入临时文件并执行
        temp_file_path = os.path.join(deploy_directory, f"{title}_{user_id}_temp.py")
        with open(temp_file_path, 'w') as temp_file:
            temp_file.write(content)
        os.system(f"python {temp_file_path}")

    thread = threading.Thread(target=start_service)
    thread.start()

    # 通知客户端
    if user_id in socket_clients:
        asyncio.run(socket_clients[user_id].send_json({'title': title, 'port': port}))


def trigger_generation(title, content, user_id):
    file_path, port = deploy_service(title, content, user_id)
    launch_service(file_path, port, user_id)
