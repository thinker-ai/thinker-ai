import asyncio
import json
from collections import deque

from fastapi import WebSocket, APIRouter

# 存储待发送的消息
message_queue = deque()
socket_router = APIRouter()
socket_clients = {}


@socket_router.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    await websocket.accept()
    socket_clients[user_id] = websocket
    try:
        while True:
            await websocket.receive_text()
    except Exception as e:
        print(f"Exception: {e}")
    finally:
        del socket_clients[user_id]
        await websocket.close()


# 主动向客户端发送消息的示例函数
async def send_message_to_client(user_id: str, message: dict):
    websocket = socket_clients.get(user_id)
    if websocket:
        await websocket.send_text(json.dumps(message,ensure_ascii=False))


# 后台任务：处理消息队列并发送消息给客户端
async def background_task():
    while True:
        while message_queue:
            user_id, message = message_queue.popleft()
            await send_message_to_client(user_id,message)
        await asyncio.sleep(1)  # 每秒检查一次消息队列


# 发送消息到队列的函数
def enqueue_message(user_id: str, message: dict):
    message_queue.append((user_id, message))
