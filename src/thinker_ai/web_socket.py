import asyncio
import json
from collections import deque

from fastapi import WebSocket, APIRouter
from starlette.websockets import WebSocketState

from thinker_ai.common.common import run_async

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
        if not websocket.application_state == WebSocketState.DISCONNECTED:
            await websocket.close()


# 主动向客户端发送消息的示例函数
def send_message_to_client(user_id: str, message: dict):
    websocket = socket_clients.get(user_id)
    if websocket:
        run_async(websocket.send_text(json.dumps(message, ensure_ascii=False)))
