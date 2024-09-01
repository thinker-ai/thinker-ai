import json
import asyncio
from fastapi import WebSocket, APIRouter, WebSocketDisconnect
from starlette.websockets import WebSocketState
from typing import Dict

# 创建 API 路由器
socket_router = APIRouter()
tasks = []


class WebsocketService:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'initialized'):  # 确保只初始化一次
            self.message_queue = asyncio.Queue()
            self.socket_clients: Dict[str, WebSocket] = {}
            self.initialized = True

    @classmethod
    async def connect(cls, websocket: WebSocket, user_id: str):
        instance = cls.get_instance()
        try:
            # 检查现有连接是否健康
            if user_id in instance.socket_clients:
                existing_websocket = instance.socket_clients[user_id]
                if existing_websocket.client_state == WebSocketState.CONNECTED:
                    print(f"Existing WebSocket for {user_id} is still connected.close it.")
                    # 可以选择关闭新连接或覆盖旧连接
                    await existing_websocket.close()  # 这里选择关闭旧连接
            await websocket.accept()  # 等待建立连接，而非等待客户端消息
            instance.socket_clients[user_id] = websocket
            process_to_background = instance.process_to_background_message(websocket, user_id)
            process_to_background_task = asyncio.create_task(process_to_background)
            await asyncio.gather(process_to_background_task, return_exceptions=True)
            tasks.append(process_to_background_task)
        except Exception as e:
            print(f"Exception: {e}")
            if user_id in instance.socket_clients:
                del instance.socket_clients[user_id]
                await websocket.close()

    async def process_to_background_message(self, websocket: WebSocket, user_id: str):
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                if message.get("type") == "heartbeat":
                    print(f"Received heartbeat from {user_id}")
                    continue
                print(f"Received message from {user_id}: {message}")
            except WebSocketDisconnect:
                print(f"WebSocket disconnected for {user_id}")
                if user_id in self.socket_clients:
                    del self.socket_clients[user_id]
                    await websocket.close()
            except Exception as e:
                print(f"Error receiving message from {user_id}: {e}")

    async def process_to_front_message(self):
        while True:
            try:
                user_id, message = await self.message_queue.get()
                websocket = self.socket_clients.get(user_id)
                if websocket and websocket.client_state == WebSocketState.CONNECTED:
                    await websocket.send_text(json.dumps(message))
                    self.message_queue.task_done()
                else:
                    if user_id in self.socket_clients:
                        del self.socket_clients[user_id]
                    if websocket and websocket.application_state != WebSocketState.DISCONNECTED:
                        await websocket.close()
            except Exception as e:
                print(f"Error sending message: {e}")

    @classmethod
    async def send_to_front(cls, user_id: str, message: dict):
        print(f"Server received message to send to front：{user_id}")
        instance = cls.get_instance()
        await instance.message_queue.put((user_id, message))

    @classmethod
    async def start_to_front_task(cls):
        instance = cls.get_instance()
        to_front_task = asyncio.create_task(instance.process_to_front_message())
        await asyncio.gather(to_front_task, return_exceptions=True)
        tasks.append(to_front_task)


@socket_router.websocket("/ws/{user_id}")
async def accept_websocket(websocket: WebSocket, user_id: str):
    await WebsocketService.connect(websocket, user_id)


@socket_router.post("/send_to_front/{user_id}")
async def accept_websocket(user_id: str, message: dict):
    await WebsocketService.send_to_front(user_id, message)
