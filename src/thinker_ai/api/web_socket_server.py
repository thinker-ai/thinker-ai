import json
import asyncio
from fastapi import WebSocket, APIRouter, WebSocketDisconnect
from starlette.websockets import WebSocketState
from typing import Dict

# 创建 API 路由器
socket_router = APIRouter()


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

    async def startup(self):
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
    @socket_router.websocket("/ws/{user_id}")
    async def accept_websocket(cls, websocket: WebSocket, user_id: str):
        instance = cls.get_instance()
        try:
            await websocket.accept()  # 等待建立连接，而非等待客户端消息
            instance.socket_clients[user_id] = websocket
            receive_task = asyncio.create_task(instance.receive_messages(websocket, user_id))
            await receive_task
        except WebSocketDisconnect:
            print(f"WebSocket disconnected for {user_id}")
        except Exception as e:
            print(f"Exception: {e}")
        finally:
            if user_id in instance.socket_clients:
                del instance.socket_clients[user_id]
            if websocket and websocket.application_state != WebSocketState.DISCONNECTED:
                await websocket.close()

    async def receive_messages(self, websocket: WebSocket, user_id: str):
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                if message.get("type") == "heartbeat":
                    continue
                print(f"Received message from {user_id}: {message}")
            except WebSocketDisconnect:
                print(f"WebSocket disconnected for {user_id}")
                break
            except Exception as e:
                print(f"Error receiving message from {user_id}: {e}")
                break
            finally:
                if user_id in self.socket_clients:
                    del self.socket_clients[user_id]
                if websocket and websocket.application_state != WebSocketState.DISCONNECTED:
                    await websocket.close()

    @classmethod
    async def send_message_to_client(cls, user_id: str, message: dict):
        instance = cls.get_instance()
        await instance.message_queue.put((user_id, message))

    @classmethod
    async def start_background_tasks(cls):
        instance = cls.get_instance()
        await instance.startup()
