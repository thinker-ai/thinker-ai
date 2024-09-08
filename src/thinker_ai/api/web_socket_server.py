import json
import asyncio
from fastapi import WebSocket, APIRouter, WebSocketDisconnect, Depends
from starlette.websockets import WebSocketState
from typing import Dict

from thinker_ai.session_manager import get_session_ws

# 创建 API 路由器
socket_router = APIRouter()
tasks = []
to_background_message_queue = asyncio.Queue()
to_front_message_queue = asyncio.Queue()


class ConnectionManager:
    def __init__(self):
        self.socket_clients: Dict[str, WebSocket] = {}

    async def connect(self, user_id: str, websocket: WebSocket):
        await websocket.accept()
        if user_id in self.socket_clients:
            existing_websocket = self.socket_clients[user_id]
            try:
                if existing_websocket.client_state == WebSocketState.CONNECTED:
                    print(f"Existing WebSocket for {user_id} is still connected. Attempting to close it.")
                    try:
                        await existing_websocket.send_text(json.dumps(
                            {"type": "info", "message": "Your connection is being closed due to a new connection."}))
                        await existing_websocket.close()
                    except Exception as e:
                        print(f"Error notifying or closing existing WebSocket for {user_id}: {e}")
                else:
                    await existing_websocket.close()
            except Exception as e:
                print(f"Error notifying or closing WebSocket for {user_id}: {e}")
        self.socket_clients[user_id] = websocket

    async def disconnect(self, user_id):
        if user_id in self.socket_clients:
            websocket = self.socket_clients.get(user_id)
            if websocket.application_state != WebSocketState.DISCONNECTED:
                try:
                    await websocket.close()
                except Exception as e:
                    print(f"Error closing WebSocket for {user_id}: {e}")
            del self.socket_clients[user_id]

    async def receive_message(self, user_id) -> str:
        websocket = self.socket_clients.get(user_id)
        if websocket:
            return await websocket.receive_text()

    async def send_message(self, user_id: str, message: str):
        websocket = self.socket_clients.get(user_id)
        if websocket:
            try:
                await websocket.send_text(message)
            except WebSocketDisconnect:
                print(f"WebSocket disconnected while sending message to {user_id}")
                await self.disconnect(user_id)
            except Exception as e:
                print(f"Error sending message to {user_id}: {e}")

    async def broadcast(self, message: str):
        for user_id, websocket in self.socket_clients.items():
            try:
                await websocket.send_text(message)
            except WebSocketDisconnect:
                print(f"WebSocket disconnected while broadcasting to {user_id}")
                await self.disconnect(user_id)
            except Exception as e:
                print(f"Error broadcasting message to {user_id}: {e}")
                await self.disconnect(user_id)


manager = ConnectionManager()


@socket_router.websocket("/ws/")
async def connect(websocket: WebSocket):
    session = await get_session_ws(websocket)
    # 根据会话对象获取 user_id 或其他会话信息
    user_id = session.get("user_id")
    if not user_id:
        print(f"user_id {user_id} not found")
        return
    await manager.connect(user_id, websocket)
    try:
        while True:
            data = await manager.receive_message(user_id)
            message = json.loads(data)
            if message.get("type") == "heartbeat":
                print(f"Received heartbeat from {user_id}")
                continue
            await to_background_message_queue.put((user_id, message))
    except WebSocketDisconnect:
        print(f"WebSocket disconnected for {user_id}")
    except Exception as e:
        print(f"Error receiving front message of {user_id}: {e}")
    finally:
        await manager.disconnect(user_id)


async def process_to_background_message():
    while True:
        user_id, message = await to_background_message_queue.get()
        to_background_message_queue.task_done()
        print(f"Received front message of {user_id}: {message}")


@socket_router.post("/send_to_front/{user_id}")
async def send_to_front(user_id: str, message: dict):
    print(f"Received background message of {user_id}: {message}")
    await to_front_message_queue.put((user_id, message))


async def process_to_front_message():
    while True:
        user_id, message = await to_front_message_queue.get()
        try:
            await manager.send_message(user_id, json.dumps(message))
            to_front_message_queue.task_done()
        except WebSocketDisconnect:
            print(f"WebSocket disconnected for {user_id}")
            await manager.disconnect(user_id)
        except Exception as e:
            print(f"Error sending background message of {user_id}: {e}")


async def start_background_tasks():
    return asyncio.gather(
        process_to_front_message(),
        process_to_background_message()
    )
