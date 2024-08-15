import json
import asyncio
from fastapi import WebSocket, APIRouter, WebSocketDisconnect
from starlette.websockets import WebSocketState

# 存储待发送的消息
message_queue = asyncio.Queue()
socket_router = APIRouter()
socket_clients = {}


@socket_router.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    await websocket.accept()
    socket_clients[user_id] = websocket

    try:
        # 分离的接收和发送任务
        receive_task = asyncio.create_task(receive_messages(websocket, user_id))
        send_task = asyncio.create_task(send_messages(websocket, user_id))

        # 持续运行接收和发送任务，直到 WebSocket 断开连接或出现异常
        await asyncio.gather(receive_task, send_task, return_exceptions=True)
    except WebSocketDisconnect:
        print(f"WebSocket disconnected for {user_id}")
    except Exception as e:
        print(f"Exception: {e}")
    finally:
        if user_id in socket_clients.keys():
            del socket_clients[user_id]
        if not websocket.application_state == WebSocketState.DISCONNECTED:
            await websocket.close()


async def receive_messages(websocket: WebSocket, user_id: str):
    while True:
        try:
            data = await websocket.receive_text()
            message = json.loads(data)
            if message.get("type") == "heartbeat":
                continue
            print(f"Received message from {user_id}: {message}")
        except WebSocketDisconnect:
            if user_id in socket_clients.keys():
                del socket_clients[user_id]
            print(f"WebSocket disconnected during receive for {user_id}")
            break
        except Exception as e:
            print(f"Error receiving message from {user_id}: {e}")
            break


async def send_messages(websocket: WebSocket, user_id: str):
    while True:
        try:
            user_id, message = await message_queue.get()
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_text(json.dumps(message))
                message_queue.task_done()
            else:
                if user_id in socket_clients.keys():
                    del socket_clients[user_id]
                print(f"WebSocket for {user_id} is not connected")
                break
        except WebSocketDisconnect:
            if user_id in socket_clients.keys():
                del socket_clients[user_id]
            print(f"WebSocket disconnected during send for {user_id}")
            break
        except Exception as e:
            print(f"Error sending message to {user_id}: {e}")
            break


def send_message_to_client(user_id: str, message: dict):
    asyncio.create_task(message_queue.put((user_id, message)))


async def process_message_queue():
    while True:
        try:
            user_id, message = await message_queue.get()
            websocket = socket_clients.get(user_id)
            if websocket and websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_text(json.dumps(message))
            message_queue.task_done()
        except Exception as e:
            print(f"Error processing message queue: {e}")
