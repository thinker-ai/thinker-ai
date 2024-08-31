# server.py
import asyncio
from fastapi import FastAPI
import uvicorn

from thinker_ai.api.web_socket_server import socket_router, WebsocketService

app = FastAPI()
app.include_router(socket_router)
websocket_service = WebsocketService.get_instance()


@app.on_event("startup")
async def start_background_tasks():
    await WebsocketService.start_background_tasks()


# server.py (添加以下部分)
@app.post("/send_message/{user_id}")
async def send_message(user_id: str, message: dict):
    print(f"Server received message to send to {user_id}")
    await websocket_service.send_message_to_client(user_id, message)
    return {"status": "message sent"}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
