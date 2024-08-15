# server.py
import asyncio
from fastapi import FastAPI
import uvicorn

from thinker_ai.web_socket_server import socket_router, process_message_queue, send_message_to_client

app = FastAPI()
app.include_router(socket_router)


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(process_message_queue())


# server.py (添加以下部分)
@app.post("/send_message/{user_id}")
async def send_message(user_id: str, message: dict):
    print(f"Server received message to send to {user_id}")
    send_message_to_client(user_id, message)
    return {"status": "message sent"}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
