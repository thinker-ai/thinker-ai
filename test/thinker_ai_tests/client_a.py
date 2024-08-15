import asyncio
import websockets
import json
from multiprocessing import Queue as MPQueue


async def client_process(user_id, result_queue: MPQueue):
    uri = f"ws://127.0.0.1:8000/ws/{user_id}"
    print(f"Connecting to WebSocket at {uri}")

    async with websockets.connect(uri) as websocket:
        print(f"Connected to WebSocket as {user_id}")
        while True:
            # 客户端接收消息
            data = await websocket.recv()
            print(f"Client received data: {data}")
            result_queue.put(data)  # 将接收到的数据放入队列
            print("Client process: response received and sent to main process.")


if __name__ == "__main__":
    from multiprocessing import Queue

    user_id = "test_user"
    result_queue = Queue()

    # Run the client process
    asyncio.run(client_process(user_id, result_queue))
