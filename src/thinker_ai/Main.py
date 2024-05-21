import os
from fastapi import FastAPI,WebSocket
from starlette.staticfiles import StaticFiles

from thinker_ai.context import get_project_root
from thinker_ai.ui.Chat import chat_router
socket_clients = set()
app = FastAPI()
root_dir = get_project_root()
# 挂载静态文件目录
app.mount("/static", StaticFiles(directory=os.path.join(root_dir, 'web','static')), name="static")
app.mount("/script", StaticFiles(directory=os.path.join(root_dir, 'web','script')), name="script")
app.mount("/css", StaticFiles(directory= os.path.join(root_dir, 'web','css')), name="css")
# 将 chat_router 挂载到主应用实例上
app.include_router(chat_router)

deploy_directory = "deployed_services"
if not os.path.exists(deploy_directory):
    os.makedirs(deploy_directory)

@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    await websocket.accept()
    socket_clients[user_id] = websocket
    try:
        while True:
            await websocket.receive_text()
    except:
        pass
    finally:
        del socket_clients[user_id]
        await websocket.close()


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)