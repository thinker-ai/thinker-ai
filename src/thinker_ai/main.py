import os
import uvicorn
from fastapi import FastAPI, WebSocket
from starlette.staticfiles import StaticFiles

from thinker_ai.chat import chat_router
from thinker_ai.login import login_router
from thinker_ai.context import get_project_root

app = FastAPI()

root_dir = get_project_root()

socket_clients = {}
# 挂载静态文件目录
app.mount("/static", StaticFiles(directory=os.path.join(root_dir, 'web', 'static')), name="static")
app.mount("/script", StaticFiles(directory=os.path.join(root_dir, 'web', 'script')), name="script")
app.mount("/css", StaticFiles(directory=os.path.join(root_dir, 'web', 'css')), name="css")


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

##不能在该文件之外执行include_router操作，因为当前文件不感知其它文件，导致include_router不会执行，
##另外，include_router操作必须在所有router的地址映射完成之后执行
app.include_router(login_router)
app.include_router(chat_router)


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)
