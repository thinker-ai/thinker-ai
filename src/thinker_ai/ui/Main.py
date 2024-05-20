import os

import uvicorn
from fastapi import FastAPI
from starlette.staticfiles import StaticFiles

from thinker_ai.context import get_project_root
from thinker_ai.ui.Chat import chat_router

app = FastAPI()
root_dir = get_project_root()
# 挂载静态文件目录
app.mount("/static", StaticFiles(directory=os.path.join(root_dir, 'web','static')), name="static")
app.mount("/script", StaticFiles(directory=os.path.join(root_dir, 'web','script')), name="script")
app.mount("/css", StaticFiles(directory= os.path.join(root_dir, 'web','css')), name="css")
# 将 chat_router 挂载到主应用实例上
app.include_router(chat_router)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)