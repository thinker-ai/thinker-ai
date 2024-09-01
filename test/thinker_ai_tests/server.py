from fastapi import FastAPI
import uvicorn

from thinker_ai.api.web_socket_server import start_background_tasks, socket_router

app = FastAPI()


@app.on_event("startup")
async def startup():
    await start_background_tasks()
    app.include_router(socket_router)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=7000)
