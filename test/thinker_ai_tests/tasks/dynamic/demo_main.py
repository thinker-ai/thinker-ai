import asyncio

import uvicorn

from thinker_ai.api.fast_api_instance import app
from thinker_ai.tasks.dynamic.service_loader import ServiceLoader

main_loop = asyncio.get_event_loop()


# 定义依赖函数
def get_service_loader() -> ServiceLoader:
    return ServiceLoader(app, main_loop)


@app.on_event("startup")
async def startup():
    from thinker_ai_tests.tasks.dynamic.ui_loader_test import service_loader_router
    app.include_router(service_loader_router)


if __name__ == '__main__':
    uvicorn.run("demo_main:app", host="0.0.0.0", port=8000, reload=True)
