import asyncio

import uvicorn

from thinker_ai.app_instance import app
main_loop = asyncio.get_event_loop()

@app.on_event("startup")
async def startup():
    from thinker_ai.tasks.dynamic.service_loader import loader_router
    from thinker_ai_tests.tasks.dynamic.service_loader_test import service_loader_router
    app.include_router(loader_router)
    app.include_router(service_loader_router)

if __name__ == '__main__':
    uvicorn.run("demo_main:app", host="0.0.0.0", port=8000, reload=True)

