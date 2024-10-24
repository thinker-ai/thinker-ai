from fastapi.routing import APIRoute
from starlette.routing import Mount,Route
from fastapi import FastAPI

#从其它地方import这个全局的app,才能保证加载的服务是存在的
app = FastAPI()
def is_route_mounted(app, path):
    # 外部import的app，它的id和被引用的实例id有可能不同，导致mounted path不存在
    for route in app.routes:
        if (isinstance(route, Mount)
            or isinstance(route, APIRoute)
            or isinstance(route, Route)
        ) and route.path.startswith(path):
            return True
    return False
