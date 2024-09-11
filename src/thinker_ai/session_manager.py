import os
import pickle

import bcrypt
from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from starlette import status
from fastapi import WebSocket, WebSocketDisconnect
from fastapi import Request
from thinker_ai.configs.const import PROJECT_ROOT


# async def debug_oauth2_scheme(request: Request):
#     authorization: str = request.headers.get("Authorization")
#     if authorization is None or not authorization.startswith("Bearer "):
#         raise HTTPException(
#             status_code=status.HTTP_401_UNAUTHORIZED,
#             detail="Invalid authentication credentials",
#             headers={"WWW-Authenticate": "Bearer"},
#         )
#     token = authorization.split(" ")[1]
#     print(f"Token extracted: {token}")  # 输出 token 进行调试
#     return token


# 创建 OAuth2 实例，用于解析 token
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
# 会话存储文件路径
SESSION_STORE_FILE = f"{PROJECT_ROOT}/src/thinker_ai/session_store.pkl"
# 读取会话存储
if os.path.exists(SESSION_STORE_FILE):
    with open(SESSION_STORE_FILE, "rb") as f:
        session_store = pickle.load(f)
else:
    session_store = {}


# 依赖项，用于解析 token 并获取会话对象
def get_session(token: str = Depends(oauth2_scheme)) -> dict:
    token_bytes = token.encode('utf-8')
    session = session_store.get(token_bytes)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return session


# 针对 WebSocket 连接获取 token 和会话对象
async def get_session_ws(websocket: WebSocket) -> dict:
    from urllib.parse import urlparse, parse_qs
    query_params = parse_qs(urlparse(str(websocket.url)).query)

    token = query_params.get('token', [None])[0]
    if not token:
        print("Token not found")
        raise WebSocketDisconnect(code=4001)  # 自定义错误码

    # 假设 session_store 是存储会话信息的字典
    token_bytes = token.encode('utf-8')
    session = session_store.get(token_bytes)

    if not session:
        print("Session not found for token")
        raise WebSocketDisconnect(code=4001)  # 自定义错误码

    return session


def save_session_store():
    with open(SESSION_STORE_FILE, "wb") as f:
        pickle.dump(session_store, f)


# 假设这里有一个用户数据库
fake_users_db = {
    "testuser": {
        "id": "abc",
        "username": "testuser",
        "full_name": "Test User",
        "hashed_password": bcrypt.hashpw("testpassword".encode('utf-8'), bcrypt.gensalt()),
        "disabled": False,
    }
}

# 定义一些常量
SECRET_KEY = os.environ.get("SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30


def hash_password(password: str):
    # 生成一个随机的 salt
    salt = bcrypt.gensalt()
    # 使用 salt 对密码进行哈希处理
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed_password
