import os
import pickle
from datetime import datetime, timedelta

import bcrypt
import jwt
from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from starlette import status

login_router = APIRouter()
# 创建 OAuth2 实例，用于解析 token
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
# 会话存储文件路径
SESSION_STORE_FILE = "session_store.pkl"
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


def save_session_store():
    with open(SESSION_STORE_FILE, "wb") as f:
        pickle.dump(session_store, f)


# 假设这里有一个用户数据库
fake_users_db = {
    "testuser": {
        "id": 1,
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


@login_router.post("/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = fake_users_db.get(form_data.username)
    if not user or not bcrypt.checkpw(form_data.password.encode('utf-8'), user.get("hashed_password")):
        raise HTTPException(status_code=400, detail="Incorrect username or password")

    # 生成 JWT token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {"sub": user["username"], "exp": datetime.utcnow() + access_token_expires}
    access_token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
    session_store[access_token] = {"user_id": user["id"]}
    save_session_store()  # 保存会话存储到文件
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user_id": user["id"]
    }
