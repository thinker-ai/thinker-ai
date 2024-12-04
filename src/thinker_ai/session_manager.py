import os
import pickle
import bcrypt
import jwt
from datetime import datetime, timedelta
from fastapi import HTTPException, WebSocket, WebSocketDisconnect
from starlette import status
from urllib.parse import urlparse, parse_qs
from thinker_ai.configs.const import PROJECT_ROOT


class SessionManager:
    _instance = None
    _session_store = {}
    _user_db = {}

    # 私有常量
    _SECRET_KEY = os.environ.get("SECRET_KEY", "default_secret_key")
    _ALGORITHM = "HS256"
    _ACCESS_TOKEN_EXPIRE_MINUTES = 30

    SESSION_STORE_FILE = f"{PROJECT_ROOT}/src/thinker_ai/session_store.pkl"

    def __init__(self):
        raise RuntimeError("Use get_instance() to access the SessionManager instance.")

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
            cls._instance._load_session_store()
            cls._instance._initialize_user_db()
        return cls._instance

    def _load_session_store(self):
        """加载会话存储数据"""
        if os.path.exists(self.SESSION_STORE_FILE):
            with open(self.SESSION_STORE_FILE, "rb") as f:
                self._session_store = pickle.load(f)
        else:
            self._session_store = {}

    def _initialize_user_db(self):
        """初始化用户数据库"""
        self._user_db = {
            "testuser": {
                "id": "abc",
                "username": "testuser",
                "full_name": "Test User",
                "hashed_password": self._hash_password("testpassword"),
                "disabled": False,
            }
        }

    def _save_session_store(self):
        """保存会话存储数据到文件"""
        with open(self.SESSION_STORE_FILE, "wb") as f:
            pickle.dump(self._session_store, f)

    def _hash_password(self, password: str) -> str:
        """生成密码的哈希值"""
        salt = bcrypt.gensalt()
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
        return hashed_password

    def verify_user_password(self, username: str, password: str) -> bool:
        """验证用户密码"""
        user = self.get_user(username)
        hashed_password = user["hashed_password"].encode('utf-8')  # 将存储的 str 转回 bytes
        return bcrypt.checkpw(password.encode('utf-8'), hashed_password)

    def add_session(self, username: str, user_id: str) -> str:
        """生成会话并返回 JWT Token"""
        access_token_expires = timedelta(minutes=self._ACCESS_TOKEN_EXPIRE_MINUTES)
        payload = {
            "sub": username,
            "exp": datetime.utcnow() + access_token_expires
        }
        access_token = jwt.encode(payload, self._SECRET_KEY, algorithm=self._ALGORITHM)
        self._session_store[access_token.encode('utf-8')] = {"user_id": user_id}
        self._save_session_store()
        return access_token

    async def get_session(self, token: str) -> dict:
        """获取会话数据"""
        token_bytes = token.encode('utf-8')
        session = self._session_store.get(token_bytes)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return session

    async def get_session_ws(self, websocket: WebSocket) -> dict:
        """通过 WebSocket 获取会话数据"""
        query_params = parse_qs(urlparse(str(websocket.url)).query)
        token = query_params.get('token', [None])[0]
        if not token:
            print("Token not found")
            raise WebSocketDisconnect(code=4001)

        token_bytes = token.encode('utf-8')
        session = self._session_store.get(token_bytes)
        if not session:
            print("Session not found for token")
            raise WebSocketDisconnect(code=4001)

        return session

    def remove_session(self, token: str):
        """移除会话数据"""
        token_bytes = token.encode('utf-8')
        if token_bytes in self._session_store:
            del self._session_store[token_bytes]
            self._save_session_store()

    def get_user(self, username: str) -> dict:
        """通过用户名获取用户信息"""
        user = self._user_db.get(username)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found",
            )
        return user