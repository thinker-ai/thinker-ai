from datetime import datetime, timedelta

import bcrypt
import jwt
from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm

from thinker_ai.session_manager import fake_users_db, session_store, ACCESS_TOKEN_EXPIRE_MINUTES, SECRET_KEY, ALGORITHM, \
    save_session_store

login_router = APIRouter()


@login_router.post("/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    username=form_data.username.strip() if form_data.username is not None else None
    user = fake_users_db.get(username)
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
