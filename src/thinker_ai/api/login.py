from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from thinker_ai.session_manager import SessionManager

login_router = APIRouter()

@login_router.post("/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    # 获取单例实例
    session_manager = SessionManager.get_instance()

    # 清理用户名空格，获取用户信息
    username = form_data.username.strip() if form_data.username else None
    user = session_manager.get_user(username)

    # 验证密码
    if not session_manager.verify_user_password(username, form_data.password):
        raise HTTPException(status_code=400, detail="Incorrect username or password")

    # 生成并保存会话
    access_token = session_manager.add_session(username, user["id"])

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user_id": user["id"]
    }