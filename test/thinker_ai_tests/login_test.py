import pytest
from fastapi.testclient import TestClient
from passlib.context import CryptContext
import jwt

from thinker_ai.api.fast_api_instance import app
from thinker_ai.api.login import SECRET_KEY, ALGORITHM

# 假设SECRET_KEY和ALGORITHM在main.py中定义

# 模拟用户数据库
fake_users_db = {
    "testuser": {
        "username": "testuser",
        "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",  # 假设这是密码'testpassword'的哈希值
        "id": 1,
    }
}

# 创建一个测试客户端
client = TestClient(app)

# 密码哈希器
pwd_context = CryptContext(schemes=["bcrypt"])


def test_login_success():
    # 准备数据
    username = "testuser"
    password = "testpassword"
    hashed_password = fake_users_db[username]["hashed_password"]

    # 发送POST请求
    response = client.post(
        "/login",
        data={"username": username, "password": password},
    )

    # 检查响应
    assert response.status_code == 200
    assert "access_token" in response.json()
    assert "token_type" in response.json()

    # 验证JWT token
    access_token = response.json()["access_token"]
    payload = jwt.decode(access_token, SECRET_KEY, algorithms=[ALGORITHM])
    assert payload["sub"] == username


def test_login_failure():
    # 准备数据
    username = "testuser"
    password = "wrongpassword"

    # 发送POST请求
    response = client.post(
        "/login",
        data={"username": username, "password": password},
    )

    # 检查响应
    assert response.status_code == 400
    assert response.json() == {"detail": "Incorrect username or password"}


# 运行测试
if __name__ == "__main__":
    pytest.main(["-v", "test_login.py"])
