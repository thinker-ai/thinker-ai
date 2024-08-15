import requests


def send_message(user_id: str):
    url = f"http://127.0.0.1:8000/send_message/{user_id}"
    message = {
        "mount_path": "/tasks/user_abc/calculator",
        "name": "calculator",
        "port": 8000
    }
    response = requests.post(url, json=message)
    print(f"Message sent to server: {message}")
    print(f"Server response: {response.text}")


if __name__ == "__main__":
    send_message("test_user")
