import threading

from thinker_ai.llm import gpt


def create_session():
    return gpt.llm.beta.threads.create()


class SessionManager:
    def __init__(self):
        self.sessions = {}  # 存储会话
        self.lock = threading.Lock()

    def get_or_create_session(self, user_id):
        with self.lock:
            if user_id not in self.sessions:
                # 创建新会话
                self.sessions[user_id] = create_session()
            return self.sessions[user_id]
