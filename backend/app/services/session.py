"""会话状态管理模块 — 支持指代消解和意图继承。"""

import time
import threading
from dataclasses import dataclass, field


SESSION_TTL_SECONDS = 30 * 60  # 30 分钟过期


@dataclass
class SessionState:
    """单个会话状态。"""
    session_id: str = ""
    current_assets: list[str] = field(default_factory=list)
    last_intent: str | None = None
    last_time_window: str | None = None
    last_query_type: str | None = None
    last_active: float = 0.0

    def update(
        self,
        assets: list[str] | None = None,
        intent: str | None = None,
        time_window: str | None = None,
        query_type: str | None = None,
    ):
        """更新会话状态。"""
        if assets is not None:
            self.current_assets = assets
        if intent is not None:
            self.last_intent = intent
        if time_window is not None:
            self.last_time_window = time_window
        if query_type is not None:
            self.last_query_type = query_type
        self.last_active = time.time()

    @property
    def is_expired(self) -> bool:
        if self.last_active == 0:
            return False
        return (time.time() - self.last_active) > SESSION_TTL_SECONDS


class SessionManager:
    """线程安全的会话管理器。"""

    def __init__(self):
        self._sessions: dict[str, SessionState] = {}
        self._lock = threading.Lock()

    def get_or_create(self, session_id: str) -> SessionState:
        """获取或创建会话状态。过期会话会被重置。"""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None or session.is_expired:
                session = SessionState(session_id=session_id, last_active=time.time())
                self._sessions[session_id] = session
            return session

    def get(self, session_id: str) -> SessionState | None:
        """获取会话状态，不创建。"""
        with self._lock:
            session = self._sessions.get(session_id)
            if session and session.is_expired:
                del self._sessions[session_id]
                return None
            return session

    def cleanup_expired(self):
        """清理所有过期会话。"""
        with self._lock:
            expired = [sid for sid, s in self._sessions.items() if s.is_expired]
            for sid in expired:
                del self._sessions[sid]


# 全局单例
session_manager = SessionManager()
