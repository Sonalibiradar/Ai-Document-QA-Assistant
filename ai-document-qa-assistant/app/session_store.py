"""
Minimal in-memory session store for multi-turn conversations.

This is intentionally simple (a dict in process memory) so the project
stays easy to run locally. In production this would be swapped for Redis
or a database so history survives restarts and works across replicas --
that tradeoff is called out in the README.
"""
from typing import Dict, List
from collections import defaultdict

_sessions: Dict[str, List[Dict[str, str]]] = defaultdict(list)


def get_history(session_id: str) -> List[Dict[str, str]]:
    return _sessions[session_id]


def append_turn(session_id: str, role: str, content: str) -> None:
    _sessions[session_id].append({"role": role, "content": content})


def clear_session(session_id: str) -> None:
    _sessions.pop(session_id, None)
