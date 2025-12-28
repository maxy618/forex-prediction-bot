import threading


CHAT_STATE = {}
CHAT_LOCKS = {}


def get_chat_lock(chat_id):
    lock = CHAT_LOCKS.get(chat_id)
    if lock is None:
        lock = threading.Lock()
        CHAT_LOCKS[chat_id] = lock
    return lock


def get_chat_state(chat_id):
    return CHAT_STATE.get(
        chat_id,
        {"msg_id": None, "has_logo": False, "awaiting_question": False, "asked_count": 0},
    )


def save_chat_state(chat_id, message_id=None, has_logo=None, **kwargs):
    state = CHAT_STATE.get(chat_id, {})
    if message_id is not None:
        state["msg_id"] = int(message_id)
    if has_logo is not None:
        state["has_logo"] = bool(has_logo)
    state.update(kwargs)
    CHAT_STATE[chat_id] = state