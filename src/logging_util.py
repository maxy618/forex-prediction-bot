import logging
import os
from datetime import datetime
import uuid


def _make_logs_dir():
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs"))
    os.makedirs(base, exist_ok=True)
    return base


def setup_logging(level=logging.INFO, name=None):
    logs_dir = _make_logs_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(logs_dir, f"{timestamp}.log")

    logger = logging.getLogger(name or "app")
    logger.setLevel(level)

    if not logger.handlers:
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(level)
        ch = logging.StreamHandler()
        ch.setLevel(level)

        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        fh.setFormatter(fmt)
        ch.setFormatter(fmt)

        logger.addHandler(fh)
        logger.addHandler(ch)

    logger.debug("Logging initialized, file=%s", log_path)
    return logger


def make_rid():
    return uuid.uuid4().hex[:8]


def exception_rid(logger, message=None, exc=None):
    rid = make_rid()
    if exc is not None:
        logger.exception("RID=%s %s - %s", rid, message or "exception", exc)
    else:
        logger.exception("RID=%s %s", rid, message or "exception")
    return rid
