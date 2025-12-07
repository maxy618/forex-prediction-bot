import logging
import os
from datetime import datetime
import uuid
from logging.handlers import TimedRotatingFileHandler


def _make_logs_dir():
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs"))
    os.makedirs(base, exist_ok=True)
    logging.debug("_make_logs_dir returning %s", base)
    return base


def setup_logging(level=logging.INFO, name=None):
    LEVELS = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    logging.debug("setup_logging called level=%s name=%s", level, name)
    if isinstance(level, str):
        level = LEVELS.get(level.upper(), logging.INFO)

    logs_dir = _make_logs_dir()

    date_str = datetime.now().strftime("%Y%m%d")
    log_path = os.path.join(logs_dir, f"{date_str}.log")

    logger = logging.getLogger(name or "app")
    logger.setLevel(level)

    if not logger.handlers:
        fh = TimedRotatingFileHandler(
            log_path,
            when="midnight",
            interval=1,
            backupCount=30,
            encoding="utf-8"
        )
        fh.suffix = "%Y%m%d"
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
    rid = uuid.uuid4().hex[:8]
    logging.debug("make_rid -> %s", rid)
    return rid


def exception_rid(logger, message=None, exc=None):
    rid = make_rid()
    if exc is not None:
        logger.exception("RID=%s %s - %s", rid, message or "exception", exc)
    else:
        logger.exception("RID=%s %s", rid, message or "exception")
    return rid
