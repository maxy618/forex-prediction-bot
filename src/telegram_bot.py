import os
import time
import threading
import json
from datetime import date
from concurrent.futures import ThreadPoolExecutor
from logging_util import setup_logging, exception_rid

logger = setup_logging(level="debug", name=__name__)

from telegram import (
    Bot,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    InputMediaPhoto,
    InputMediaAnimation,
)
from telegram.error import BadRequest
from telegram.ext import Updater, CommandHandler, CallbackQueryHandler, MessageHandler, Filters
from telegram.utils.request import Request

from parser import SESSION, HTTP_POOL_SIZE, fetch_sequences_all_pairs
from plotter import plot_sequence, make_forecast_gif
from model_engine import load_model, forecast_diffs, forecast_signs


TELEGRAM_TOKEN = None
TEMP_FOLDER = None
LOGO_PATH = None
ASK_IMG_PATH = None
AI_THINKING_PATH = None
PREDICTING_PATH = None
MODELS_PATH = None
MODELS_SETTINGS = None
CURRENCIES = None
user_interface = None
CACHE_TTL = 300
GEMINI_API_KEY = None
GEMINI_MODEL = None
GEMINI_URL = None
PROMPT_TEMPLATE = None

CHAT_STATE = {}
CHAT_LOCKS = {}

EXECUTOR = None


def _temp_file(uid, ext="png"):
    os.makedirs(TEMP_FOLDER, exist_ok=True)
    return os.path.join(TEMP_FOLDER, f"{uid}_{int(time.time() * 1000)}.{ext}")


def _get_chat_lock(chat_id):
    lock = CHAT_LOCKS.get(chat_id)
    if lock is None:
        lock = threading.Lock()
        CHAT_LOCKS[chat_id] = lock
    return lock


def _markup_repr(reply_markup):
    if not reply_markup:
        return None
    try:
        rows = []
        for row in getattr(reply_markup, "inline_keyboard", []):
            cols = []
            for b in row:
                cols.append((getattr(b, "text", None), getattr(b, "callback_data", None)))
            rows.append(tuple(cols))
        return tuple(rows)
    except Exception:
        return str(reply_markup)


def _make_rows(pairs):
    keyboard = []
    for row in pairs:
        keyboard.append([InlineKeyboardButton(text, callback_data=cb) for text, cb in row])
    return InlineKeyboardMarkup(keyboard)


def _protected_assets():
    return {p for p in (LOGO_PATH, ASK_IMG_PATH, AI_THINKING_PATH, PREDICTING_PATH) if p}


def _is_protected_asset(path):
    return isinstance(path, str) and path in _protected_assets()


def _is_generated_media(path):
    try:
        if not isinstance(path, str):
            return False
        if not TEMP_FOLDER:
            return False
        abs_temp = os.path.abspath(TEMP_FOLDER)
        abs_path = os.path.abspath(path)
        return abs_path.startswith(abs_temp)
    except Exception:
        return False


def _kb_first():
    codes = user_interface["buttons"].get("currency_codes", CURRENCIES)
    n = len(codes) if codes else 0
    rows_cnt = 1 if n <= 3 else 2 if n <= 6 else 3
    chunk = (n + rows_cnt - 1) // rows_cnt if n else 1
    pairs = []
    for i in range(0, n, chunk):
        pairs.append([(c, f"first:{c}") for c in codes[i : i + chunk]])
    return _make_rows(pairs)


def _kb_second(first):
    codes = user_interface["buttons"].get("currency_codes", CURRENCIES)
    buttons = [(c, f"second:{first}:{c}") for c in codes if c != first]
    n = len(buttons)
    rows_cnt = 1 if n <= 3 else 2 if n <= 6 else 3
    chunk = (n + rows_cnt - 1) // rows_cnt if n else 1
    pairs = [buttons[i : i + chunk] for i in range(0, n, chunk)]
    pairs.append([(user_interface["buttons"]["back_label"], "back:first")])
    return _make_rows(pairs)


def _kb_days(first, second):
    rows = []
    row = []
    day_labels = user_interface["buttons"].get("days", [str(i) for i in range(1, 10)])
    for i, label in enumerate(day_labels, start=1):
        row.append((label, f"days:{first}:{second}:{i}"))
        if len(row) == 3:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    rows.append([(user_interface["buttons"]["back_label"], f"back:second:{first}")])
    return _make_rows(rows)


def _kb_confirm(first, second, days):
    confirm_label = user_interface["buttons"].get("confirm_label", "Все верно")
    back_label = user_interface["buttons"]["back_label"]
    return _make_rows([[(confirm_label, f"confirm:{first}:{second}:{days}")], [(back_label, f"back:second:{first}")]])


def _save_chat_state(chat_id, message_id, has_logo, **kwargs):
    state = CHAT_STATE.get(chat_id, {})
    state.update({"msg_id": int(message_id), "has_logo": bool(has_logo)})
    state.update(kwargs)
    CHAT_STATE[chat_id] = state


def _get_chat_state(chat_id):
    return CHAT_STATE.get(
        chat_id,
        {"msg_id": None, "has_logo": False, "awaiting_question": False, "asked_count": 0},
    )


def _kb_for_state(state):
    awaiting = bool(state.get("awaiting_question", False))
    asked = int(state.get("asked_count", 0))
    asked_at_pred = int(state.get("asked_count_at_prediction", 0)) if state.get("asked_count_at_prediction") is not None else asked
    show_toggle = (not awaiting) and (asked == asked_at_pred)
    media_format = state.get("media_format", "gif")
    toggle_label = user_interface["buttons"]["png"] if media_format == "gif" else user_interface["buttons"]["gif"]
    first = state.get("first")
    second = state.get("second")
    days = int(state.get("days", 1)) if state.get("days") is not None else 1

    rows = []
    if show_toggle:
        rows.append([(toggle_label, "toggle")])

    if first and second and (asked > asked_at_pred):
        back_cb = "back:restore"
    else:
        back_cb = "back:first" if not (first and second) else "back:first"

    if first and second:
        ask_label = user_interface["buttons"]["ask_label_first"] if asked == 0 else user_interface["buttons"]["ask_label_more"]
        rows.append([(user_interface["buttons"]["back_label"], back_cb), (ask_label, f"ask:{first}:{second}:{days}")])
    else:
        rows.append([(user_interface["buttons"]["back_label"], "back:first")])

    return _make_rows(rows)


def _handle_generic_error(bot: Bot, chat_id: int, message_id: int, exc: Exception):
    rid = exception_rid(logger, "Generic Error Handler caught exception", exc=exc)
    try:
        if message_id:
            bot.delete_message(chat_id=chat_id, message_id=message_id)
    except Exception:
        pass
    err_text = user_interface['captions']['unexpected_error']
    try:
        bot.send_message(chat_id=chat_id, text=f"{err_text} (id: {rid})")
    except Exception:
        logger.exception("Failed to send error text to chat=%s", chat_id)
    _clear_chat_media_and_cache(chat_id)
    caption = user_interface["captions"]["choose_first"]
    markup = _kb_first()
    try:
        with open(LOGO_PATH, "rb") as f:
            msg = bot.send_photo(chat_id=chat_id, photo=f, caption=caption, reply_markup=markup)
            _save_chat_state(chat_id, msg.message_id, True, last_caption=caption, last_markup=_markup_repr(markup))
    except Exception:
        try:
            msg = bot.send_message(chat_id=chat_id, text=caption, reply_markup=markup)
            _save_chat_state(chat_id, msg.message_id, False, last_caption=caption, last_markup=_markup_repr(markup))
        except Exception:
             logger.exception("Failed to send main menu fallback to chat=%s", chat_id)


def _submit_background(bot, chat_id, message_id, fn, *args, **kwargs):
    lock = _get_chat_lock(chat_id)
    with lock:
        state = CHAT_STATE.get(chat_id, {})
        if state.get("running_task"):
            return False
        state["running_task"] = True
        CHAT_STATE[chat_id] = state

    def task_wrapper():
        try:
            fn(*args, **kwargs)
        except Exception as e:
            raise e
        finally:
            with lock:
                s = CHAT_STATE.get(chat_id, {})
                s["running_task"] = False
                CHAT_STATE[chat_id] = s

    future = EXECUTOR.submit(task_wrapper)
    def done_callback(f):
        try:
            f.result()
        except Exception as exc:
            if bot:
                _handle_generic_error(bot, chat_id, message_id, exc)
    future.add_done_callback(done_callback)
    return True


def _send_start_message(bot: Bot, chat_id: int):
    with open(LOGO_PATH, "rb") as f:
        kb = InlineKeyboardMarkup([[InlineKeyboardButton(user_interface["buttons"]["warning_label"], callback_data="do_predict")]])
        msg = bot.send_photo(chat_id=chat_id, photo=f, caption=user_interface["captions"]["warning"], reply_markup=kb)
    _save_chat_state(chat_id, msg.message_id, True, last_caption=user_interface["captions"]["warning"], last_markup=_markup_repr(kb))
    return msg


def start_handler(update, context):
    chat_id = update.effective_chat.id
    try:
        _send_start_message(context.bot, chat_id)
    except Exception as e:
        _handle_generic_error(context.bot, chat_id, None, e)


def _edit_with_retries(action_callable, bot: Bot, chat_id: int, message_id: int, max_attempts: int = 3, delay: float = 1.0):
    lock = _get_chat_lock(chat_id)
    last_exc = None
    with lock:
        for attempt in range(1, max_attempts + 1):
            try:
                action_callable()
                return True
            except BadRequest as be:
                last_exc = be
                msg = str(be)
                if "Message is not modified" in msg:
                    return True
                if "Message to edit not found" in msg or "Message to delete not found" in msg or "message to edit not found" in msg.lower():
                    break 
            except Exception as e:
                last_exc = e
            if attempt < max_attempts:
                time.sleep(delay)
    _handle_generic_error(bot, chat_id, message_id, last_exc or Exception("Edit retries failed"))
    return False


def _replace_with_logo(bot, chat_id, message_id, caption=None, reply_markup=None, max_attempts=3, delay=1.0):
    def try_edit_media():
        with open(LOGO_PATH, "rb") as f:
            media = InputMediaPhoto(f, caption=caption)
            bot.edit_message_media(media=media, chat_id=chat_id, message_id=message_id, reply_markup=reply_markup)
    success = _edit_with_retries(try_edit_media, bot, chat_id, message_id, max_attempts=max_attempts, delay=delay)
    if success:
        _save_chat_state(chat_id, message_id, True, last_caption=caption or "", last_markup=_markup_repr(reply_markup))
        return
    return


def _save_media_state(chat_id, new_msg, media_path, is_gif):
    try:
        if not _is_generated_media(media_path):
            return
        with _get_chat_lock(chat_id):
            s = _get_chat_state(chat_id)
            s["last_media"] = media_path
            if is_gif:
                fid = None
                if getattr(new_msg, "animation", None):
                    fid = new_msg.animation.file_id
                elif getattr(new_msg, "document", None) and getattr(new_msg.document, "file_id", None):
                    fid = new_msg.document.file_id
                s["last_media_gif"] = media_path
                if fid:
                    s["last_media_gif_file_id"] = fid
            else:
                fid = None
                if getattr(new_msg, "photo", None):
                    try:
                        fid = new_msg.photo[-1].file_id
                    except Exception:
                        fid = None
                elif getattr(new_msg, "document", None) and getattr(new_msg.document, "file_id", None):
                    fid = new_msg.document.file_id
                s["last_media_png"] = media_path
                if fid:
                    s["last_media_png_file_id"] = fid
            CHAT_STATE[chat_id] = s
    except Exception:
        logger.exception("_save_media_state: failed to save media info for chat=%s", chat_id)


def _send_replace_media(bot, chat_id, message_id, media_path, is_gif, caption, reply_markup):
    def _is_file_id(m):
        return isinstance(m, str) and (not os.path.exists(m)) and (not _is_protected_asset(m))
    try:
        if _is_file_id(media_path):
            media = InputMediaAnimation(media=media_path, caption=caption) if is_gif else InputMediaPhoto(media=media_path, caption=caption)
            bot.edit_message_media(media=media, chat_id=chat_id, message_id=message_id, reply_markup=reply_markup)
            try:
                if _is_generated_media(media_path):
                    with _get_chat_lock(chat_id):
                        s = _get_chat_state(chat_id)
                        s["last_media"] = media_path
                        if is_gif:
                            s["last_media_gif_file_id"] = media_path
                        else:
                            s["last_media_png_file_id"] = media_path
                        s["last_caption"] = caption or ""
                        s["last_markup"] = _markup_repr(reply_markup)
                        CHAT_STATE[chat_id] = s
            except Exception:
                pass
            return True

        if _is_protected_asset(media_path):
            try:
                with open(media_path, "rb") as f:
                    media = InputMediaAnimation(f, caption=caption) if is_gif else InputMediaPhoto(f, caption=caption)
                    bot.edit_message_media(media=media, chat_id=chat_id, message_id=message_id, reply_markup=reply_markup)
                return True
            except BadRequest as be:
                msg = str(be)
                if "Message is not modified" in msg:
                    return True

        def try_edit_media_local():
            with open(media_path, "rb") as f:
                media = InputMediaAnimation(f, caption=caption) if is_gif else InputMediaPhoto(f, caption=caption)
                bot.edit_message_media(media=media, chat_id=chat_id, message_id=message_id, reply_markup=reply_markup)

        lock = _get_chat_lock(chat_id)
        try:
            with lock:
                try_edit_media_local()
            if not _is_protected_asset(media_path) and _is_generated_media(media_path):
                try:
                    with _get_chat_lock(chat_id):
                        s = _get_chat_state(chat_id)
                        s["last_media"] = media_path
                        if is_gif:
                            s["last_media_gif"] = media_path
                        else:
                            s["last_media_png"] = media_path
                        s["last_caption"] = caption or ""
                        s["last_markup"] = _markup_repr(reply_markup)
                        CHAT_STATE[chat_id] = s
                except Exception:
                    pass
            return True
        except BadRequest as be:
            msg = str(be)
            if "Message is not modified" in msg:
                return True
    except BadRequest as be:
        msg = str(be)
        if "Message is not modified" in msg:
            return True
    except Exception:
        pass

    try:
        try:
            bot.delete_message(chat_id=chat_id, message_id=message_id)
        except Exception:
            pass

        if _is_file_id(media_path):
            if is_gif:
                new_msg = bot.send_animation(chat_id=chat_id, animation=media_path, caption=caption, reply_markup=reply_markup)
            else:
                new_msg = bot.send_photo(chat_id=chat_id, photo=media_path, caption=caption, reply_markup=reply_markup)
            try:
                if _is_generated_media(media_path):
                    with _get_chat_lock(chat_id):
                        s = _get_chat_state(chat_id)
                        if is_gif:
                            s["last_media_gif_file_id"] = media_path
                            s["last_media_gif"] = media_path
                        else:
                            s["last_media_png_file_id"] = media_path
                            s["last_media_png"] = media_path
                        s["last_media"] = media_path
                        s["msg_id"] = new_msg.message_id
                        s["last_caption"] = caption or ""
                        s["last_markup"] = _markup_repr(reply_markup)
                        CHAT_STATE[chat_id] = s
            except Exception:
                pass
            return True

        if _is_protected_asset(media_path):
            try:
                with open(media_path, "rb") as f:
                    if is_gif:
                        new_msg = bot.send_animation(chat_id=chat_id, animation=f, caption=caption, reply_markup=reply_markup)
                    else:
                        new_msg = bot.send_photo(chat_id=chat_id, photo=f, caption=caption, reply_markup=reply_markup)
                return True
            except Exception as e:
                _handle_generic_error(bot, chat_id, message_id, e)
                return False

        with open(media_path, "rb") as f:
            if is_gif:
                new_msg = bot.send_animation(chat_id=chat_id, animation=f, caption=caption, reply_markup=reply_markup)
            else:
                new_msg = bot.send_photo(chat_id=chat_id, photo=f, caption=caption, reply_markup=reply_markup)

        try:
            _save_media_state(chat_id, new_msg, media_path, is_gif)
            with _get_chat_lock(chat_id):
                s = _get_chat_state(chat_id)
                s["msg_id"] = new_msg.message_id
                s["last_caption"] = caption or ""
                s["last_markup"] = _markup_repr(reply_markup)
                CHAT_STATE[chat_id] = s
        except Exception:
            pass
        return True

    except Exception as e:
        _handle_generic_error(bot, chat_id, message_id, e)
        return False


def call_gemini_advice(question_text: str, summary_text: str):
    prompt = PROMPT_TEMPLATE + summary_text + "\n\nВопрос пользователя:\n" + question_text + "\n\nОтвет:"
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    headers = {"Content-Type": "application/json", "x-goog-api-key": GEMINI_API_KEY}
    try:
        r = SESSION.post(GEMINI_URL, headers=headers, data=json.dumps(payload), timeout=30)
        r.raise_for_status()
    except Exception:
        raise RuntimeError("Gemini API request failed")
    try:
        j = r.json()
    except Exception:
        raise RuntimeError("Failed to parse Gemini response")
    try:
        found = j["candidates"][0]["content"]["parts"][0]["text"].strip()
        return found
    except Exception:
        def walk(o):
            if isinstance(o, str):
                return o
            if isinstance(o, dict):
                for v in o.values():
                    found = walk(v)
                    if found:
                        return found
            if isinstance(o, list):
                for v in o:
                    found = walk(v)
                    if found:
                        return found
            return None
        txt = walk(j)
        if txt:
            return txt.strip()
        return None


def _clear_chat_media_and_cache(chat_id):
    try:
        with _get_chat_lock(chat_id):
            state = _get_chat_state(chat_id)
            for key in ("last_media_png", "last_media_gif", "last_media"):
                p = state.get(key)
                if isinstance(p, str) and p and not _is_protected_asset(p) and os.path.exists(p):
                    try:
                        os.remove(p)
                    except Exception:
                        pass
            keys_to_pop = [
                "cached_all_rates",
                "cached_pair_key",
                "forecasted_prices",
                "forecasted_diffs",
                "forecast_delta",
                "advice_text",
                "forecast_ts",
                "media_format",
                "last_media_png",
                "last_media_gif",
                "last_media",
                "last_media_png_pred",
                "last_media_gif_pred",
                "media_format_at_prediction",
                "advice_text_pred",
                "asked_count_at_prediction",
            ]
            for k in keys_to_pop:
                state.pop(k, None)
            state["awaiting_question"] = False
            state["qa_history"] = []
            state["asked_count"] = 0
            state["running_task"] = False
            CHAT_STATE[chat_id] = state
    except Exception:
        logger.exception("_clear_chat_media_and_cache: failed for chat=%s", chat_id)


def _perform_prediction_and_edit(bot, chat_id, message_id, user_id, first, second, days):
    _clear_chat_media_and_cache(chat_id)
    try:
        needed_days = max(MODELS_SETTINGS["reg"]["max_n"], MODELS_SETTINGS["markov"]["max_n"])
    except Exception:
        needed_days = days if days > 0 else 10

    all_rates = fetch_sequences_all_pairs(CURRENCIES, days=needed_days)

    pair_key = f"{second}_per_{first}"
    dates = sorted(all_rates.keys())
    prices = []

    for d in dates:
        day_rates = all_rates[d]
        if pair_key not in day_rates:
            raise RuntimeError(f"Pair {pair_key} missing in data for {d}")
        prices.append(float(day_rates[pair_key]))

    if not prices:
        raise ValueError("Prices list is empty after fetch")

    diffs = [0.0]
    for i in range(1, len(prices)):
        diffs.append(prices[i] - prices[i - 1])
    signs = ["+" if d >= 0 else "-" for d in diffs]
    old_prices = prices[-3:] if len(prices) >= 3 else prices[:]

    if not old_prices:
        raise ValueError("Not enough data to plot")

    last_price = old_prices[-1]
    reg_models = []
    markov_models = []
    for n in range(3, 11):
        p = os.path.join(MODELS_PATH, f"regression_{first}{second}_{n}.pkl")
        if os.path.exists(p):
            reg_models.append(load_model(p))
        p2 = os.path.join(MODELS_PATH, f"markov_{first}{second}_{n}.pkl")
        if os.path.exists(p2):
            markov_models.append(load_model(p2))

    max_nlags = max([len(c) - 1 for c in reg_models]) if reg_models else 1
    max_nlags = min(max_nlags, len(diffs))
    last_diffs = diffs[-max_nlags:] if len(diffs) >= max_nlags else diffs[:]

    forecasted_diffs = forecast_diffs(last_diffs, reg_models, days) if reg_models else [0.0] * days
    forecasted_signs = forecast_signs(signs, markov_models, days) if markov_models else [None] * days

    adjusted = []
    for d, s in zip(forecasted_diffs, forecasted_signs):
        if s == "-":
            adjusted.append(-abs(d))
        elif s == "+":
            adjusted.append(abs(d))
        else:
            adjusted.append(d)

    cur = last_price
    new_prices = []
    for d in adjusted:
        cur = cur + d
        new_prices.append(cur)

    png_path = _temp_file(user_id, ext="png")
    plot_sequence(old_prices, new_prices, png_path)
    gif_path = _temp_file(user_id, ext="gif")

    try:
        make_forecast_gif(old_prices, new_prices, gif_path)
    except Exception:
        logger.warning("GIF creation failed, using PNG")
        gif_path = png_path

    delta = new_prices[-1] - last_price
    delta_percent = (delta / last_price) * 100 if last_price != 0 else 0
    arrow = "↑" if delta > 0 else "↓" if delta < 0 else "→"
    color_emoji = "🟢" if delta > 0 else "🔴" if delta < 0 else "🟡"
    trend = "рост" if delta > 0 else "падение" if delta < 0 else "без изменений"
    verb = "стоит" if delta > 0 else "не стоит"
    advice = f"Скорее всего {verb} покупать"
    caption = (
        f"{first}/{second} → {last_price:.6f}\n"
        f"{color_emoji} Прогноз: {trend}\n"
        f"{arrow} {delta_percent:+.2f}% за {days} дн. ({delta:+.6f})\n{advice}\n\n"
        f"Данные на {date.today().strftime('%d.%m.%Y')}"
    )

    state = _get_chat_state(chat_id)
    state.setdefault("qa_history", [])
    state.setdefault("asked_count", 0)
    asked_count_before = state.get("asked_count", 0)

    kb = _kb_for_state(
        {
            "media_format": "gif",
            "first": first,
            "second": second,
            "days": days,
            "asked_count": state.get("asked_count", 0),
            "asked_count_at_prediction": state.get("asked_count_at_prediction", 0),
        }
    )
    success = _send_replace_media(bot, chat_id, message_id, gif_path, is_gif=True, caption=caption, reply_markup=kb)
    if not success:
        raise RuntimeError("Failed to send prediction media")

    try:
        with _get_chat_lock(chat_id):
            current = _get_chat_state(chat_id)
            current.update(
                {
                    "first": first,
                    "second": second,
                    "days": days,
                    "awaiting_question": False,
                    "forecasted_prices": [float(x) for x in new_prices],
                    "forecasted_diffs": [float(x) for x in adjusted],
                    "forecast_delta": float(delta),
                    "advice_text": caption,
                    "forecast_ts": int(time.time()),
                    "cached_all_rates": all_rates,
                    "cached_pair_key": pair_key,
                    "media_format": "gif",
                    "asked_count_at_prediction": asked_count_before,
                    "media_format_at_prediction": "gif",
                    "last_media_png_pred": png_path,
                    "last_media_gif_pred": gif_path,
                    "advice_text_pred": caption,
                }
            )
            current["last_media"] = current.get("last_media", gif_path)
            if not current.get("last_media_gif"):
                current["last_media_gif"] = gif_path
            if not current.get("last_media_png"):
                current["last_media_png"] = png_path
            CHAT_STATE[chat_id] = current
    except Exception:
        pass


def cb_query(update, context):
    query = update.callback_query
    data = (query.data or "").strip()
    user_id = update.effective_user.id
    chat_id = query.message.chat.id
    message_id = query.message.message_id

    try:
        try:
            query.answer(cache_time=0)
        except Exception:
            try:
                query.answer()
            except Exception:
                pass

        parts = data.split(":")
        cmd = parts[0]

        if cmd == "do_predict":
            def try_edit():
                context.bot.edit_message_caption(chat_id=chat_id, message_id=message_id, caption=user_interface["captions"]["choose_first"], reply_markup=_kb_first())
            _edit_with_retries(try_edit, context.bot, chat_id, message_id)
            return

        if cmd == "first" and len(parts) == 2:
            first = parts[1]
            def try_edit():
                context.bot.edit_message_caption(chat_id=chat_id, message_id=message_id, caption=f"Первая валюта: {first}\n{user_interface['captions']['choose_second']}", reply_markup=_kb_second(first))
            _edit_with_retries(try_edit, context.bot, chat_id, message_id)
            return

        if cmd == "second" and len(parts) == 3:
            _, first, second = parts
            def try_edit():
                context.bot.edit_message_caption(chat_id=chat_id, message_id=message_id, caption=f"Валютная пара: {first}/{second}\n{user_interface['captions']['choose_days']}", reply_markup=_kb_days(first, second))
            _edit_with_retries(try_edit, context.bot, chat_id, message_id)
            return

        if cmd == "days" and len(parts) == 4:
            _, first, second, days_str = parts
            try:
                days = int(days_str)
            except ValueError:
                raise ValueError("Invalid days selected")

            caption_tpl = user_interface["captions"].get("confirm_selection")
            if caption_tpl:
                caption = caption_tpl.format(first=first, second=second, days=days)
            else:
                caption = f"Валютная пара: {first}/{second}\nКоличество дней: {days}"
            kb = _kb_confirm(first, second, days)
            def try_show_confirm():
                context.bot.edit_message_caption(chat_id=chat_id, message_id=message_id, caption=caption, reply_markup=kb)
            _edit_with_retries(try_show_confirm, context.bot, chat_id, message_id)
            return

        if cmd == "confirm" and len(parts) == 4:
            _, first, second, days_str = parts
            try:
                days = int(days_str)
            except ValueError:
                 raise ValueError("Invalid days selected")

            def try_edit_predicting_caption():
                context.bot.edit_message_caption(chat_id=chat_id, message_id=message_id, caption=user_interface["captions"]["predicting"])
            _edit_with_retries(try_edit_predicting_caption, context.bot, chat_id, message_id)

            if PREDICTING_PATH:
                _send_replace_media(context.bot, chat_id, message_id, PREDICTING_PATH, is_gif=False, caption=user_interface["captions"]["predicting"], reply_markup=None)

            if not _submit_background(context.bot, chat_id, message_id, _perform_prediction_and_edit, context.bot, chat_id, message_id, user_id, first, second, days):
                try:
                    query.answer(text="Подождите, выполняется операция...", show_alert=False)
                except Exception:
                    pass
            return

        if cmd == "ask" and len(parts) == 4:
            _, first, second, _ = parts
            with _get_chat_lock(chat_id):
                state = _get_chat_state(chat_id)
                state.update({"awaiting_question": True})
                CHAT_STATE[chat_id] = state

            cancel_kb = _make_rows([[(user_interface["buttons"]["cancel_label"], f"cancel_ask")]])
            _send_replace_media(context.bot, chat_id, message_id, ASK_IMG_PATH, is_gif=False, caption=user_interface["captions"]["ask_question"], reply_markup=cancel_kb)
            return

        if cmd == "cancel_ask":
            with _get_chat_lock(chat_id):
                state = _get_chat_state(chat_id)
                state["awaiting_question"] = False
                CHAT_STATE[chat_id] = state

            pred_media_format = state.get("media_format_at_prediction") or state.get("media_format")
            gif_restore = state.get("last_media_gif_pred_file_id") or state.get("last_media_gif_pred") or state.get("last_media_gif_file_id") or state.get("last_media_gif")
            png_restore = state.get("last_media_png_pred_file_id") or state.get("last_media_png_pred") or state.get("last_media_png_file_id") or state.get("last_media_png")

            if pred_media_format == "gif":
                media_path = gif_restore or png_restore
                is_gif = bool(gif_restore)
            else:
                media_path = png_restore or gif_restore
                is_gif = False

            if not media_path:
                _replace_with_logo(context.bot, chat_id, message_id, caption=state.get("advice_text") or user_interface["captions"]["choose_first"], reply_markup=_kb_first())
                return

            kb_state = state.copy()
            kb_state["media_format"] = pred_media_format
            kb_state["asked_count"] = state.get("asked_count_at_prediction", state.get("asked_count", 0))
            kb_state["asked_count_at_prediction"] = state.get("asked_count_at_prediction", kb_state["asked_count"])
            kb = _kb_for_state(kb_state)
            caption = state.get("advice_text_pred") or state.get("advice_text", "")

            success = _send_replace_media(context.bot, chat_id, message_id, media_path, is_gif, caption=caption, reply_markup=kb)
            if success:
                with _get_chat_lock(chat_id):
                    state = _get_chat_state(chat_id)
                    if state.get("last_media_png_pred"):
                        state["last_media_png"] = state["last_media_png_pred"]
                    if state.get("last_media_gif_pred"):
                        state["last_media_gif"] = state["last_media_gif_pred"]
                    CHAT_STATE[chat_id] = state
            return

        if cmd == "back":
            if len(parts) >= 2 and parts[1] == "first":
                _clear_chat_media_and_cache(chat_id)
                _replace_with_logo(context.bot, chat_id, message_id, caption=user_interface["captions"]["choose_first"], reply_markup=_kb_first())
                return

            if len(parts) >= 3 and parts[1] == "second":
                first = parts[2]
                _clear_chat_media_and_cache(chat_id)
                _replace_with_logo(context.bot, chat_id, message_id, caption=f"Первая валюта: {first}\n{user_interface['captions']['choose_second']}", reply_markup=_kb_second(first))
                return

            if len(parts) >= 2 and parts[1] in ("restore", "media"):
                with _get_chat_lock(chat_id):
                    state = _get_chat_state(chat_id)

                pred_media_format = state.get("media_format_at_prediction") or state.get("media_format")
                if pred_media_format == "gif":
                    media_path = state.get("last_media_gif_pred_file_id") or state.get("last_media_gif_pred") or state.get("last_media_gif_file_id") or state.get("last_media_gif") or state.get("last_media_png")
                    is_gif = bool(state.get("last_media_gif_pred_file_id") or state.get("last_media_gif_pred") or state.get("last_media_gif_file_id") or state.get("last_media_gif"))
                else:
                    media_path = state.get("last_media_png_pred_file_id") or state.get("last_media_png_pred") or state.get("last_media_png_file_id") or state.get("last_media_png") or state.get("last_media_gif")
                    is_gif = False

                if not media_path:
                    _replace_with_logo(context.bot, chat_id, message_id, caption=state.get("advice_text") or user_interface["captions"]["choose_first"], reply_markup=_kb_first())
                    return

                kb_state = state.copy()
                kb_state["media_format"] = pred_media_format
                kb_state["asked_count"] = state.get("asked_count_at_prediction", state.get("asked_count", 0))
                kb_state["asked_count_at_prediction"] = state.get("asked_count_at_prediction", kb_state["asked_count"])
                kb = _kb_for_state(kb_state)
                caption = state.get("advice_text_pred") or state.get("advice_text", "")
                _send_replace_media(context.bot, chat_id, message_id, media_path, is_gif, caption=caption, reply_markup=kb)
                return

        if cmd == "toggle":
            with _get_chat_lock(chat_id):
                state = _get_chat_state(chat_id)

            if state.get("running_task"):
                 try:
                    query.answer(text="Обработка...", show_alert=False)
                 except Exception:
                    pass
                 return

            asked = int(state.get("asked_count", 0))
            asked_at_pred = int(state.get("asked_count_at_prediction", 0)) if state.get("asked_count_at_prediction") is not None else asked
            if state.get("awaiting_question") or (asked > asked_at_pred):
                try:
                    query.answer(text="Переключение недоступно в этом состоянии", show_alert=False)
                except Exception:
                    pass
                return

            msg_obj = query.message
            current = "gif" if getattr(msg_obj, "animation", None) else "png" if getattr(msg_obj, "photo", None) else state.get("media_format", "gif")
            target = "png" if current == "gif" else "gif"

            if target == "png":
                media_candidate = state.get("last_media_png_file_id") or state.get("last_media_png")
            else:
                media_candidate = state.get("last_media_gif_file_id") or state.get("last_media_gif")

            def _has_media_for_target():
                if target == "png":
                    return bool(state.get("last_media_png_file_id") or (state.get("last_media_png") and os.path.exists(state.get("last_media_png"))))
                else:
                    return bool(state.get("last_media_gif_file_id") or (state.get("last_media_gif") and os.path.exists(state.get("last_media_gif"))))

            need_generate = not _has_media_for_target()

            if need_generate:
                 def bg_toggle():
                    with _get_chat_lock(chat_id):
                         s_curr = _get_chat_state(chat_id)
                         cached_all = s_curr.get("cached_all_rates")
                         pair_key = s_curr.get("cached_pair_key")
                         old_prices = []
                         if cached_all and pair_key:
                            dates = sorted(cached_all.keys())
                            prices = [float(cached_all[d][pair_key]) for d in dates]
                            old_prices = prices[-3:] if len(prices) >= 3 else prices[:]
                         else:
                            forecasted = s_curr.get("forecasted_prices", [])
                            old_prices = [forecasted[0]] if forecasted else []

                         if target == "png":
                            new_media_path = _temp_file(user_id, ext="png")
                            plot_sequence(old_prices, s_curr.get("forecasted_prices", []), new_media_path)
                            new_is_gif = False
                         else:
                            new_media_path = _temp_file(user_id, ext="gif")
                            try:
                                make_forecast_gif(old_prices, s_curr.get("forecasted_prices", []), new_media_path)
                                new_is_gif = True
                            except Exception:
                                new_media_path = s_curr.get("last_media_png") or new_media_path
                                new_is_gif = False

                         next_state = s_curr.copy()
                         next_state["media_format"] = target
                         kb = _kb_for_state(next_state)
                         caption = s_curr.get("advice_text", "")

                    success = _send_replace_media(context.bot, chat_id, message_id, new_media_path, new_is_gif, caption=caption, reply_markup=kb)

                    if success:
                        with _get_chat_lock(chat_id):
                            s_upd = CHAT_STATE.get(chat_id, {})
                            s_upd["media_format"] = target
                            if not new_is_gif:
                                s_upd["last_media_png"] = new_media_path
                            else:
                                s_upd["last_media_gif"] = new_media_path
                            CHAT_STATE[chat_id] = s_upd

                 _submit_background(context.bot, chat_id, message_id, bg_toggle)
                 return

            next_state = state.copy()
            next_state["media_format"] = target
            kb = _kb_for_state(next_state)
            caption = state.get("advice_text", "")

            success = _send_replace_media(context.bot, chat_id, message_id, media_candidate, target == "gif", caption=caption, reply_markup=kb)

            if success:
                with _get_chat_lock(chat_id):
                    st = _get_chat_state(chat_id)
                    st["media_format"] = target
                    if target == "gif":
                        if media_candidate:
                            st["last_media_gif"] = media_candidate
                            if isinstance(media_candidate, str) and not os.path.exists(media_candidate):
                                st["last_media_gif_file_id"] = media_candidate
                    else:
                        if media_candidate:
                            st["last_media_png"] = media_candidate
                            if isinstance(media_candidate, str) and not os.path.exists(media_candidate):
                                st["last_media_png_file_id"] = media_candidate
                    CHAT_STATE[chat_id] = st
            return

        query.answer(text="Неизвестная команда", show_alert=False)

    except Exception as e:
        _handle_generic_error(context.bot, chat_id, message_id, e)


def _background_question_logic(bot, chat_id, bot_msg_id, text):
    with _get_chat_lock(chat_id):
        state = _get_chat_state(chat_id)
        first = state.get("first")
        second = state.get("second")
        days = int(state.get("days", 1))
        cached_all = state.get("cached_all_rates")
        cached_key = state.get("cached_pair_key")
        ts = state.get("forecast_ts")

        now_ts = int(time.time())
        pair_key = f"{second}_per_{first}"

        use_cached = False
        if cached_all and cached_key and ts:
            if cached_key == pair_key and (now_ts - int(ts)) <= CACHE_TTL:
                use_cached = True

    if use_cached:
        all_rates = cached_all
    else:
        all_rates = fetch_sequences_all_pairs(CURRENCIES, days=max(MODELS_SETTINGS["reg"]["max_n"], MODELS_SETTINGS["markov"]["max_n"]))

    dates = sorted(all_rates.keys())
    prices = [float(all_rates[d][pair_key]) for d in dates]
    last_prices_text = ", ".join(f"{p:.6f}" for p in (prices[-3:] if prices else []))

    with _get_chat_lock(chat_id):
        state = _get_chat_state(chat_id)
        forecasted_prices = state.get("forecasted_prices")
        forecast_delta = state.get("forecast_delta")
        qa_history = state.get("qa_history", [])

    history_text = ""
    if qa_history:
        parts = []
        for i, item in enumerate(qa_history[-5:]):
            q = item.get("q", "")
            a = item.get("a", "")
            parts.append(f"Q{i+1}: {q}\nA{i+1}: {a}")
        history_text = "\n".join(parts) + "\n"

    if forecasted_prices:
        forecast_text = ", ".join(f"{p:.6f}" for p in forecasted_prices)
        if forecast_delta is not None:
            delta_val = forecast_delta
        else:
            delta_val = forecasted_prices[-1] - prices[-1] if prices else 0.0

        summary_text = (
            f"{history_text}"
            f"Pair: {first}/{second}\n"
            f"Latest prices: {last_prices_text}\n"
            f"Forecast days: {days}\n"
            f"Forecasted prices: {forecast_text}\n"
            f"Forecast delta (last vs current): {delta_val:.6f}\n"
        )
    else:
        summary_text = f"{history_text}Pair: {first}/{second}\nLatest prices: {last_prices_text}\nForecast days: {days}\n"

    gemini_resp = call_gemini_advice(text, summary_text)
    final_caption = gemini_resp if gemini_resp else "Ассистент не смог предоставить ответ (ошибка подключения к модели)."

    with _get_chat_lock(chat_id):
        state = _get_chat_state(chat_id)
        qa_history = state.get("qa_history", [])
        qa_history.append({"q": text, "a": final_caption})
        qa_history = qa_history[-5:]
        state["qa_history"] = qa_history
        state["asked_count"] = state.get("asked_count", 0) + 1
        CHAT_STATE[chat_id] = state

        media_format_pred = state.get("media_format_at_prediction") or state.get("media_format")
        if media_format_pred == "gif":
            media_path = state.get("last_media_gif_pred_file_id") or state.get("last_media_gif_file_id") or state.get("last_media_gif") or state.get("last_media_png")
            is_gif = bool(state.get("last_media_gif_pred_file_id") or state.get("last_media_gif_file_id") or state.get("last_media_gif"))
        else:
            media_path = state.get("last_media_png_pred_file_id") or state.get("last_media_png_file_id") or state.get("last_media_png") or state.get("last_media_gif")
            is_gif = False

        kb_state = state.copy()
        kb_state["media_format"] = media_format_pred
        kb_state["asked_count_at_prediction"] = state.get("asked_count_at_prediction", state.get("asked_count", 0))
        kb = _kb_for_state(kb_state)

    if media_path:
        success = _send_replace_media(bot, chat_id, bot_msg_id, media_path, is_gif, caption=final_caption, reply_markup=kb)
        if success:
            with _get_chat_lock(chat_id):
                state = _get_chat_state(chat_id)
                state["media_format"] = "gif" if is_gif else "png"
                state["last_media"] = media_path
                if is_gif:
                    state["last_media_gif"] = media_path
                else:
                    state["last_media_png"] = media_path
                CHAT_STATE[chat_id] = state
    else:
        def try_edit_final():
            bot.edit_message_caption(chat_id=chat_id, message_id=bot_msg_id, caption=final_caption, reply_markup=kb)
        _edit_with_retries(try_edit_final, bot, chat_id, bot_msg_id)


def question_message_handler(update, context):
    msg = update.message
    chat_id = msg.chat.id
    user_msg_id = msg.message_id
    text = (msg.text or "").strip()

    try:
        context.bot.delete_message(chat_id=chat_id, message_id=user_msg_id)
    except Exception:
        pass

    if not text:
        return

    try:
        with _get_chat_lock(chat_id):
            state = _get_chat_state(chat_id)
            if not state.get("awaiting_question"):
                return
            if state.get("running_task"):
                return

            state["awaiting_question"] = False
            state.setdefault("asked_count", 0)
            state.setdefault("asked_count_at_prediction", state.get("asked_count", 0))
            CHAT_STATE[chat_id] = state

        bot_msg_id = state.get("msg_id")

        _send_replace_media(context.bot, chat_id, bot_msg_id, AI_THINKING_PATH, is_gif=False, caption=user_interface["captions"]["awaiting_assistant"], reply_markup=None)

        if not _submit_background(context.bot, chat_id, bot_msg_id, _background_question_logic, context.bot, chat_id, bot_msg_id, text):
             pass

    except Exception as e:
        bot_msg_id = CHAT_STATE.get(chat_id, {}).get("msg_id")
        _handle_generic_error(context.bot, chat_id, bot_msg_id, e)


def telegram_main(config: dict):
    global TELEGRAM_TOKEN, TEMP_FOLDER, LOGO_PATH, MODELS_PATH, MODELS_SETTINGS, CURRENCIES, user_interface, CACHE_TTL, GEMINI_API_KEY, GEMINI_MODEL, GEMINI_URL, PROMPT_TEMPLATE, ASK_IMG_PATH, AI_THINKING_PATH, PREDICTING_PATH
    global EXECUTOR

    TELEGRAM_TOKEN = config.get("TELEGRAM_TOKEN")
    TEMP_FOLDER = config.get("TEMP_FOLDER")
    LOGO_PATH = config.get("LOGO_PATH")
    ASK_IMG_PATH = config.get("ASK_IMG_PATH")
    AI_THINKING_PATH = config.get("AI_THINKING_PATH")
    PREDICTING_PATH = config.get("PREDICTING_PATH")
    MODELS_PATH = config.get("MODELS_PATH")
    MODELS_SETTINGS = config.get("MODELS_SETTINGS")
    CURRENCIES = config.get("CURRENCIES")
    user_interface = config.get("user_interface")
    CACHE_TTL = int(config.get("CACHE_TTL", 300))
    GEMINI_API_KEY = config.get("GEMINI_API_KEY")
    GEMINI_MODEL = config.get("GEMINI_MODEL")
    PROMPT_TEMPLATE = config.get("PROMPT_TEMPLATE")

    if GEMINI_MODEL:
        GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    if TEMP_FOLDER and os.path.isdir(TEMP_FOLDER):
        for f in os.listdir(TEMP_FOLDER):
            p = os.path.join(TEMP_FOLDER, f)
            if os.path.isfile(p):
                try:
                    os.remove(p)
                except Exception:
                    logger.exception("telegram_main: failed to clear temp folder file %s", p)

    EXECUTOR = ThreadPoolExecutor(max_workers=os.cpu_count())

    req = Request(con_pool_size=HTTP_POOL_SIZE, connect_timeout=30, read_timeout=30)
    bot = Bot(token=TELEGRAM_TOKEN, request=req)
    updater = Updater(bot=bot, use_context=True)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler("start", start_handler))
    dp.add_handler(CallbackQueryHandler(cb_query))
    dp.add_handler(MessageHandler(Filters.text & (~Filters.command), question_message_handler))

    updater.start_polling()
    try:
        updater.idle()
    finally:
        if EXECUTOR:
            EXECUTOR.shutdown(wait=False)
