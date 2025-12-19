import os
import time
import threading
import json
from datetime import date

from logging_util import setup_logging, exception_rid
logger = setup_logging(level="debug", name=__name__)

from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup, InputMediaPhoto, InputMediaAnimation
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


def _temp_file(user_id, ext="png"):
    os.makedirs(TEMP_FOLDER, exist_ok=True)
    out = os.path.join(TEMP_FOLDER, f"{user_id}_{int(time.time()*1000)}.{ext}")
    return out


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


def _kb_first():
    codes = user_interface["buttons"].get("currency_codes", CURRENCIES)
    n = len(codes) if codes else 0
    if n <= 3:
        rows_cnt = 1
    elif n <= 6:
        rows_cnt = 2
    else:
        rows_cnt = 3
    chunk = (n + rows_cnt - 1) // rows_cnt if n else 1
    pairs = []
    for i in range(0, n, chunk):
        pairs.append([(c, f"first:{c}") for c in codes[i:i+chunk]])
    return _make_rows(pairs)


def _kb_second(first):
    codes = user_interface["buttons"].get("currency_codes", CURRENCIES)
    buttons = [(c, f"second:{first}:{c}") for c in codes if c != first]
    n = len(buttons)
    if n <= 3:
        rows_cnt = 1
    elif n <= 6:
        rows_cnt = 2
    else:
        rows_cnt = 3
    chunk = (n + rows_cnt - 1) // rows_cnt if n else 1
    pairs = []
    for i in range(0, n, chunk):
        pairs.append(buttons[i:i+chunk])
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
    rows.append([ (user_interface["buttons"]["back_label"], f"back:second:{first}") ])
    return _make_rows(rows)


def _make_rows(pairs):
    keyboard = []
    for row in pairs:
        keyboard.append([InlineKeyboardButton(text, callback_data=cb) for text, cb in row])
    return InlineKeyboardMarkup(keyboard)


def _save_chat_state(chat_id, message_id, has_logo, **kwargs):
    state = CHAT_STATE.get(chat_id, {})
    state.update({"msg_id": int(message_id), "has_logo": bool(has_logo)})
    for k, v in kwargs.items():
        state[k] = v
    CHAT_STATE[chat_id] = state


def _get_chat_state(chat_id):
    return CHAT_STATE.get(chat_id, {"msg_id": None, "has_logo": False, "awaiting_question": False, "asked_count": 0})


def _kb_for_state(state):
    """
    Генерирует корректную клавиатуру для текущего состояния `state`.
    Правила:
     - Если state['awaiting_question'] == True -> НЕ показываем кнопку toggle (вопрос в процессе).
     - Если показ текущего экрана — это экран ПОСЛЕ того, как пользователь задал вопрос (asked_count > asked_count_at_prediction),
       то НЕ показываем кнопку toggle и кнопка "назад" ведёт к восстановлению графика (back:restore).
     - В остальных случаях показываем toggle и строки [toggle] + [back, ask].
    """
    # determine whether to show toggle:
    awaiting = bool(state.get("awaiting_question", False))
    asked_count = int(state.get("asked_count", 0))
    asked_at_pred = int(state.get("asked_count_at_prediction", 0)) if state.get("asked_count_at_prediction") is not None else asked_count

    show_toggle = (not awaiting) and (asked_count == asked_at_pred)

    media_format = state.get("media_format", "gif")
    # choose toggle label: offer the *other* format as action
    if media_format == "gif":
        toggle_label = user_interface["buttons"]["png"]
    else:
        toggle_label = user_interface["buttons"]["gif"]

    first = state.get("first")
    second = state.get("second")
    days = int(state.get("days", 1)) if state.get("days") is not None else 1

    rows = []
    if show_toggle:
        rows.append([(toggle_label, "toggle")])

    # Decide back callback:
    # If we have a pair (graph shown) and we're in post-question state (asked_count > asked_at_pred),
    # make back restore the graph as it was at prediction time.
    if first and second and (asked_count > asked_at_pred):
        back_cb = "back:restore"
    elif first and second:
        # if we have pair but not post-question, back should go to first selection (as previous design)
        back_cb = "back:first"
    else:
        back_cb = "back:first"

    # Ask label (if pair exists)
    if first and second:
        ask_label = user_interface["buttons"]["ask_label_first"] if asked_count == 0 else user_interface["buttons"]["ask_label_more"]
        rows.append([(user_interface["buttons"]["back_label"], back_cb),
                     (ask_label, f"ask:{first}:{second}:{days}")])
    else:
        rows.append([(user_interface["buttons"]["back_label"], "back:first")])

    return _make_rows(rows)


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
    except Exception:
        logger.exception("start_handler: failed to send start message to chat=%s", chat_id)


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
    try:
        bot.delete_message(chat_id=chat_id, message_id=message_id)
    except Exception:
        logger.exception("_edit_with_retries: failed to delete old message chat=%s msg=%s", chat_id, message_id)
    try:
        rid = exception_rid(logger, "_edit_with_retries: final failure", exc=last_exc)
        bot.send_message(chat_id=chat_id, text=f"{user_interface['captions']['unexpected_error']} (id: {rid})")
    except Exception:
        logger.exception("_edit_with_retries: failed to send unexpected_error to chat=%s", chat_id)
    try:
        _send_start_message(bot, chat_id)
    except Exception:
        logger.exception("_edit_with_retries: failed to resend start message to chat=%s", chat_id)
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
    def try_edit_caption_no_logo():
        bot.edit_message_caption(chat_id=chat_id, message_id=message_id, caption=caption, reply_markup=reply_markup)
    success = _edit_with_retries(try_edit_caption_no_logo, bot, chat_id, message_id, max_attempts=max_attempts, delay=delay)
    if success:
        _save_chat_state(chat_id, message_id, False, last_caption=caption or "", last_markup=_markup_repr(reply_markup))
    return


def _send_replace_media(bot, chat_id, message_id, media_path, is_gif, caption, reply_markup):
    """
    Try to edit message media; if edit succeeds, update CHAT_STATE with media info.
    If edit fails and we send a new message, _save_chat_state is called with new msg id and media info as well.
    Returns True on success, False on fatal error.
    """
    def try_edit_media():
        with open(media_path, "rb") as f:
            if is_gif:
                media = InputMediaAnimation(f, caption=caption)
            else:
                media = InputMediaPhoto(f, caption=caption)
            bot.edit_message_media(media=media, chat_id=chat_id, message_id=message_id, reply_markup=reply_markup)

    # Attempt to edit existing message
    success = _edit_with_retries(try_edit_media, bot, chat_id, message_id)
    if success:
        # update CHAT_STATE to reflect new media (edit preserves message_id)
        try:
            # save last_media paths and media_format so callers and toggle logic stay consistent
            if is_gif:
                _save_chat_state(chat_id, message_id, False,
                                 last_caption=caption or "",
                                 last_markup=_markup_repr(reply_markup),
                                 last_media=media_path,
                                 last_media_gif=media_path,
                                 media_format="gif")
            else:
                _save_chat_state(chat_id, message_id, False,
                                 last_caption=caption or "",
                                 last_markup=_markup_repr(reply_markup),
                                 last_media=media_path,
                                 last_media_png=media_path,
                                 media_format="png")
        except Exception:
            logger.exception("_send_replace_media: failed to update chat state after edit for chat=%s", chat_id)
        return True

    # Edit failed — fallback to sending a new message and deleting the old one
    try:
        with open(media_path, "rb") as f:
            if is_gif:
                new_msg = bot.send_animation(chat_id=chat_id, animation=f, caption=caption, reply_markup=reply_markup)
            else:
                new_msg = bot.send_photo(chat_id=chat_id, photo=f, caption=caption, reply_markup=reply_markup)
        try:
            bot.delete_message(chat_id=chat_id, message_id=message_id)
        except Exception:
            logger.exception("_send_replace_media: failed to delete old media message chat=%s msg=%s", chat_id, message_id)
        # save state using new message id and include media paths + format
        try:
            if is_gif:
                _save_chat_state(chat_id, new_msg.message_id, False,
                                 last_caption=caption or "",
                                 last_markup=_markup_repr(reply_markup),
                                 last_media=new_msg,  # keep original pointer (not used), main fields below
                                 last_media_gif=media_path,
                                 media_format="gif")
            else:
                _save_chat_state(chat_id, new_msg.message_id, False,
                                 last_caption=caption or "",
                                 last_markup=_markup_repr(reply_markup),
                                 last_media=new_msg,
                                 last_media_png=media_path,
                                 media_format="png")
        except Exception:
            logger.exception("_send_replace_media: failed to save chat state after send for chat=%s", chat_id)
        return True
    except Exception as e:
        logger.exception("_send_replace_media: send media failed for chat=%s path=%s", chat_id, media_path)
        try:
            bot.delete_message(chat_id=chat_id, message_id=message_id)
        except Exception:
            logger.exception("_send_replace_media: failed to delete old message on fallback chat=%s msg=%s", chat_id, message_id)
        try:
            rid = exception_rid(logger, "_send_replace_media final failure", exc=e)
            bot.send_message(chat_id=chat_id, text=f"{user_interface['captions']['unexpected_error']} (id: {rid})")
        except Exception:
            logger.exception("_send_replace_media: failed to send unexpected_error to chat=%s", chat_id)
        try:
            _send_start_message(bot, chat_id)
        except Exception:
            logger.exception("_send_replace_media: failed to send start message to chat=%s", chat_id)
        return False


def call_gemini_advice(question_text: str, summary_text: str):
    prompt = PROMPT_TEMPLATE + summary_text + "\n\nUser question:\n" + question_text + "\n\nОтвет:"
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    headers = {"Content-Type": "application/json", "x-goog-api-key": GEMINI_API_KEY}
    try:
        r = SESSION.post(GEMINI_URL, headers=headers, data=json.dumps(payload), timeout=30)
        r.raise_for_status()
    except Exception:
        logger.exception("call_gemini_advice: request to model failed")
        return None
    try:
        j = r.json()
    except Exception:
        logger.exception("call_gemini_advice: failed to parse json response")
        return None
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
        logger.warning("call_gemini_advice: no text found in response")
        return None


def _perform_prediction_and_edit(bot, chat_id, message_id, user_id, first, second, days):
    try:
        with _get_chat_lock(chat_id):
            state = _get_chat_state(chat_id)
            for key in ("last_media_png", "last_media_gif"):
                p = state.get(key)
                protected_assets = [LOGO_PATH, ASK_IMG_PATH, AI_THINKING_PATH]
                if p and os.path.exists(p) and p not in protected_assets:                
                    try:
                        os.remove(p)
                    except Exception:
                        logger.exception("_perform_prediction_and_edit: failed to remove old temp file %s", p)
            state.pop("cached_all_rates", None)
            state.pop("cached_pair_key", None)
            state.pop("forecasted_prices", None)
            state.pop("forecasted_diffs", None)
            state.pop("forecast_delta", None)
            state.pop("advice_text", None)
            state.pop("forecast_ts", None)
            CHAT_STATE[chat_id] = state
    except Exception:
        logger.exception("_perform_prediction_and_edit: failed to clear previous cache for chat=%s", chat_id)
    try:
        needed_days = max(MODELS_SETTINGS["reg"]["max_n"], MODELS_SETTINGS["markov"]["max_n"])
    except Exception:
        needed_days = days if days > 0 else 10
    try:
        all_rates = fetch_sequences_all_pairs(CURRENCIES, days=needed_days)
    except Exception as e:
        def try_edit_error():
            bot.edit_message_caption(chat_id=chat_id, message_id=message_id, caption=f"Ошибка при получении актуальных данных: {e}")
        _edit_with_retries(try_edit_error, bot, chat_id, message_id)
        return
    pair_key = f"{second}_per_{first}"
    dates = sorted(all_rates.keys())
    prices = []
    try:
        for d in dates:
            day_rates = all_rates[d]
            if pair_key not in day_rates:
                raise RuntimeError(f"Пара {pair_key} отсутствует в данных за {d}")
            prices.append(float(day_rates[pair_key]))
    except Exception as e:
        def try_edit_error2():
            bot.edit_message_caption(chat_id=chat_id, message_id=message_id, caption=f"Ошибка при формировании последовательности цен: {e}")
        _edit_with_retries(try_edit_error2, bot, chat_id, message_id)
        return
    if not prices:
        def try_edit_err3():
            bot.edit_message_caption(chat_id=chat_id, message_id=message_id, caption="Ошибка: недостаточно данных от парсера")
        _edit_with_retries(try_edit_err3, bot, chat_id, message_id)
        return
    diffs = [0.0]
    for i in range(1, len(prices)):
        diffs.append(prices[i] - prices[i-1])
    signs = ["+" if d >= 0 else "-" for d in diffs]
    old_prices = prices[-3:] if len(prices) >= 3 else prices[:]
    if not old_prices:
        def try_edit_err4():
            bot.edit_message_caption(chat_id=chat_id, message_id=message_id, caption="Ошибка: недостаточно данных для построения графика")
        _edit_with_retries(try_edit_err4, bot, chat_id, message_id)
        return
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
        logger.exception("_perform_prediction_and_edit: failed to create gif, falling back to png")
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

    # store asked_count at prediction time so later we can detect "post-question" state
    asked_count_before = state.get("asked_count", 0)

    # default: show GIF when available
    kb = _kb_for_state({
        "media_format": "gif",
        "first": first,
        "second": second,
        "days": days,
        "asked_count": state.get("asked_count", 0),
        "asked_count_at_prediction": state.get("asked_count_at_prediction", 0)
    })
    success = _send_replace_media(bot, chat_id, message_id, gif_path, is_gif=True, caption=caption, reply_markup=kb)
    if not success:
        return
    # Save state and also save "snapshot" of media/KB at prediction moment to allow correct restore
    state.update({
        "last_media": gif_path,
        "last_media_png": png_path,
        "last_media_gif": gif_path,
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
        # snapshot fields to restore exactly what was shown at prediction time
        "asked_count_at_prediction": asked_count_before,
        "media_format_at_prediction": "gif",
        "last_media_png_pred": png_path,
        "last_media_gif_pred": gif_path,
        "advice_text_pred": caption
    })
    try:
        with _get_chat_lock(chat_id):
            CHAT_STATE[chat_id] = state
    except Exception:
        logger.exception("_perform_prediction_and_edit: failed to save chat state for chat=%s", chat_id)


def cb_query(update, context):
    query = update.callback_query
    data = (query.data or "").strip()
    user_id = update.effective_user.id
    chat_id = query.message.chat.id
    message_id = query.message.message_id

    try:
        query.answer(cache_time=2)
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
            def try_edit_err():
                context.bot.edit_message_caption(chat_id=chat_id, message_id=message_id, caption="Неверный выбор дней")
            _edit_with_retries(try_edit_err, context.bot, chat_id, message_id)
            return
        def try_edit_predicting():
            context.bot.edit_message_caption(chat_id=chat_id, message_id=message_id, caption=user_interface["captions"]["predicting"])
        _edit_with_retries(try_edit_predicting, context.bot, chat_id, message_id)
        _perform_prediction_and_edit(context.bot, chat_id, message_id, user_id, first, second, days)
        return

    if cmd == "ask" and len(parts) == 4:
        _, first, second, days_str = parts
        try:
            with _get_chat_lock(chat_id):
                state = _get_chat_state(chat_id)
                state.update({"awaiting_question": True})
                CHAT_STATE[chat_id] = state
        except Exception:
            logger.exception("cb_query: failed to set awaiting_question for chat=%s", chat_id)
        cancel_kb = _make_rows([[ (user_interface["buttons"]["cancel_label"], f"cancel_ask") ]])
        try:
            _send_replace_media(context.bot, chat_id, message_id, ASK_IMG_PATH, is_gif=False, caption=user_interface["captions"]["ask_question"], reply_markup=cancel_kb)
        except Exception:
            logger.exception("cb_query: failed to show ask image for chat=%s", chat_id)
        return

    if cmd == "cancel_ask":
        try:
            with _get_chat_lock(chat_id):
                state = _get_chat_state(chat_id)
                state["awaiting_question"] = False
                CHAT_STATE[chat_id] = state
        except Exception:
            logger.exception("cb_query: failed to clear awaiting_question for chat=%s", chat_id)

        # Use restore logic: try to restore the prediction media snapshot if present,
        # otherwise fallback to last media.
        # Prefer saved prediction snapshot so it matches what user saw after days selection.
        pred_media_format = state.get("media_format_at_prediction") or state.get("media_format")
        
        # Определяем пути восстановления
        gif_restore_path = state.get("last_media_gif_pred") or state.get("last_media_gif")
        png_restore_path = state.get("last_media_png_pred") or state.get("last_media_png")

        if pred_media_format == "gif":
            media_path = gif_restore_path or png_restore_path
            is_gif = bool(gif_restore_path)
        else:
            media_path = png_restore_path or gif_restore_path
            is_gif = False

        if not media_path or not os.path.exists(media_path):
            _replace_with_logo(context.bot, chat_id, message_id, caption=state.get("advice_text") or user_interface["captions"]["choose_first"], reply_markup=_kb_first())
            return

        # Build keyboard matching the prediction snapshot's asked_count_at_prediction so toggle shown if it was shown then.
        kb_state = state.copy()
        kb_state["media_format"] = pred_media_format
        kb_state["asked_count"] = state.get("asked_count_at_prediction", state.get("asked_count", 0))
        kb_state["asked_count_at_prediction"] = state.get("asked_count_at_prediction", kb_state["asked_count"])
        kb = _kb_for_state(kb_state)
        caption = state.get("advice_text_pred") or state.get("advice_text", "")
        
        try:
            success = _send_replace_media(context.bot, chat_id, message_id, media_path, is_gif, caption=caption, reply_markup=kb)
            # --- FIX START: Принудительное восстановление путей к графикам в состоянии ---
            if success:
                with _get_chat_lock(chat_id):
                    state = _get_chat_state(chat_id)
                    # Восстанавливаем правильные пути к графикам из snapshot-а (pred),
                    # чтобы стереть путь к ask_question.png, который записался в last_media_png
                    if state.get("last_media_png_pred"):
                        state["last_media_png"] = state["last_media_png_pred"]
                    if state.get("last_media_gif_pred"):
                        state["last_media_gif"] = state["last_media_gif_pred"]
                    CHAT_STATE[chat_id] = state
            # --- FIX END ---
        except Exception:
            logger.exception("cb_query: failed to restore media after cancel for chat=%s", chat_id)
        return

    if cmd == "back":
        # back:first -> go to choose first
        if len(parts) >= 2 and parts[1] == "first":
            try:
                with _get_chat_lock(chat_id):
                    state = _get_chat_state(chat_id)
                    for key in ("last_media_png", "last_media_gif", "last_media"):
                        p = state.get(key)
                        if p and os.path.exists(p):
                            try:
                                os.remove(p)
                            except Exception:
                                logger.exception("cb_query.back:first: failed to remove temp file %s", p)
                    keys_to_pop = ["cached_all_rates", "cached_pair_key", "forecasted_prices",
                                   "forecasted_diffs", "forecast_delta", "advice_text", "forecast_ts",
                                   "media_format", "last_media_png", "last_media_gif", "last_media",
                                   "last_media_png_pred", "last_media_gif_pred", "media_format_at_prediction", "advice_text_pred", "asked_count_at_prediction"]
                    for k in keys_to_pop:
                        state.pop(k, None)
                    state["awaiting_question"] = False
                    state["qa_history"] = []
                    state["asked_count"] = 0
                    CHAT_STATE[chat_id] = state
            except Exception:
                logger.exception("cb_query.back:first: failed to clear cache for chat=%s", chat_id)
            _replace_with_logo(context.bot, chat_id, message_id, caption=user_interface["captions"]["choose_first"], reply_markup=_kb_first())
            return

        # back:second:<first> -> go back to second-selection screen
        if len(parts) >= 3 and parts[1] == "second":
            first = parts[2]
            try:
                with _get_chat_lock(chat_id):
                    state = _get_chat_state(chat_id)
                    for key in ("last_media_png", "last_media_gif", "last_media"):
                        p = state.get(key)
                        if p and os.path.exists(p):
                            try:
                                os.remove(p)
                            except Exception:
                                logger.exception("cb_query.back:second: failed to remove temp file %s", p)
                    keys_to_pop = ["cached_all_rates", "cached_pair_key", "forecasted_prices",
                                   "forecasted_diffs", "forecast_delta", "advice_text", "forecast_ts",
                                   "media_format", "last_media_png", "last_media_gif", "last_media",
                                   "last_media_png_pred", "last_media_gif_pred", "media_format_at_prediction", "advice_text_pred", "asked_count_at_prediction"]
                    for k in keys_to_pop:
                        state.pop(k, None)
                    state["awaiting_question"] = False
                    state["qa_history"] = []
                    state["asked_count"] = 0
                    CHAT_STATE[chat_id] = state
            except Exception:
                logger.exception("cb_query.back:second: failed to clear cache for chat=%s", chat_id)
            _replace_with_logo(context.bot, chat_id, message_id, caption=f"Первая валюта: {first}\n{user_interface['captions']['choose_second']}", reply_markup=_kb_second(first))
            return

        # back:restore -> restore prediction snapshot media (used to return from post-question UI back to graph)
        if len(parts) >= 2 and parts[1] in ("restore", "media"):
            try:
                with _get_chat_lock(chat_id):
                    state = _get_chat_state(chat_id)
            except Exception:
                logger.exception("cb_query.back:restore: failed to get chat state for chat=%s", chat_id)
                return

            pred_media_format = state.get("media_format_at_prediction") or state.get("media_format")
            if pred_media_format == "gif":
                media_path = state.get("last_media_gif_pred") or state.get("last_media_gif") or state.get("last_media_png")
                is_gif = bool(state.get("last_media_gif_pred") or state.get("last_media_gif"))
            else:
                media_path = state.get("last_media_png_pred") or state.get("last_media_png") or state.get("last_media_gif")
                is_gif = False

            if not media_path or not os.path.exists(media_path):
                _replace_with_logo(context.bot, chat_id, message_id, caption=state.get("advice_text") or user_interface["captions"]["choose_first"], reply_markup=_kb_first())
                return

            # Build keyboard corresponding to snapshot at prediction time:
            kb_state = state.copy()
            kb_state["media_format"] = pred_media_format
            kb_state["asked_count"] = state.get("asked_count_at_prediction", state.get("asked_count", 0))
            kb_state["asked_count_at_prediction"] = state.get("asked_count_at_prediction", kb_state["asked_count"])
            kb = _kb_for_state(kb_state)
            caption = state.get("advice_text_pred") or state.get("advice_text", "")
            try:
                _send_replace_media(context.bot, chat_id, message_id, media_path, is_gif, caption=caption, reply_markup=kb)
            except Exception:
                logger.exception("cb_query: failed to restore media on back:restore for chat=%s", chat_id)
            return

    if cmd == "toggle":
        try:
            with _get_chat_lock(chat_id):
                state = _get_chat_state(chat_id)
        except Exception:
            logger.exception("cb_query.toggle: failed to get chat state for chat=%s", chat_id)
            return

        # If awaiting_question or in post-question mode, toggle must be hidden - ignore
        asked_count = int(state.get("asked_count", 0))
        asked_at_pred = int(state.get("asked_count_at_prediction", 0)) if state.get("asked_count_at_prediction") is not None else asked_count
        if state.get("awaiting_question") or (asked_count > asked_at_pred):
            try:
                query.answer(text="Переключение недоступно в этом состоянии", show_alert=False)
            except Exception:
                pass
            return

        # 1. Determine current format and target format
        msg_obj = query.message
        if getattr(msg_obj, "animation", None):
            current = "gif"
        elif getattr(msg_obj, "photo", None):
            current = "png"
        else:
            current = state.get("media_format", "gif")

        target = "png" if current == "gif" else "gif"

        # 2. Prepare media path for target
        if target == "png":
            media_path = state.get("last_media_png")
            is_gif = False
        else:
            media_path = state.get("last_media_gif")
            is_gif = True

        # Generate if missing
        if not media_path or not os.path.exists(media_path):
            try:
                cached_all = state.get("cached_all_rates")
                pair_key = state.get("cached_pair_key")
                old_prices = []
                if cached_all and pair_key:
                    dates = sorted(cached_all.keys())
                    prices = [float(cached_all[d][pair_key]) for d in dates]
                    old_prices = prices[-3:] if len(prices) >= 3 else prices[:]
                else:
                    forecasted = state.get("forecasted_prices", [])
                    if forecasted:
                        old_prices = [forecasted[0]] if forecasted else []

                if target == "png":
                    media_path = _temp_file(user_id, ext="png")
                    plot_sequence(old_prices, state.get("forecasted_prices", []), media_path)
                else:
                    media_path = _temp_file(user_id, ext="gif")
                    try:
                        make_forecast_gif(old_prices, state.get("forecasted_prices", []), media_path)
                    except Exception:
                        media_path = state.get("last_media_png") or media_path
                        is_gif = False
            except Exception:
                logger.exception("cb_query.toggle: failed to regenerate media")
                return

        # 3. Create a temporary state dict to generate the CORRECT future keyboard
        next_state = state.copy()
        next_state["media_format"] = target

        kb = _kb_for_state(next_state)
        caption = state.get("advice_text", "")

        # 4. Send
        success = _send_replace_media(context.bot, chat_id, message_id, media_path, is_gif, caption=caption, reply_markup=kb)

        # 5. Update and Save real state
        if success:
            state["media_format"] = target
            if not is_gif:
                state["last_media_png"] = media_path
            else:
                state["last_media_gif"] = media_path

            try:
                with _get_chat_lock(chat_id):
                    CHAT_STATE[chat_id] = state
            except Exception:
                logger.exception("cb_query.toggle: failed to save chat state")
        return

    query.answer(text="Неизвестная команда", show_alert=False)


def question_message_handler(update, context):
    msg = update.message
    chat_id = msg.chat.id
    user_msg_id = msg.message_id
    text = (msg.text or "").strip()
    if not text:
        return
    with _get_chat_lock(chat_id):
        state = _get_chat_state(chat_id)
        if not state.get("awaiting_question"):
            return
        # mark not awaiting (we will process)
        state["awaiting_question"] = False
        # store previous asked_count snapshot (so we can detect post-question later)
        state.setdefault("asked_count", 0)
        # keep asked_count_at_prediction as is; if missing, set to current asked_count
        state.setdefault("asked_count_at_prediction", state.get("asked_count", 0))
        CHAT_STATE[chat_id] = state
    try:
        context.bot.delete_message(chat_id=chat_id, message_id=user_msg_id)
    except Exception:
        logger.exception("question_message_handler: failed to delete user message %s in chat=%s", user_msg_id, chat_id)
    bot_msg_id = state.get("msg_id")
    try:
        _send_replace_media(context.bot, chat_id, bot_msg_id, AI_THINKING_PATH, is_gif=False, caption=user_interface["captions"]["awaiting_assistant"], reply_markup=None)
    except Exception:
        logger.exception("question_message_handler: failed to show ai_thinking for chat=%s", chat_id)
    try:
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
    except Exception:
        logger.exception("question_message_handler: failed to fetch prices for chat=%s", chat_id)
        prices = []
    last_prices_text = ", ".join(f"{p:.6f}" for p in (prices[-3:] if prices else []))
    forecasted_prices = state.get("forecasted_prices")
    forecast_delta = state.get("forecast_delta")
    advice_text = state.get("advice_text")
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
        summary_text = (
            f"{history_text}"
            f"Pair: {first}/{second}\n"
            f"Latest prices: {last_prices_text}\n"
            f"Forecast days: {days}\n"
            f"Forecasted prices: {forecast_text}\n"
            f"Forecast delta (last vs current): {forecast_delta:.6f}\n"
        )
    else:
        summary_text = f"{history_text}Pair: {first}/{second}\nLatest prices: {last_prices_text}\nForecast days: {days}\n"
    gemini_resp = call_gemini_advice(text, summary_text)
    final_caption = gemini_resp if gemini_resp else "Ассистент не смог предоставить ответ (ошибка подключения к модели)."
    try:
        with _get_chat_lock(chat_id):
            state = _get_chat_state(chat_id)
            qa_history = state.get("qa_history", [])
            qa_history.append({"q": text, "a": final_caption})
            qa_history = qa_history[-5:]
            state["qa_history"] = qa_history
            # increment asked_count (we asked assistant)
            state["asked_count"] = state.get("asked_count", 0) + 1
            CHAT_STATE[chat_id] = state
    except Exception:
        logger.exception("question_message_handler: failed to update QA history for chat=%s", chat_id)
    try:
        state = _get_chat_state(chat_id)

        # Build keyboard for the POST-ANSWER screen:
        # IMPORTANT: For post-answer, toggle must be hidden and back should restore the prediction snapshot.
        # We'll reuse the stored prediction snapshot fields if present.
        media_format_pred = state.get("media_format_at_prediction") or state.get("media_format")
        if media_format_pred == "gif":
            media_path = state.get("last_media_gif_pred") or state.get("last_media_gif") or state.get("last_media_png")
            is_gif = bool(state.get("last_media_gif_pred") or state.get("last_media_gif"))
        else:
            media_path = state.get("last_media_png_pred") or state.get("last_media_png") or state.get("last_media_gif")
            is_gif = False

        # Build keyboard that will show only ask+back (no toggle) for post-answer state.
        kb_state = state.copy()
        # media_format remains as actual displayed media format in state (will be overwritten below)
        kb_state["media_format"] = media_format_pred
        # ensure asked_count_at_prediction is present so _kb_for_state can detect post-question state
        kb_state["asked_count_at_prediction"] = state.get("asked_count_at_prediction", state.get("asked_count", 0))
        kb = _kb_for_state(kb_state)

        if media_path and os.path.exists(media_path):
            success = _send_replace_media(context.bot, chat_id, bot_msg_id, media_path, is_gif, caption=final_caption, reply_markup=kb)
            if success:
                try:
                    with _get_chat_lock(chat_id):
                        state = _get_chat_state(chat_id)
                        # mark that displayed media is whatever we restored
                        state["media_format"] = "gif" if is_gif else "png"
                        state["last_media"] = media_path
                        if is_gif:
                            state["last_media_gif"] = media_path
                        else:
                            state["last_media_png"] = media_path
                        # we also keep advice_text / qa history etc
                        CHAT_STATE[chat_id] = state
                except Exception:
                    logger.exception("question_message_handler: failed to update state after restoring media for chat=%s", chat_id)
        else:
            def try_edit_final():
                context.bot.edit_message_caption(chat_id=chat_id, message_id=bot_msg_id, caption=final_caption, reply_markup=kb)
            _edit_with_retries(try_edit_final, context.bot, chat_id, bot_msg_id)
    except Exception:
        logger.exception("question_message_handler: failed to restore media or edit final caption for chat=%s", chat_id)


def telegram_main(config: dict):
    global TELEGRAM_TOKEN, TEMP_FOLDER, LOGO_PATH, MODELS_PATH, MODELS_SETTINGS, CURRENCIES, user_interface, CACHE_TTL, GEMINI_API_KEY, GEMINI_MODEL, GEMINI_URL, PROMPT_TEMPLATE, ASK_IMG_PATH, AI_THINKING_PATH
    TELEGRAM_TOKEN = config.get('TELEGRAM_TOKEN')
    TEMP_FOLDER = config.get('TEMP_FOLDER')
    LOGO_PATH = config.get('LOGO_PATH')
    ASK_IMG_PATH = config.get('ASK_IMG_PATH')
    AI_THINKING_PATH = config.get('AI_THINKING_PATH')
    MODELS_PATH = config.get('MODELS_PATH')
    MODELS_SETTINGS = config.get('MODELS_SETTINGS')
    CURRENCIES = config.get('CURRENCIES')
    user_interface = config.get('user_interface')
    CACHE_TTL = int(config.get('CACHE_TTL', 300))
    GEMINI_API_KEY = config.get('GEMINI_API_KEY')
    GEMINI_MODEL = config.get('GEMINI_MODEL')
    PROMPT_TEMPLATE = config.get('PROMPT_TEMPLATE')
    if GEMINI_MODEL:
        GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    if TEMP_FOLDER and os.path.isdir(TEMP_FOLDER):
        [os.remove(os.path.join(TEMP_FOLDER, f)) for f in os.listdir(TEMP_FOLDER) if os.path.isfile(os.path.join(TEMP_FOLDER, f))]
    req = Request(con_pool_size=HTTP_POOL_SIZE, connect_timeout=30, read_timeout=30)
    bot = Bot(token=TELEGRAM_TOKEN, request=req)
    updater = Updater(bot=bot, use_context=True)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler("start", start_handler))
    dp.add_handler(CallbackQueryHandler(cb_query))
    dp.add_handler(MessageHandler(Filters.text & (~Filters.command), question_message_handler))
    updater.start_polling()
    updater.idle()
