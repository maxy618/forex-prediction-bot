import os
import time
from datetime import date
from concurrent.futures import ThreadPoolExecutor
from logging_util import setup_logging, exception_rid

logger = setup_logging(level="debug", name=__name__)

from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Updater, CommandHandler, CallbackQueryHandler, MessageHandler, Filters
from telegram.utils.request import Request

from parser import HTTP_POOL_SIZE, fetch_sequences_all_pairs
from plotter import plot_sequence, make_forecast_gif
from model_engine import load_cached_data, find_knn_forecast

from telegram_utils.state import get_chat_lock, get_chat_state, save_chat_state, CHAT_STATE
from telegram_utils.keyboards import kb_first, kb_second, kb_days, kb_confirm, kb_for_state, markup_repr, _make_rows
from telegram_utils.llm_client import call_gemini_advice
from telegram_utils.media_sender import send_replace_media, replace_with_logo, edit_with_retries, is_protected_asset


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

EXECUTOR = None
PROTECTED_ASSETS = set()


def _temp_file(uid, ext="png"):
    os.makedirs(TEMP_FOLDER, exist_ok=True)
    return os.path.join(TEMP_FOLDER, f"{uid}_{int(time.time() * 1000)}.{ext}")


def _clear_chat_media_and_cache(chat_id):
    try:
        with get_chat_lock(chat_id):
            state = get_chat_state(chat_id)
            for key in ("last_media_png", "last_media_gif", "last_media"):
                p = state.get(key)
                if isinstance(p, str) and p and os.path.exists(p):
                    if not is_protected_asset(p, PROTECTED_ASSETS):
                        try:
                            os.remove(p)
                        except Exception:
                            pass
            
            keys_to_pop = [
                "cached_all_rates", "cached_pair_key", "forecasted_prices",
                "forecasted_diffs", "forecast_delta", "advice_text",
                "forecast_ts", "media_format", "last_media_png",
                "last_media_gif", "last_media", "last_media_png_pred",
                "last_media_gif_pred", "media_format_at_prediction",
                "advice_text_pred", "asked_count_at_prediction"
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


def _handle_generic_error(bot, chat_id, message_id, exc):
    rid = exception_rid(logger, "Error", exc=exc)
    try:
        if message_id:
            bot.delete_message(chat_id=chat_id, message_id=message_id)
    except Exception:
        pass
    try:
        bot.send_message(chat_id=chat_id, text=f"{user_interface['captions']['unexpected_error']} (id: {rid})")
    except Exception:
        pass
    _clear_chat_media_and_cache(chat_id)
    caption = user_interface["captions"]["choose_first"]
    markup = kb_first(user_interface, CURRENCIES)
    try:
        with open(LOGO_PATH, "rb") as f:
            msg = bot.send_photo(chat_id=chat_id, photo=f, caption=caption, reply_markup=markup)
            save_chat_state(chat_id, msg.message_id, True, last_caption=caption, last_markup=markup_repr(markup))
    except Exception:
        pass


def _submit_background(bot, chat_id, message_id, fn, *args, **kwargs):
    lock = get_chat_lock(chat_id)
    with lock:
        state = get_chat_state(chat_id)
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
                s = get_chat_state(chat_id)
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


def start_handler(update, context):
    chat_id = update.effective_chat.id
    try:
        with open(LOGO_PATH, "rb") as f:
            kb = InlineKeyboardMarkup([[InlineKeyboardButton(user_interface["buttons"]["warning_label"], callback_data="do_predict")]])
            msg = context.bot.send_photo(chat_id=chat_id, photo=f, caption=user_interface["captions"]["warning"], reply_markup=kb)
        save_chat_state(chat_id, msg.message_id, True, last_caption=user_interface["captions"]["warning"], last_markup=markup_repr(kb))
    except Exception as e:
        _handle_generic_error(context.bot, chat_id, None, e)


def _perform_prediction_and_edit(bot, chat_id, message_id, user_id, first, second, days):
    _clear_chat_media_and_cache(chat_id)
    
    needed_days = MODELS_SETTINGS["knn"]["window_size"] + 5
    all_rates = fetch_sequences_all_pairs(CURRENCIES, days=needed_days)
    pair_key = f"{second}_per_{first}"
    dates = sorted(all_rates.keys())
    prices = [float(all_rates[d][pair_key]) for d in dates]

    if len(prices) < 2:
        raise ValueError("Not enough data")

    diffs = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
    model_path = os.path.join(MODELS_PATH, f"knn_data_{first}{second}.pkl")
    history_diffs = load_cached_data(model_path)
    
    current_sequence = diffs[-MODELS_SETTINGS["knn"]["window_size"]:]
    forecasted_diffs = find_knn_forecast(history_diffs, current_sequence, k=MODELS_SETTINGS["knn"]["k"], horizon=days)
    
    old_prices = prices[-3:]
    last_price = old_prices[-1]
    new_prices = []
    cur = last_price
    for d in forecasted_diffs:
        cur += d
        new_prices.append(cur)

    png_path = _temp_file(user_id, ext="png")
    plot_sequence(old_prices, new_prices, png_path)
    gif_path = _temp_file(user_id, ext="gif")
    try:
        make_forecast_gif(old_prices, new_prices, gif_path)
    except Exception:
        gif_path = png_path

    delta = new_prices[-1] - last_price
    delta_percent = (delta / last_price) * 100 if last_price != 0 else 0
    arrow = "↑" if delta > 0 else "↓" if delta < 0 else "→"
    color_emoji = "🟢" if delta > 0 else "🔴" if delta < 0 else "🟡"
    trend = "рост" if delta > 0 else "падение" if delta < 0 else "без изменений"
    verb = "стоит" if delta > 0 else "не стоит"
    advice = f"Скорее всего {verb} покупать"
    caption = (
        f"{first}/{second} → {new_prices[-1]:.6f}\n"
        f"{color_emoji} Прогноз: {trend}\n"
        f"{arrow} {delta_percent:+.2f}% за {days} дн. ({delta:+.6f})\n{advice}\n\n"
        f"Данные на {date.today().strftime('%d.%m.%Y')}"
    )

    state = get_chat_state(chat_id)
    asked = state.get("asked_count", 0)
    
    with get_chat_lock(chat_id):
        current = get_chat_state(chat_id)
        current.update({
            "first": first, "second": second, "days": days,
            "forecasted_prices": new_prices,
            "forecasted_diffs": forecasted_diffs,
            "forecast_delta": delta,
            "advice_text": caption,
            "forecast_ts": int(time.time()),
            "cached_all_rates": all_rates,
            "cached_pair_key": pair_key,
            "media_format": "gif",
            "media_format_at_prediction": "gif",
            "last_media_png_pred": png_path,
            "last_media_gif_pred": gif_path,
            "advice_text_pred": caption,
            "asked_count_at_prediction": asked
        })
        CHAT_STATE[chat_id] = current

    kb = kb_for_state(user_interface, current)
    success = send_replace_media(bot, chat_id, message_id, gif_path, True, caption, kb, PROTECTED_ASSETS, TEMP_FOLDER)
    if not success:
        raise RuntimeError("Failed to send media")


def cb_query(update, context):
    query = update.callback_query
    data = (query.data or "").strip()
    user_id = update.effective_user.id
    chat_id = query.message.chat.id
    message_id = query.message.message_id

    try:
        try:
            query.answer()
        except: pass

        parts = data.split(":")
        cmd = parts[0]

        if cmd == "do_predict":
            def try_edit():
                context.bot.edit_message_caption(chat_id=chat_id, message_id=message_id, caption=user_interface["captions"]["choose_first"], reply_markup=kb_first(user_interface, CURRENCIES))
            edit_with_retries(try_edit, context.bot, chat_id, message_id)
            return

        if cmd == "first":
            first = parts[1]
            def try_edit():
                context.bot.edit_message_caption(chat_id=chat_id, message_id=message_id, caption=f"Первая валюта: {first}\n{user_interface['captions']['choose_second']}", reply_markup=kb_second(user_interface, CURRENCIES, first))
            edit_with_retries(try_edit, context.bot, chat_id, message_id)
            return

        if cmd == "second":
            _, first, second = parts
            def try_edit():
                context.bot.edit_message_caption(chat_id=chat_id, message_id=message_id, caption=f"Валютная пара: {first}/{second}\n{user_interface['captions']['choose_days']}", reply_markup=kb_days(user_interface, first, second))
            edit_with_retries(try_edit, context.bot, chat_id, message_id)
            return

        if cmd == "days":
            _, first, second, days_str = parts
            days = int(days_str)
            caption = user_interface["captions"].get("confirm_selection", "").format(first=first, second=second, days=days)
            def try_show():
                context.bot.edit_message_caption(chat_id=chat_id, message_id=message_id, caption=caption, reply_markup=kb_confirm(user_interface, first, second, days))
            edit_with_retries(try_show, context.bot, chat_id, message_id)
            return

        if cmd == "confirm":
            _, first, second, days_str = parts
            days = int(days_str)
            def try_edit_pred():
                context.bot.edit_message_caption(chat_id=chat_id, message_id=message_id, caption=user_interface["captions"]["predicting"])
            edit_with_retries(try_edit_pred, context.bot, chat_id, message_id)
            
            if PREDICTING_PATH:
                send_replace_media(context.bot, chat_id, message_id, PREDICTING_PATH, False, user_interface["captions"]["predicting"], None, PROTECTED_ASSETS, TEMP_FOLDER)

            _submit_background(context.bot, chat_id, message_id, _perform_prediction_and_edit, context.bot, chat_id, message_id, user_id, first, second, days)
            return

        if cmd == "ask":
            with get_chat_lock(chat_id):
                state = get_chat_state(chat_id)
                state["awaiting_question"] = True
                CHAT_STATE[chat_id] = state
            
            cancel_kb = _make_rows([[(user_interface["buttons"]["cancel_label"], "cancel_ask")]])
            send_replace_media(context.bot, chat_id, message_id, ASK_IMG_PATH, False, user_interface["captions"]["ask_question"], cancel_kb, PROTECTED_ASSETS, TEMP_FOLDER)
            return

        if cmd == "cancel_ask":
            with get_chat_lock(chat_id):
                state = get_chat_state(chat_id)
                state["awaiting_question"] = False
                CHAT_STATE[chat_id] = state
            
            fmt = state.get("media_format_at_prediction") or "gif"
            is_gif = (fmt == "gif")
            media = state.get("last_media_gif_pred") if is_gif else state.get("last_media_png_pred")
            if not media or (isinstance(media, str) and not os.path.exists(media) and not media.startswith("http")):
                 media = state.get("last_media_gif_pred_file_id") if is_gif else state.get("last_media_png_pred_file_id")

            if not media:
                replace_with_logo(context.bot, chat_id, message_id, LOGO_PATH, user_interface["captions"]["choose_first"], kb_first(user_interface, CURRENCIES))
                return

            kb = kb_for_state(user_interface, state)
            send_replace_media(context.bot, chat_id, message_id, media, is_gif, state.get("advice_text_pred", ""), kb, PROTECTED_ASSETS, TEMP_FOLDER)
            return

        if cmd == "back":
            if parts[1] == "first":
                _clear_chat_media_and_cache(chat_id)
                replace_with_logo(context.bot, chat_id, message_id, LOGO_PATH, user_interface["captions"]["choose_first"], kb_first(user_interface, CURRENCIES))
                return
            if parts[1] == "second":
                first = parts[2]
                _clear_chat_media_and_cache(chat_id)
                replace_with_logo(context.bot, chat_id, message_id, LOGO_PATH, f"Первая валюта: {first}\n{user_interface['captions']['choose_second']}", kb_second(user_interface, CURRENCIES, first))
                return
            if parts[1] == "restore":
                 with get_chat_lock(chat_id):
                    state = get_chat_state(chat_id)
                 fmt = state.get("media_format_at_prediction") or "gif"
                 is_gif = (fmt == "gif")
                 media = state.get("last_media_gif_pred") if is_gif else state.get("last_media_png_pred")
                 if not media or (isinstance(media, str) and not os.path.exists(media)):
                     media = state.get("last_media_gif_pred_file_id") if is_gif else state.get("last_media_png_pred_file_id")
                 
                 kb = kb_for_state(user_interface, state)
                 send_replace_media(context.bot, chat_id, message_id, media, is_gif, state.get("advice_text_pred", ""), kb, PROTECTED_ASSETS, TEMP_FOLDER)
                 return


        if cmd == "toggle":
            with get_chat_lock(chat_id):
                state = get_chat_state(chat_id)
            
            if state.get("running_task"): return

            current_fmt = state.get("media_format", "gif")
            target_fmt = "png" if current_fmt == "gif" else "gif"
            target_is_gif = (target_fmt == "gif")
            
            media_candidate = state.get(f"last_media_{target_fmt}_pred")
            if not media_candidate or (isinstance(media_candidate, str) and not os.path.exists(media_candidate)):
                 media_candidate = state.get(f"last_media_{target_fmt}_pred_file_id")

            if not media_candidate:
                def bg_toggle():
                    with get_chat_lock(chat_id):
                        s = get_chat_state(chat_id)
                        cached = s.get("cached_all_rates")
                        pair_key = s.get("cached_pair_key")
                        if not cached or not pair_key: 
                            return 
                        
                        dates = sorted(cached.keys())
                        prices = [float(cached[d][pair_key]) for d in dates]
                        old_prices = prices[-3:]
                        new_prices = s.get("forecasted_prices", [])
                        
                        new_path = _temp_file(user_id, target_fmt)
                        if target_is_gif:
                            try:
                                make_forecast_gif(old_prices, new_prices, new_path)
                            except:
                                new_path = _temp_file(user_id, "png")
                                plot_sequence(old_prices, new_prices, new_path)
                        else:
                            plot_sequence(old_prices, new_prices, new_path)

                        s_upd = get_chat_state(chat_id)
                        s_upd["media_format"] = target_fmt
                        s_upd[f"last_media_{target_fmt}_pred"] = new_path
                        CHAT_STATE[chat_id] = s_upd
                        
                        kb = kb_for_state(user_interface, s_upd)
                        send_replace_media(context.bot, chat_id, message_id, new_path, target_is_gif, s_upd.get("advice_text", ""), kb, PROTECTED_ASSETS, TEMP_FOLDER)

                _submit_background(context.bot, chat_id, message_id, bg_toggle)
                return

            with get_chat_lock(chat_id):
                s = get_chat_state(chat_id)
                s["media_format"] = target_fmt
                CHAT_STATE[chat_id] = s
            
            kb = kb_for_state(user_interface, state)
            state["media_format"] = target_fmt 
            kb = kb_for_state(user_interface, state)
            
            send_replace_media(context.bot, chat_id, message_id, media_candidate, target_is_gif, state.get("advice_text", ""), kb, PROTECTED_ASSETS, TEMP_FOLDER)
            return

    except Exception as e:
        _handle_generic_error(context.bot, chat_id, message_id, e)


def _background_question_logic(bot, chat_id, bot_msg_id, text):
    with get_chat_lock(chat_id):
        state = get_chat_state(chat_id)
        first = state.get("first")
        second = state.get("second")
        days = state.get("days", 1)
        cached = state.get("cached_all_rates")
        pair_key = state.get("cached_pair_key")
        
        if cached and pair_key:
             dates = sorted(cached.keys())
             prices = [float(cached[d][pair_key]) for d in dates]
        else:
             all_rates = fetch_sequences_all_pairs(CURRENCIES, days=20)
             pair_key = f"{second}_per_{first}"
             dates = sorted(all_rates.keys())
             prices = [float(all_rates[d][pair_key]) for d in dates]

    last_prices_text = ", ".join(f"{p:.6f}" for p in prices[-3:])
    
    with get_chat_lock(chat_id):
        state = get_chat_state(chat_id)
        qa_history = state.get("qa_history", [])
        forecasted = state.get("forecasted_prices", [])
        delta = state.get("forecast_delta", 0.0)

    history_text = "\n".join([f"User: {i['q']}\nAssistant: {i['a']}" for i in qa_history]) if qa_history else "No history yet."
    
    forecast_text = ", ".join(f"{p:.6f}" for p in forecasted)
    context_text = (
        f"Pair: {first}/{second}\nLatest: {last_prices_text}\n"
        f"Forecast ({days} days): {forecast_text}\nDelta: {delta:.6f}"
    )

    gemini_resp = call_gemini_advice(GEMINI_API_KEY, GEMINI_URL, PROMPT_TEMPLATE, text, history_text, context_text)
    final_caption = gemini_resp or "Ассистент не смог предоставить ответ."

    with get_chat_lock(chat_id):
        state = get_chat_state(chat_id)
        state["qa_history"] = (state.get("qa_history", []) + [{"q": text, "a": final_caption}])[-5:]
        state["asked_count"] = state.get("asked_count", 0) + 1
        CHAT_STATE[chat_id] = state

        fmt = state.get("media_format_at_prediction") or "gif"
        is_gif = (fmt == "gif")
        media = state.get("last_media_gif_pred") if is_gif else state.get("last_media_png_pred")
        if not media or (isinstance(media, str) and not os.path.exists(media)):
             media = state.get("last_media_gif_pred_file_id") if is_gif else state.get("last_media_png_pred_file_id")
        
        kb = kb_for_state(user_interface, state)

    if media:
        send_replace_media(bot, chat_id, bot_msg_id, media, is_gif, final_caption, kb, PROTECTED_ASSETS, TEMP_FOLDER)
    else:
        try:
            bot.edit_message_caption(chat_id=chat_id, message_id=bot_msg_id, caption=final_caption, reply_markup=kb)
        except: pass


def question_message_handler(update, context):
    msg = update.message
    chat_id = msg.chat.id
    try:
        context.bot.delete_message(chat_id=chat_id, message_id=msg.message_id)
    except: pass

    text = (msg.text or "").strip()
    if not text: return

    with get_chat_lock(chat_id):
        state = get_chat_state(chat_id)
        if not state.get("awaiting_question") or state.get("running_task"):
            return
        state["awaiting_question"] = False
        CHAT_STATE[chat_id] = state
    
    bot_msg_id = state.get("msg_id")
    send_replace_media(context.bot, chat_id, bot_msg_id, AI_THINKING_PATH, False, user_interface["captions"]["awaiting_assistant"], None, PROTECTED_ASSETS, TEMP_FOLDER)
    _submit_background(context.bot, chat_id, bot_msg_id, _background_question_logic, context.bot, chat_id, bot_msg_id, text)


def telegram_main(config: dict):
    global TELEGRAM_TOKEN, TEMP_FOLDER, LOGO_PATH, MODELS_PATH, MODELS_SETTINGS, CURRENCIES, user_interface, CACHE_TTL, GEMINI_API_KEY, GEMINI_MODEL, GEMINI_URL, PROMPT_TEMPLATE, ASK_IMG_PATH, AI_THINKING_PATH, PREDICTING_PATH
    global EXECUTOR, PROTECTED_ASSETS

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

    PROTECTED_ASSETS = {LOGO_PATH, ASK_IMG_PATH, AI_THINKING_PATH, PREDICTING_PATH}
    
    if TEMP_FOLDER and os.path.isdir(TEMP_FOLDER):
        for f in os.listdir(TEMP_FOLDER):
            p = os.path.join(TEMP_FOLDER, f)
            if os.path.isfile(p):
                try:
                    os.remove(p)
                except: pass

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