import os
import random
import csv
import pickle
import time
import json

from collections import defaultdict, Counter
from datetime import date, timedelta

import numpy as np
import requests
from requests.adapters import HTTPAdapter, Retry

from dotenv import load_dotenv
load_dotenv()

from logging_util import setup_logging, exception_rid
logger = setup_logging(level="debug", name=__name__)

from matplotlib import use
use("Agg")
import matplotlib.pyplot as plt

import threading
from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup, InputMediaPhoto
from telegram.error import BadRequest
from telegram.ext import Updater, CommandHandler, CallbackQueryHandler, MessageHandler, Filters
from telegram.utils.request import Request


# ==============================
# CONFIG
# ==============================
CURRENCIES = ["EUR", "USD", "RUB"]

MODELS_SETTINGS = {
    "REBUILD": True,
    "reg": {
        "min_n": 3,
        "max_n": 10,
    },
    "markov": {
        "min_n": 3,
        "max_n": 10,
        "k" : 0.2,
    }
}

CACHE_TTL = int(os.getenv("CACHE_TTL", 300))

user_interface = {
    "captions": {
        "warning": "Прогноз — это лишь вероятность, а не гарантия. Используй результат как подсказку, а не как финансовый совет",
        "choose_first": "Выбери первую валюту",
        "choose_second": "Выбери вторую валюту",
        "choose_days": "Выберите количество дней (1-9)",
        "predicting": "Выполняется прогноз, секунду...",
        "ask_question": "Отправьте вопрос по графику",
        "awaiting_assistant": "Ожидайте ответ ассистента...",
        "unexpected_error": "❌ Произошла непредвиденная ошибка, попробуйте снова",
        "send_logo_caption": "Нажми чтобы сделать прогноз",
    },
    "buttons": {
        "back_label": "Назад",
        "ask_label_first": "Задать вопрос",
        "ask_label_more": "Ещё вопрос",
        "cancel_label": "Отмена",
        "currency_codes": CURRENCIES,
        "days": [str(i) for i in range(1, 10)],
        "warning_label": "Я понял(-а)"
    }
}

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

DATASETS_PATH = "../datasets/"
MODELS_PATH = "../models/"
TEMP_FOLDER = "../temp/"
LOGO_PATH = "../assets/logo.png"

BASE_LATEST = "https://www.cbr-xml-daily.ru/daily_json.js"
BASE_ARCHIVE = "https://www.cbr-xml-daily.ru/archive"

HTTP_POOL_SIZE = int(os.getenv("HTTP_POOL_SIZE", 10))
SESSION = requests.Session()
retry_strategy = Retry(
    total=3,
    backoff_factor=0.3,
    status_forcelist=(429, 500, 502, 503, 504),
    allowed_methods=("GET",)
)
adapter = HTTPAdapter(pool_connections=HTTP_POOL_SIZE, pool_maxsize=HTTP_POOL_SIZE, max_retries=retry_strategy)
SESSION.mount("https://", adapter)
SESSION.mount("http://", adapter)
SESSION.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36"})

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

PROMPT_TEMPLATE = (
    """
Ты — casual ассистент для телеграм бота прогнозов валютных пар. Отвечай прямо, без лишних оговорок и формальностей.

Основные правила:
- Отвечай на любой вопрос, даже если он не напрямую о прогнозе. Если вопрос про общие валютные тренды, экономику или стратегии — помогай, но привязывай к текущему прогнозу если возможно.
- Технические детали (регрессия, марковские цепи) упоминай только если пользователь прямо спросил "как это работает" или "по какой методе".
- В обычных ответах просто давай числа, тренд и краткую оценку — никаких "это не рекомендация" и похожих фраз.
- Ответ: 4–6 предложений максимум. Будь дружелюбен, но деловит.

Что показываешь:
- Последние цены, прогнозные значения, абсолютные/процентные изменения
- Простые выводы: "растет / падает", "выгодно купить / продать" с числами
- Отвечай на вопросы про стратегию, риски, альтернативы — ты не робот, ты нормальный советчик

Что НЕ делаешь:
- Не придумываешь данные, которых нет в summary_text
- Не говоришь "я не могу это обсуждать" — найди способ помочь в рамках контекста

Формат входа (summary_text):
Pair: EUR/USD
Latest prices: 100.123456, 100.234567, 100.345678
Forecast days: 3
Forecasted prices: 100.400000, 100.450000, 100.500000
Forecast delta (last vs current): 0.154322
Advice (bot): стоит покупать — прогноз роста

Читай summary_text, слушай вопрос пользователя и отвечай как нормальный человек.
"""
)


# ==============================
# UTILS
# ==============================
def save_model(model, path_to_model):
    with open(path_to_model, 'wb') as file:
        pickle.dump(model, file)


def load_model(path_to_model):
    if not os.path.exists(path_to_model):
        raise FileNotFoundError(f"{path_to_model} does not exist")
    with open(path_to_model, 'rb') as file:
        model = pickle.load(file)
        return model


def read_column(column_name, csv_path):
    logger.debug("read_column called column=%s csv_path=%s", column_name, csv_path)
    values = []
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            values.append(row[column_name])
    return values


# ==============================
# MARKOV CHAIN
# ==============================
def counts_to_probabilities(counter, k=MODELS_SETTINGS["markov"]["k"]):
    laplace_smoothing = lambda m, n, k, v: (m + k) / (n + k * v)
    n = sum(counter.values())
    v = len(counter)
    
    probs = {}
    for key, count in counter.items():
        p = laplace_smoothing(count, n, k, v)
        probs[key] = p
    
    return probs


def build_markov_model(sequence, order=1):
    logger.debug("build_markov_model called sequence_len=%s order=%s", len(sequence), order)
    if order >= len(sequence):
        raise ValueError("sequence length should be > order")
    
    counters = defaultdict(Counter)
    
    for idx in range(len(sequence) - order):
        current_state = tuple(sequence[idx:idx+order])
        next_state = sequence[idx+order]
        counters[current_state][next_state] += 1
    
    model = {}
    for state, counter in counters.items():
        model[state] = counts_to_probabilities(counter)
    
    model = {
        "order": order,
        "table": model
        }
    
    logger.debug("build_markov_model finished order=%s states=%d", order, len(model))
    return model


def predict_state(sequence, model):
    order = model["order"]
    table = model["table"]

    if len(sequence) < order:
        raise ValueError("sequence length should be >= order")

    current_state = tuple(sequence[-order:])

    next_states = table.get(current_state)
    if next_states is None:
        return None

    rand_num = random.random()
    cumul_sum = 0

    for sign, prob in next_states.items():
        cumul_sum += prob
        if rand_num <= cumul_sum:
            return sign

    winner = max(next_states, key=next_states.get)
    return winner


def predict_ensemble_sign(sequence, models):
    votes = {"+": 0, "-": 0}
    for model in models:
        next_state = predict_state(sequence, model)
        if next_state is None:
            continue
        votes[next_state] += 1
    
    max_votes = max(votes.values())
    leaders = [k for k, v in votes.items() if v == max_votes]
    choice = random.choice(leaders)
    return choice


def forecast_signs(sequence, models, n=1):
    logger.debug("forecast_signs called seq_len=%s models=%s n=%s", len(sequence), len(models), n)
    result = []
    curr_seq = sequence.copy()
    
    for _ in range(n):
        next_state = predict_ensemble_sign(curr_seq, models)
        result.append(next_state)
        curr_seq.append(next_state)
    
    logger.debug("forecast_signs finished produced=%s", result)
    return result


# ==============================
# REGRESSION
# ==============================
def build_regression(diffs_list, n_lags=1):
    logger.debug("build_regression called len=%s n_lags=%s", len(diffs_list), n_lags)
    if len(diffs_list) <= n_lags:
        raise ValueError(f"diffs_list length should be > n_lags")  
    
    x = []
    y = []
    for i in range(n_lags, len(diffs_list)):
        prev_diffs = diffs_list[i - n_lags:i]
        curr_diff  = diffs_list[i]
        x.append(prev_diffs)
        y.append(curr_diff)

    X_mat = np.array(x, dtype=float)
    y_vec = np.array(y, dtype=float).reshape(-1, 1)

    ones = np.ones((X_mat.shape[0], 1))
    X_with_bias = np.hstack([X_mat, ones])

    Xt = X_with_bias.T
    XtX = Xt @ X_with_bias
    Xty = Xt @ y_vec

    coeffs = np.linalg.inv(XtX) @ Xty

    coeffs_out = coeffs.flatten()
    logger.debug("build_regression finished coeffs_len=%s", len(coeffs_out))
    return coeffs_out


def predict_diff(last_diffs, coeffs):
    n_lags = len(coeffs) - 1
    
    if len(last_diffs) < n_lags:
        raise ValueError("last_diffs length is smaller than required n_lags")
    
    weights = coeffs[:-1]
    bias = coeffs[-1]

    inputs = last_diffs[-n_lags:]

    prediction = 0
    for i in range(n_lags):
        prediction += weights[i] * inputs[i]
    prediction += bias

    return prediction


def predict_ensemble_diff(last_diffs, models):
    results = []
    for coeffs in models:
        res = predict_diff(last_diffs, coeffs)
        results.append(res)
    
    avg = sum(results) / len(results)
    return avg


def forecast_diffs(last_diffs, models, n=1):
    logger.debug("forecast_diffs called last_diffs_len=%s models=%s n=%s", len(last_diffs), len(models), n)
    result = []
    diffs = last_diffs.copy()

    for _ in range(n):
        next_diff = predict_ensemble_diff(diffs, models)
        result.append(next_diff)
        diffs.append(next_diff)

    logger.debug("forecast_diffs finished produced=%s", result)
    return result


# ==============================
# PARSER
# ==============================
def get_dates_list(num_days):
    logger.debug("get_dates_list called num_days=%s", num_days)
    today = date.today()
    start = today - timedelta(days=num_days-1)
    
    day = start
    res = []
    while day <= today:
        res.append(day)
        day += timedelta(days=1)
    logger.debug("get_dates_list finished len=%s start=%s end=%s", len(res), res[0] if res else None, res[-1] if res else None)
    return res


def fetch_for_date(d, base_latest, base_archive):
    logger.debug("fetch_for_date called date=%s", d)
    if d == date.today():
        url = base_latest
    else:
        url = f"{base_archive}/{d.year}/{d.month:02d}/{d.day:02d}/daily_json.js"
    try:
        r = SESSION.get(url, timeout=6)
        if r.status_code == 200:
            try:
                json_res = r.json()
                logger.debug("fetch_for_date success url=%s keys=%s", url, list(json_res.keys())[:5])
                return json_res
            except ValueError as e:
                logger.warning("fetch_for_date: invalid json for %s: %s", url, e)
    except Exception as e:
        logger.exception("fetch_for_date: initial request failed for %s", url)
    try:
        time.sleep(0.2)
        r = SESSION.get(url, timeout=6)
        if r.status_code == 200:
            try:
                json_res = r.json()
                logger.debug("fetch_for_date retry success url=%s keys=%s", url, list(json_res.keys())[:5])
                return json_res
            except ValueError:
                pass
    except Exception as e:
        logger.exception("fetch_for_date: retry request failed for %s", url)
    if url != base_latest:
        try:
            r = SESSION.get(base_latest, timeout=6)
            if r.status_code == 200:
                try:
                    return r.json()
                except ValueError as e:
                    logger.warning("fetch_for_date: invalid json for base_latest %s: %s", base_latest, e)
        except Exception as e:
            logger.exception("fetch_for_date: fallback request failed for %s", base_latest)
    logger.debug("fetch_for_date finished returning None for date=%s", d)
    return None


def rub_per_one_from_json(js, code):
    logger.debug("rub_per_one_from_json called code=%s present=%s", code, bool(js and "Valute" in js))
    if code == "RUB":
        return 1.0
    if not js or "Valute" not in js:
        return None
    info = js["Valute"].get(code)
    if not info:
        return None
    nominal = int(info.get("Nominal", 1))
    value = float(info.get("Value"))
    logger.debug("rub_per_one_from_json for code=%s value=%s nominal=%s result=%s", code, value, nominal, value/nominal)
    return value / nominal


def fetch_sequences_all_pairs(currencies, days=max(MODELS_SETTINGS["markov"]["max_n"], MODELS_SETTINGS["reg"]["max_n"])):
    logger.debug("fetch_sequences_all_pairs called currencies=%s days=%s", currencies, days)
    dates = get_dates_list(days)
    last_known = {c: None for c in currencies}

    res = {}
    for d in dates:
        js = fetch_for_date(d, BASE_LATEST, BASE_ARCHIVE)
        if js is None:
            time.sleep(0.2)
            js = fetch_for_date(d, BASE_LATEST, BASE_ARCHIVE)

        if js is not None:
            for c in currencies:
                val = rub_per_one_from_json(js, c)
                if val is not None:
                    last_known[c] = val
        else:
            raise RuntimeError("fetched empty json")

        for c in currencies:
            if last_known[c] is None:
                raise RuntimeError(f"no data for {c} on / before {d.isoformat()}")

        day_rates = {}
        for base in currencies:
            for target in currencies:
                pair = f"{target}_per_{base}"
                if base == target:
                    rate = 1.0
                else:
                    B = last_known[base]
                    T = last_known[target]
                    rate = B / T
                day_rates[pair] = float(rate)
        res[d.isoformat()] = day_rates

    logger.debug("fetch_sequences_all_pairs finished days=%s result_dates=%s", len(res), list(res.keys())[:3])
    return res


# ==============================
# PLOTTER
# ==============================
def make_axes_limits(prices):
    logger.debug("make_axes_limits called len=%s", len(prices))
    min_price = min(prices)
    max_price = max(prices)
    diff = max_price - min_price

    if diff == 0:
        diff = min_price * 0.01 if min_price != 0 else 1

    low = min_price - diff * 0.5
    high = max_price + diff * 0.5

    logger.debug("make_axes_limits returned low=%s high=%s", low, high)
    return low, high


def plot_sequence(old_prices, new_prices, filename):
    logger.debug("plot_sequence called old_len=%s new_len=%s filename=%s", len(old_prices), len(new_prices), filename)
    all_prices = old_prices + new_prices
    if not all_prices:
        raise ValueError("no prices to plot")

    y_min, y_max = make_axes_limits(all_prices)

    color_new = "green" if new_prices and new_prices[-1] > old_prices[-1] else "red"

    m = len(old_prices)
    n = len(new_prices)
    total = m + n

    start_date = date.today() - timedelta(days=max(0, m - 1))
    dates = [start_date + timedelta(days=i) for i in range(total)]
    labels = [d.strftime("%d.%m") for d in dates]

    plt.figure(figsize=(5, 3), facecolor="black")
    ax = plt.gca()
    ax.set_facecolor("black")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    ax.tick_params(colors="white", labelsize=8)
    plt.ylim(y_min, y_max)

    old_x = list(range(m))
    plt.plot(old_x, old_prices, color="white", linewidth=2)

    if n > 0:
        new_x = list(range(m - 1, total))
        new_y = [old_prices[-1]] + new_prices
        plt.plot(new_x, new_y, color=color_new, linewidth=2)

    plt.xticks(ticks=list(range(total)), labels=labels, rotation=45, fontsize=7)
    plt.tight_layout()

    out_dir = os.path.dirname(filename) or "."
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(filename, bbox_inches="tight", facecolor="black")
    plt.close()

    logger.debug("plot_sequence saved filename=%s", filename)
    return filename


# ==============================
# TELEGRAM
# ==============================
CHAT_STATE = {}
CHAT_LOCKS = {}


def _get_chat_lock(chat_id):
    logger.debug("_get_chat_lock called chat_id=%s", chat_id)
    lock = CHAT_LOCKS.get(chat_id)
    if lock is None:
        lock = threading.Lock()
        CHAT_LOCKS[chat_id] = lock
    logger.debug("_get_chat_lock returning lock for chat_id=%s", chat_id)
    return lock


def _markup_repr(reply_markup):
    """Return a lightweight normalized representation of InlineKeyboardMarkup
       so we can compare whether reply_markup actually changed."""
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
        logger.debug("_markup_repr failed to normalize reply_markup, returning str")
        return str(reply_markup)


def _temp_file(user_id):
    logger.debug("_temp_file called user_id=%s", user_id)
    os.makedirs(TEMP_FOLDER, exist_ok=True)
    out = os.path.join(TEMP_FOLDER, f"{user_id}_{int(time.time()*1000)}.png")
    logger.debug("_temp_file returning %s", out)
    return out


def _kb_first():
    logger.debug("_kb_first called")
    codes = user_interface["buttons"].get("currency_codes", CURRENCIES)
    return _make_rows([[(c, f"first:{c}") for c in codes]])


def _kb_second(first):
    logger.debug("_kb_second called first=%s", first)
    codes = user_interface["buttons"].get("currency_codes", CURRENCIES)
    buttons = [(c, f"second:{first}:{c}") for c in codes if c != first]
    buttons.append((user_interface["buttons"]["back_label"], "back:first"))
    return _make_rows([buttons])


def _kb_days(first, second):
    logger.debug("_kb_days called first=%s second=%s", first, second)
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
    logger.debug("_make_rows called rows=%s", len(pairs))
    keyboard = []
    for row in pairs:
        keyboard.append([InlineKeyboardButton(text, callback_data=cb) for text, cb in row])
    return InlineKeyboardMarkup(keyboard)


def _save_chat_state(chat_id, message_id, has_logo, **kwargs):
    logger.debug("_save_chat_state called chat_id=%s message_id=%s has_logo=%s kwargs=%s", chat_id, message_id, has_logo, list(kwargs.keys()))
    state = CHAT_STATE.get(chat_id, {})
    state.update({"msg_id": int(message_id), "has_logo": bool(has_logo)})
    for k, v in kwargs.items():
        state[k] = v
    CHAT_STATE[chat_id] = state


def _get_chat_state(chat_id):
    logger.debug("_get_chat_state called chat_id=%s", chat_id)
    return CHAT_STATE.get(chat_id, {"msg_id": None, "has_logo": False, "awaiting_question": False})


def _send_start_message(bot: Bot, chat_id: int):
    logger.debug("_send_start_message called chat_id=%s", chat_id)
    if os.path.exists(LOGO_PATH):
        try:
            with open(LOGO_PATH, "rb") as f:
                kb = InlineKeyboardMarkup(
                    [[InlineKeyboardButton(user_interface["buttons"]["warning_label"], callback_data="do_predict")]]
                )
                msg = bot.send_photo(
                    chat_id=chat_id,
                    photo=f,
                    caption=user_interface["captions"]["warning"],
                    reply_markup=kb
                )
            _save_chat_state(chat_id, msg.message_id, True,
                             last_caption=user_interface["captions"]["warning"],
                             last_markup=_markup_repr(kb))
            logger.debug("_send_start_message sent photo chat_id=%s message_id=%s", chat_id, msg.message_id)
            return msg
        except Exception as e:
            logger.exception("_send_start_message: failed to send photo to %s", chat_id)
    kb = InlineKeyboardMarkup(
        [[InlineKeyboardButton(user_interface["buttons"]["warning_label"], callback_data="do_predict")]]
    )
    msg = bot.send_message(
        chat_id=chat_id,
        text=user_interface["captions"]["warning"],
        reply_markup=kb
    )
    _save_chat_state(chat_id, msg.message_id, False,
                     last_caption=user_interface["captions"]["warning"],
                     last_markup=_markup_repr(kb))
    logger.debug("_send_start_message sent text chat_id=%s message_id=%s", chat_id, msg.message_id)
    return msg


def _edit_with_retries(action_callable, bot: Bot, chat_id: int, message_id: int,
                       max_attempts: int = 3, delay: float = 1.0):
    logger.debug("_edit_with_retries called chat_id=%s message_id=%s max_attempts=%s", chat_id, message_id, max_attempts)
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
                # If message is not modified — nothing to do
                if "Message is not modified" in msg:
                    logger.debug("_edit_with_retries: ignored BadRequest 'Message is not modified' for chat=%s msg=%s", chat_id, message_id)
                    return True
                # If message not found — no point in retrying
                if "Message to edit not found" in msg or "Message to delete not found" in msg or "message to edit not found" in msg.lower():
                    logger.warning("_edit_with_retries: message to edit not found for chat=%s msg=%s", chat_id, message_id)
                    break
                logger.exception("_edit_with_retries: BadRequest attempt %s failed", attempt, exc_info=be)
            except Exception as e:
                last_exc = e
                logger.exception("_edit_with_retries: attempt %s failed", attempt)
            if attempt < max_attempts:
                time.sleep(delay)

    # final fallback: try to delete old message and inform user
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


def start_handler(update, context):
    logger.debug("start_handler called chat=%s", getattr(update, 'effective_chat', None) and update.effective_chat.id)
    chat_id = update.effective_chat.id
    kb = InlineKeyboardMarkup([[InlineKeyboardButton(user_interface["buttons"]["warning_label"], callback_data="do_predict")]])
    if os.path.exists(LOGO_PATH):
        try:
            with open(LOGO_PATH, "rb") as f:
                msg = context.bot.send_photo(chat_id=chat_id, photo=f, caption=user_interface["captions"]["warning"], reply_markup=kb)
            _save_chat_state(chat_id, msg.message_id, True,
                             last_caption=user_interface["captions"]["warning"],
                             last_markup=_markup_repr(kb))
            return
        except Exception as e:
            logger.exception("start_handler: failed to send photo to %s", chat_id)
    try:
        msg = context.bot.send_message(chat_id=chat_id, text=user_interface["captions"]["warning"], reply_markup=kb)
        _save_chat_state(chat_id, msg.message_id, False,
                 last_caption=user_interface["captions"]["warning"],
                 last_markup=_markup_repr(kb))
    except Exception as e:
        logger.exception("start_handler: failed to send warning message to %s", chat_id)


def _replace_with_logo(bot, chat_id, message_id, caption=None, reply_markup=None, max_attempts=3, delay=1.0):
    logger.debug("_replace_with_logo called chat=%s message_id=%s caption_len=%s", chat_id, message_id, len(caption) if caption else 0)
    state = _get_chat_state(chat_id)
    if state["has_logo"]:
        def try_edit_caption():
            bot.edit_message_caption(chat_id=chat_id, message_id=message_id, caption=caption, reply_markup=reply_markup)
        success = _edit_with_retries(try_edit_caption, bot, chat_id, message_id, max_attempts=max_attempts, delay=delay)
        if not success:
            return
        return

    if os.path.exists(LOGO_PATH):
        def try_edit_media():
            with open(LOGO_PATH, "rb") as f:
                media = InputMediaPhoto(f, caption=caption)
                bot.edit_message_media(media=media, chat_id=chat_id, message_id=message_id, reply_markup=reply_markup)
        success = _edit_with_retries(try_edit_media, bot, chat_id, message_id, max_attempts=max_attempts, delay=delay)
        if success:
            _save_chat_state(chat_id, message_id, True)
            return
        return

    def try_edit_caption_no_logo():
        bot.edit_message_caption(chat_id=chat_id, message_id=message_id, caption=caption, reply_markup=reply_markup)
    success = _edit_with_retries(try_edit_caption_no_logo, bot, chat_id, message_id, max_attempts=max_attempts, delay=delay)
    if success:
        _save_chat_state(chat_id, message_id, False)
    return


def _edit_or_send_media(bot, chat_id, message_id, caption=None, media_path=None, reply_markup=None, max_attempts=3, delay=1.0):
    logger.debug("_edit_or_send_media called chat=%s message_id=%s media_path=%s reply_markup=%s", chat_id, message_id, media_path, bool(reply_markup))
    if media_path and os.path.exists(media_path):
        def try_edit_media():
            with open(media_path, "rb") as f:
                media = InputMediaPhoto(f, caption=caption)
                bot.edit_message_media(media=media, chat_id=chat_id, message_id=message_id, reply_markup=reply_markup)
        success = _edit_with_retries(try_edit_media, bot, chat_id, message_id, max_attempts=max_attempts, delay=delay)
        if success:
            _save_chat_state(chat_id, message_id, False)
            return
        try:
            with open(media_path, "rb") as f:
                new_msg = bot.send_photo(chat_id=chat_id, photo=f, caption=caption, reply_markup=reply_markup)
            try:
                bot.delete_message(chat_id=chat_id, message_id=message_id)
            except Exception as e:
                logger.exception("_edit_or_send_media: failed to delete old media message chat=%s msg=%s", chat_id, message_id)
            _save_chat_state(chat_id, new_msg.message_id, False,
                             last_caption=caption or "",
                             last_markup=_markup_repr(reply_markup))
            return
        except Exception as e:
            logger.exception("_edit_or_send_media: send media failed for chat=%s path=%s", chat_id, media_path)
            try:
                try:
                    bot.delete_message(chat_id=chat_id, message_id=message_id)
                except Exception as e:
                    logger.exception("_edit_or_send_media: failed to delete old message on fallback chat=%s msg=%s", chat_id, message_id)
                _save_chat_state(chat_id, new_msg.message_id, False,
                                 last_caption=caption or "",
                                 last_markup=_markup_repr(reply_markup))
                return
            except Exception as e:
                logger.exception("_edit_or_send_media: fully failed to send for chat=%s", chat_id)
                try:
                    bot.delete_message(chat_id=chat_id, message_id=message_id)
                except Exception as e:
                    logger.exception("_edit_or_send_media: failed delete after full failure chat=%s msg=%s", chat_id, message_id)
                try:
                    rid = exception_rid(logger, "_edit_or_send_media final failure", exc=e)
                    bot.send_message(chat_id=chat_id, text=f"{user_interface['captions']['unexpected_error']} (id: {rid})")
                except Exception as e:
                    logger.exception("_edit_or_send_media: failed to send unexpected_error to chat=%s", chat_id)
                try:
                    _send_start_message(bot, chat_id)
                except Exception as e:
                    logger.exception("_edit_or_send_media: failed to send start message to chat=%s", chat_id)
                return
    else:
        def try_edit_caption():
            bot.edit_message_caption(chat_id=chat_id, message_id=message_id, caption=caption, reply_markup=reply_markup)
        success = _edit_with_retries(try_edit_caption, bot, chat_id, message_id, max_attempts=max_attempts, delay=delay)
        if success:
            state = _get_chat_state(chat_id)
            _save_chat_state(chat_id, message_id, state.get("has_logo", False))
            return
        return


def call_gemini_advice(question_text: str, summary_text: str):
    logger.debug("call_gemini_advice called question_len=%s summary_len=%s", len(question_text or ""), len(summary_text or ""))
    prompt = PROMPT_TEMPLATE + summary_text + "\n\nUser question:\n" + question_text + "\n\nОтвет:"
    payload = {
        "contents": [
            {"parts": [{"text": prompt}]}
        ],
    }
    headers = {"Content-Type": "application/json", "x-goog-api-key": GEMINI_API_KEY}
    try:
        r = SESSION.post(GEMINI_URL, headers=headers, data=json.dumps(payload), timeout=30)
        r.raise_for_status()
    except Exception as e:
        logger.exception("call_gemini_advice: request to model failed")
        return None

    try:
        j = r.json()
    except Exception as e:
        logger.exception("call_gemini_advice: failed to parse json response")
        return None

    try:
        found = j["candidates"][0]["content"]["parts"][0]["text"].strip()
        logger.debug("call_gemini_advice returned len=%s", len(found))
        return found
    except Exception as e:
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
    logger.debug("_perform_prediction_and_edit called chat=%s user=%s pair=%s/%s days=%s", chat_id, user_id, first, second, days)
    # clear old cache before computing a new forecast (safety)
    try:
        with _get_chat_lock(chat_id):
            state = _get_chat_state(chat_id)
            state.pop("cached_all_rates", None)
            state.pop("cached_pair_key", None)
            state.pop("forecasted_prices", None)
            state.pop("forecasted_diffs", None)
            state.pop("forecast_delta", None)
            state.pop("advice_text", None)
            state.pop("forecast_ts", None)
            CHAT_STATE[chat_id] = state
    except Exception:
        # non-critical: log and continue
        logger.exception("_perform_prediction_and_edit: failed to clear previous cache for chat=%s", chat_id)

    try:
        needed_days = max(MODELS_SETTINGS["reg"]["max_n"], MODELS_SETTINGS["markov"]["max_n"])
    except Exception as e:
        needed_days = days if days > 0 else 10

    try:
        all_rates = fetch_sequences_all_pairs(CURRENCIES, days=needed_days)
    except Exception as e:
        def try_edit_error():
            bot.edit_message_caption(chat_id=chat_id, message_id=message_id,
                                     caption=f"Ошибка при получении актуальных данных: {e}")
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
            bot.edit_message_caption(chat_id=chat_id, message_id=message_id,
                                     caption=f"Ошибка при формировании последовательности цен: {e}")
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

    out_path = _temp_file(user_id)
    plot_sequence(old_prices, new_prices, out_path)

    delta = new_prices[-1] - last_price
    sign_text = "вырастет" if delta > 0 else "упадет"
    verb = "стоит" if delta > 0 else "не стоит"
    advice = f"Скорее всего {verb} покупать — цена {sign_text} на {abs(delta):.6f}"

    state = _get_chat_state(chat_id)
    state.setdefault("qa_history", [])
    state.setdefault("asked_count", 0)

    ask_cb = f"ask:{first}:{second}:{days}"
    ask_label = user_interface["buttons"]["ask_label_first"] if state.get("asked_count", 0) == 0 else user_interface["buttons"]["ask_label_more"]
    kb = _make_rows([[ (user_interface["buttons"]["back_label"], "back:first"), (ask_label, ask_cb) ]])
    _edit_or_send_media(bot, chat_id, message_id, caption=f"{first}/{second}\n{advice}", media_path=out_path, reply_markup=kb)

    # --- save cached data so follow-up question replies use the exact same prices
    state.update({
        "last_media": out_path,
        "first": first,
        "second": second,
        "days": days,
        "awaiting_question": False,
        "forecasted_prices": [float(x) for x in new_prices],
        "forecasted_diffs": [float(x) for x in adjusted],
        "forecast_delta": float(delta),
        "advice_text": advice,
        "forecast_ts": int(time.time()),
        "cached_all_rates": all_rates,
        "cached_pair_key": pair_key
    })
    # persist state under lock
    try:
        with _get_chat_lock(chat_id):
            CHAT_STATE[chat_id] = state
    except Exception:
        logger.exception("_perform_prediction_and_edit: failed to save chat state for chat=%s", chat_id)

    try:
        os.remove(out_path)
    except Exception as e:
        logger.exception("_perform_prediction_and_edit: failed to remove temp file %s", out_path)


def cb_query(update, context):
    logger.debug("cb_query called user=%s data=%s", getattr(update.callback_query, 'from_user', None), getattr(update.callback_query, 'data', None))
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
            context.bot.edit_message_caption(chat_id=chat_id, message_id=message_id, caption=f"Первая валюта: {first}\n{user_interface["captions"]["choose_second"]}", reply_markup=_kb_second(first))
        _edit_with_retries(try_edit, context.bot, chat_id, message_id)
        return

    if cmd == "second" and len(parts) == 3:
        _, first, second = parts
        def try_edit():
            context.bot.edit_message_caption(chat_id=chat_id, message_id=message_id, caption=f"Валютная пара: {first}/{second}\n{user_interface["captions"]["choose_days"]}", reply_markup=_kb_days(first, second))
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
        # set awaiting_question under lock
        try:
            with _get_chat_lock(chat_id):
                state = _get_chat_state(chat_id)
                state.update({"awaiting_question": True})
                CHAT_STATE[chat_id] = state
        except Exception:
            logger.exception("cb_query: failed to set awaiting_question for chat=%s", chat_id)
        try:
            def try_edit_ask():
                context.bot.edit_message_caption(chat_id=chat_id, message_id=message_id,
                                                 caption=user_interface["captions"]["ask_question"], reply_markup=_make_rows([[ (user_interface["buttons"]["cancel_label"], f"cancel_ask") ]]))
            _edit_with_retries(try_edit_ask, context.bot, chat_id, message_id)
        except Exception as e:
            try:
                context.bot.send_message(chat_id=chat_id, text=user_interface["captions"]["ask_question"], reply_markup=_make_rows([[ (user_interface["buttons"]["cancel_label"], f"cancel_ask") ]]))
            except Exception as e:
                logger.exception("cb_query: failed to send ask_question fallback to chat=%s", chat_id)
        return

    if cmd == "cancel_ask":
        try:
            with _get_chat_lock(chat_id):
                state = _get_chat_state(chat_id)
                state["awaiting_question"] = False
                CHAT_STATE[chat_id] = state
        except Exception:
            logger.exception("cb_query: failed to clear awaiting_question for chat=%s", chat_id)

        advice_text = state.get("advice_text")
        first = state.get("first")
        second = state.get("second")
        days = int(state.get("days", 1)) if state.get("days") is not None else 1

        kb = None
        if first and second:
            ask_label = user_interface["buttons"]["ask_label_first"] if state.get("asked_count", 0) == 0 else user_interface["buttons"]["ask_label_more"]
            kb = _make_rows([[ (user_interface["buttons"]["back_label"], "back:first"), (ask_label, f"ask:{first}:{second}:{days}") ]])
        try:
            def try_edit_back():
                context.bot.edit_message_caption(chat_id=chat_id, message_id=message_id,
                                                 caption=advice_text, reply_markup=kb)
            _edit_with_retries(try_edit_back, context.bot, chat_id, message_id)
        except Exception as e:
            try:
                context.bot.send_message(chat_id=chat_id, text=advice_text, reply_markup=kb)
            except Exception as e:
                logger.exception("cb_query: failed to send advice_text fallback to chat=%s", chat_id)

    if cmd == "back":
        if len(parts) >= 2 and parts[1] == "first":
            # clear cached forecast data on back
            try:
                with _get_chat_lock(chat_id):
                    state = _get_chat_state(chat_id)
                    state.pop("cached_all_rates", None)
                    state.pop("cached_pair_key", None)
                    state.pop("forecasted_prices", None)
                    state.pop("forecasted_diffs", None)
                    state.pop("forecast_delta", None)
                    state.pop("advice_text", None)
                    state.pop("forecast_ts", None)
                    state["awaiting_question"] = False
                    CHAT_STATE[chat_id] = state
            except Exception:
                logger.exception("cb_query.back:first: failed to clear cache for chat=%s", chat_id)
            _replace_with_logo(context.bot, chat_id, message_id, caption=user_interface["captions"]["choose_first"], reply_markup=_kb_first())
            return

        if len(parts) >= 3 and parts[1] == "second":
            first = parts[2]
            try:
                with _get_chat_lock(chat_id):
                    state = _get_chat_state(chat_id)
                    state.pop("cached_all_rates", None)
                    state.pop("cached_pair_key", None)
                    state.pop("forecasted_prices", None)
                    state.pop("forecasted_diffs", None)
                    state.pop("forecast_delta", None)
                    state.pop("advice_text", None)
                    state.pop("forecast_ts", None)
                    state["awaiting_question"] = False
                    CHAT_STATE[chat_id] = state
            except Exception:
                logger.exception("cb_query.back:second: failed to clear cache for chat=%s", chat_id)
            _replace_with_logo(context.bot, chat_id, message_id, caption=f"Первая валюта: {first}\n{user_interface["captions"]["choose_second"]}", reply_markup=_kb_second(first))
            return

    query.answer(text="Неизвестная команда", show_alert=False)


def question_message_handler(update, context):
    logger.debug("question_message_handler called user=%s chat=%s", getattr(update, 'effective_user', None), getattr(update.message, 'chat', None) and update.message.chat.id)
    msg = update.message
    chat_id = msg.chat.id
    user_msg_id = msg.message_id
    text = (msg.text or "").strip()
    if not text:
        return

    # mark awaiting_question -> False under lock (thread-safe)
    with _get_chat_lock(chat_id):
        state = _get_chat_state(chat_id)
        if not state.get("awaiting_question"):
            return
        state["awaiting_question"] = False
        CHAT_STATE[chat_id] = state

    try:
        context.bot.delete_message(chat_id=chat_id, message_id=user_msg_id)
    except Exception as e:
        logger.exception("question_message_handler: failed to delete user message %s in chat=%s", user_msg_id, chat_id)

    bot_msg_id = state.get("msg_id")
    def try_edit_awaiting():
        context.bot.edit_message_caption(chat_id=chat_id, message_id=bot_msg_id, caption=user_interface["captions"]["awaiting_assistant"], reply_markup=None)
    _edit_with_retries(try_edit_awaiting, context.bot, chat_id, bot_msg_id)

    try:
        first = state.get("first")
        second = state.get("second")
        days = int(state.get("days", 1))
        try:
            first = state.get("first")
            second = state.get("second")
            days = int(state.get("days", 1))

            # try to use cached values first
            cached_all = state.get("cached_all_rates")
            cached_key = state.get("cached_pair_key")
            ts = state.get("forecast_ts")
            now_ts = int(time.time())

            pair_key = f"{second}_per_{first}"
            use_cached = False
            if cached_all and cached_key and ts:
                if cached_key == pair_key and (now_ts - int(ts)) <= CACHE_TTL:
                    use_cached = True
                else:
                    # cache expired or pair mismatch
                    logger.debug("question_message_handler: cache invalid or expired for chat=%s (key=%s ts=%s now=%s) expected=%s", chat_id, cached_key, ts, now_ts, pair_key)

            if use_cached:
                all_rates = cached_all
            else:
                all_rates = fetch_sequences_all_pairs(CURRENCIES, days=max(MODELS_SETTINGS["reg"]["max_n"], MODELS_SETTINGS["markov"]["max_n"]))

            dates = sorted(all_rates.keys())
            prices = [float(all_rates[d][pair_key]) for d in dates]
        except Exception as e:
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
                f"Advice (bot): {advice_text}\n"
            )
        else:
            summary_text = f"{history_text}Pair: {first}/{second}\nLatest prices: {last_prices_text}\nForecast days: {days}\n"

    except Exception as e:
        summary_text = "Chart data: unavailable.\n"

    gemini_resp = call_gemini_advice(text, summary_text)
    logger.debug("question_message_handler got gemini_resp_len=%s", len(gemini_resp or ""))

    final_caption = None
    if gemini_resp:
        final_caption = gemini_resp
    else:
        final_caption = "Ассистент не смог предоставить ответ (ошибка подключения к модели)."

    # update qa history and counters under lock
    try:
        with _get_chat_lock(chat_id):
            state = _get_chat_state(chat_id)
            qa_history = state.get("qa_history", [])
            qa_history.append({"q": text, "a": final_caption})
            qa_history = qa_history[-5:]
            state["qa_history"] = qa_history
            state["asked_count"] = state.get("asked_count", 0) + 1
            CHAT_STATE[chat_id] = state
    except Exception:
        logger.exception("question_message_handler: failed to update QA history for chat=%s", chat_id)

    first = state.get("first")
    second = state.get("second")
    kb = None
    if first and second:
        ask_label = user_interface["buttons"]["ask_label_first"] if state.get("asked_count", 0) == 0 else user_interface["buttons"]["ask_label_more"]
        kb = _make_rows([[ (user_interface["buttons"]["back_label"], "back:first"), (ask_label, f"ask:{first}:{second}:{state.get('days',1)}") ]])
    try:
        bot_msg_id = state.get("msg_id")
        def try_edit_final():
            context.bot.edit_message_caption(chat_id=chat_id, message_id=bot_msg_id, caption=final_caption, reply_markup=kb)
        _edit_with_retries(try_edit_final, context.bot, chat_id, bot_msg_id)
    except Exception as e:
        try:
            context.bot.send_message(chat_id=chat_id, text=final_caption, reply_markup=kb)
        except Exception as e:
            logger.exception("question_message_handler: failed to send message to chat=%s", chat_id)


def telegram_main():
    logger.debug("telegram_main starting")
    req = Request(connect_timeout=30, read_timeout=30)
    bot = Bot(token=TELEGRAM_TOKEN, request=req)
    updater = Updater(bot=bot, use_context=True)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler("start", start_handler))
    dp.add_handler(CallbackQueryHandler(cb_query))
    dp.add_handler(MessageHandler(Filters.text & (~Filters.command), question_message_handler))
    updater.start_polling()
    updater.idle()
    logger.debug("telegram_main stopped")


# ==============================
# MAIN
# ==============================
def train_models_if_needed(markov_min_n, markov_max_n, reg_min_n, reg_max_n):
    logger.debug("train_models_if_needed called markov_min=%s markov_max=%s reg_min=%s reg_max=%s", markov_min_n, markov_max_n, reg_min_n, reg_max_n)
    os.makedirs(MODELS_PATH, exist_ok=True)
    pairs = []
    for a in CURRENCIES:
        for b in CURRENCIES:
            if a == b:
                continue
            pairs.append((a, b))

    any_models = any(fname.startswith("regression") or fname.startswith("markov") for fname in os.listdir(MODELS_PATH))
    if any_models and not MODELS_SETTINGS["REBUILD"]:
        return

    for a, b in pairs:
        csv_path = os.path.join(DATASETS_PATH, f"{a}{b}.csv")
        signs = read_column("Sign", csv_path)
        diffs_s = read_column("Difference", csv_path)
        diffs = [float(x) for x in diffs_s]

        for n_lags in range(reg_min_n, reg_max_n+1):
            if len(diffs) > n_lags:
                coeffs = build_regression(diffs, n_lags=n_lags)
                save_model(coeffs, os.path.join(MODELS_PATH, f"regression_{a}{b}_{n_lags}.pkl"))

        for order in range(markov_min_n, markov_max_n+1):
            if len(signs) > order:
                m = build_markov_model(signs, order=order)
                save_model(m, os.path.join(MODELS_PATH, f"markov_{a}{b}_{order}.pkl"))


if __name__ == "__main__":
    train_models_if_needed(
        markov_min_n= MODELS_SETTINGS["markov"]["min_n"],
        markov_max_n= MODELS_SETTINGS["markov"]["max_n"],
        reg_min_n= MODELS_SETTINGS["reg"]["min_n"],
        reg_max_n= MODELS_SETTINGS["reg"]["max_n"],
    )
    telegram_main()
