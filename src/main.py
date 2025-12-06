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

from matplotlib import use
use("Agg")
import matplotlib.pyplot as plt

from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup, InputMediaPhoto
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
        "ask_label_more": "Задать ещё вопрос",
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

SESSION = requests.Session()
retry_strategy = Retry(
    total=3,
    backoff_factor=0.3,
    status_forcelist=(429, 500, 502, 503, 504),
    allowed_methods=("GET",)
)
SESSION.mount("https://", HTTPAdapter(max_retries=retry_strategy))
SESSION.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) cbr-client/1.0"})

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

PROMPT_TEMPLATE = (
    """
Кратко и прямо: ты — ассистент для телеграм бота прогнозов валютных пар.

Основные правила:
- НЕ упоминай, как строится прогноз (регрессии, марковские цепи и т.п.), если пользователь явно не спросил "Как построен прогноз?" или не потребовал технических деталей. В обычных ответах давай только результат, числа и краткое пояснение значения.
- Регрессия обучается на АБСОЛЮТНЫХ разностях (abs(diffs)). Знак приращения определяется марковским компонентом. Технические детали раскрывай только по прямому запросу.
- Поддерживай практичный, краткий и в то же время формальный, дружеский тон если это того требует.
- Твой ответ должен состоять не более чем из 4-6 предложений

Что можно показывать:
- Последние N цен, прогнозные цены по дням, абсолютные и процентные изменения; знак (рост/падение) и величина приращения; метрики согласованности (консенсус марковского ансамбля, std регрессии) — если доступны.
- Простые рекомендации: "стоит" / "не стоит" покупать с числовыми обоснованиями.

Запрещено: использовать стандартные оговорки типа "это не инвестиционная рекомендация" или похожие фразы; придумывать недоступные данные (если данных нет — скажи, что недоступны). 

Вход (summary_text) выглядит так:
Pair: EUR/USD
Latest prices: 100.123456, 100.234567, 100.345678
Forecast days: 3
Forecasted prices: 100.400000, 100.450000, 100.500000
Forecast delta (last vs current): 0.154322
Advice (bot): стоит покупать — прогноз роста

Теперь: прочитай summary_text и вопрос пользователя и ответь.
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

    return max(next_states, key=next_states.get)


def predict_ensemble_sign(sequence, models):
    votes = {"+": 0, "-": 0}
    for model in models:
        next_state = predict_state(sequence, model)
        if next_state is None:
            continue
        votes[next_state] += 1
    
    max_votes = max(votes.values())
    leaders = [k for k, v in votes.items() if v == max_votes]
    return random.choice(leaders)


def forecast_signs(sequence, models, n=1):
    result = []
    curr_seq = sequence.copy()
    
    for _ in range(n):
        next_state = predict_ensemble_sign(curr_seq, models)
        result.append(next_state)
        curr_seq.append(next_state)
    
    return result


# ==============================
# REGRESSION
# ==============================
def build_regression(diffs_list, n_lags=1):
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

    return coeffs.flatten()


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
    
    return sum(results) / len(results)


def forecast_diffs(last_diffs, models, n=1):
    result = []
    diffs = last_diffs.copy()

    for _ in range(n):
        next_diff = predict_ensemble_diff(diffs, models)
        result.append(next_diff)
        diffs.append(next_diff)

    return result


# ==============================
# PARSER
# ==============================
def get_dates_list(num_days):
    today = date.today()
    start = today - timedelta(days=num_days-1)
    
    day = start
    res = []
    while day <= today:
        res.append(day)
        day += timedelta(days=1)
    return res


def fetch_for_date(d, base_latest, base_archive):
    if d == date.today():
        url = base_latest
    else:
        url = f"{base_archive}/{d.year}/{d.month:02d}/{d.day:02d}/daily_json.js"
    try:
        r = SESSION.get(url, timeout=6)
        if r.status_code == 200:
            try:
                return r.json()
            except ValueError:
                pass
    except Exception:
        pass
    try:
        time.sleep(0.2)
        r = SESSION.get(url, timeout=6)
        if r.status_code == 200:
            try:
                return r.json()
            except ValueError:
                pass
    except Exception:
        pass
    if url != base_latest:
        try:
            r = SESSION.get(base_latest, timeout=6)
            if r.status_code == 200:
                try:
                    return r.json()
                except ValueError:
                    pass
        except Exception:
            pass
    return None


def rub_per_one_from_json(js, code):
    if code == "RUB":
        return 1.0
    if not js or "Valute" not in js:
        return None
    info = js["Valute"].get(code)
    if not info:
        return None
    nominal = int(info.get("Nominal", 1))
    value = float(info.get("Value"))
    return value / nominal


def fetch_sequences_all_pairs(currencies, days=max(MODELS_SETTINGS["markov"]["max_n"], MODELS_SETTINGS["reg"]["max_n"])):
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

    return res


# ==============================
# PLOTTER
# ==============================
def make_axes_limits(prices):
    min_price = min(prices)
    max_price = max(prices)
    diff = max_price - min_price

    if diff == 0:
        diff = min_price * 0.01 if min_price != 0 else 1

    low = min_price - diff * 0.5
    high = max_price + diff * 0.5

    return low, high


def plot_sequence(old_prices, new_prices, filename):
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

    return filename


# ==============================
# TELEGRAM
CHAT_STATE = {}

def _temp_file(user_id):
    os.makedirs(TEMP_FOLDER, exist_ok=True)
    return os.path.join(TEMP_FOLDER, f"{user_id}_{int(time.time()*1000)}.png")

def _kb_first():
    # Use labels from user_interface to build first-currency buttons
    codes = user_interface["buttons"].get("currency_codes", CURRENCIES)
    return _make_rows([[(c, f"first:{c}") for c in codes]])

def _kb_second(first):
    codes = user_interface["buttons"].get("currency_codes", CURRENCIES)
    buttons = [(c, f"second:{first}:{c}") for c in codes if c != first]
    buttons.append((user_interface["buttons"]["back_label"], "back:first"))
    return _make_rows([buttons])

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
    return CHAT_STATE.get(chat_id, {"msg_id": None, "has_logo": False, "awaiting_question": False})

def _send_start_message(bot: Bot, chat_id: int):
    if os.path.exists(LOGO_PATH):
        try:
            with open(LOGO_PATH, "rb") as f:
                msg = bot.send_photo(
                    chat_id=chat_id,
                    photo=f,
                    caption=user_interface["captions"]["warning"],
                    reply_markup=InlineKeyboardMarkup(
                        [[InlineKeyboardButton(user_interface["buttons"]["warning_label"], callback_data="do_predict")]]
                    )
                )
            _save_chat_state(chat_id, msg.message_id, True)
            return msg
        except Exception:
            pass
    msg = bot.send_message(
        chat_id=chat_id,
        text=user_interface["captions"]["warning"],
        reply_markup=InlineKeyboardMarkup(
            [[InlineKeyboardButton(user_interface["captions"]["warning"], callback_data="do_predict")]]
        )
    )
    _save_chat_state(chat_id, msg.message_id, False)
    return msg

def _edit_with_retries(action_callable, bot: Bot, chat_id: int, message_id: int,
                       max_attempts: int = 3, delay: float = 1.0):
    last_exc = None
    for attempt in range(1, max_attempts + 1):
        try:
            action_callable()
            return True
        except Exception as e:
            last_exc = e
            if attempt < max_attempts:
                time.sleep(delay)
                continue
    try:
        bot.delete_message(chat_id=chat_id, message_id=message_id)
    except Exception:
        pass
    try:
        bot.send_message(chat_id=chat_id, text=user_interface["captions"]["unexpected_error"])
    except Exception:
        pass
    try:
        _send_start_message(bot, chat_id)
    except Exception:
        pass
    return False

def start_handler(update, context):
    chat_id = update.effective_chat.id
    kb = InlineKeyboardMarkup([[InlineKeyboardButton(user_interface["buttons"]["warning_label"], callback_data="do_predict")]])
    if os.path.exists(LOGO_PATH):
        try:
            with open(LOGO_PATH, "rb") as f:
                msg = context.bot.send_photo(chat_id=chat_id, photo=f, caption=user_interface["captions"]["warning"], reply_markup=kb)
            _save_chat_state(chat_id, msg.message_id, True)
            return
        except Exception:
            pass
    msg = context.bot.send_message(chat_id=chat_id, text=user_interface["captions"]["warning"], reply_markup=kb)
    _save_chat_state(chat_id, msg.message_id, False)

def _replace_with_logo(bot, chat_id, message_id, caption=None, reply_markup=None, max_attempts=3, delay=1.0):
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
            except Exception:
                pass
            _save_chat_state(chat_id, new_msg.message_id, False)
            return
        except Exception:
            try:
                new_msg = bot.send_message(chat_id=chat_id, text=caption or "", reply_markup=reply_markup)
                try:
                    bot.delete_message(chat_id=chat_id, message_id=message_id)
                except Exception:
                    pass
                _save_chat_state(chat_id, new_msg.message_id, False)
                return
            except Exception:
                try:
                    bot.delete_message(chat_id=chat_id, message_id=message_id)
                except Exception:
                    pass
                try:
                    bot.send_message(chat_id=chat_id, text=user_interface["captions"]["unexpected_error"])
                except Exception:
                    pass
                try:
                    _send_start_message(bot, chat_id)
                except Exception:
                    pass
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
    except Exception:
        return None

    try:
        j = r.json()
    except Exception:
        return None

    try:
        return j["candidates"][0]["content"]["parts"][0]["text"].strip()
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
        return txt.strip() if txt else None

def _perform_prediction_and_edit(bot, chat_id, message_id, user_id, first, second, days):
    try:
        needed_days = max(MODELS_SETTINGS["reg"]["max_n"], MODELS_SETTINGS["markov"]["max_n"])
    except Exception:
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
        "forecast_ts": int(time.time())
    })
    CHAT_STATE[chat_id] = state

    try:
        os.remove(out_path)
    except Exception:
        pass

def cb_query(update, context):
    query = update.callback_query
    data = (query.data or "").strip()
    user_id = update.effective_user.id
    chat_id = query.message.chat.id
    message_id = query.message.message_id

    query.answer()

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
        state = _get_chat_state(chat_id)
        state.update({"awaiting_question": True})
        CHAT_STATE[chat_id] = state
        try:
            def try_edit_ask():
                context.bot.edit_message_caption(chat_id=chat_id, message_id=message_id,
                                                 caption=user_interface["captions"]["ask_question"], reply_markup=_make_rows([[ (user_interface["buttons"]["cancel_label"], f"cancel_ask") ]]))
            _edit_with_retries(try_edit_ask, context.bot, chat_id, message_id)
        except Exception:
            try:
                context.bot.send_message(chat_id=chat_id, text=user_interface["captions"]["ask_question"], reply_markup=_make_rows([[ (user_interface["buttons"]["cancel_label"], f"cancel_ask") ]]))
            except Exception:
                pass
        return

    if cmd == "cancel_ask":
        state = _get_chat_state(chat_id)
        state["awaiting_question"] = False
        CHAT_STATE[chat_id] = state

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
        except Exception:
            try:
                context.bot.send_message(chat_id=chat_id, text=advice_text, reply_markup=kb)
            except Exception:
                pass

    if cmd == "back":
        if len(parts) >= 2 and parts[1] == "first":
            _replace_with_logo(context.bot, chat_id, message_id, caption=user_interface["captions"]["choose_first"], reply_markup=_kb_first())
            return

        if len(parts) >= 3 and parts[1] == "second":
            first = parts[2]
            _replace_with_logo(context.bot, chat_id, message_id, caption=f"Первая валюта: {first}\n{user_interface["captions"]["choose_second"]}", reply_markup=_kb_second(first))
            return

    query.answer(text="Неизвестная команда", show_alert=False)

def question_message_handler(update, context):
    msg = update.message
    chat_id = msg.chat.id
    user_msg_id = msg.message_id
    text = (msg.text or "").strip()
    if not text:
        return

    state = _get_chat_state(chat_id)
    if not state.get("awaiting_question"):
        return

    state["awaiting_question"] = False
    CHAT_STATE[chat_id] = state

    try:
        context.bot.delete_message(chat_id=chat_id, message_id=user_msg_id)
    except Exception:
        pass

    bot_msg_id = state.get("msg_id")
    def try_edit_awaiting():
        context.bot.edit_message_caption(chat_id=chat_id, message_id=bot_msg_id, caption=user_interface["captions"]["awaiting_assistant"], reply_markup=None)
    _edit_with_retries(try_edit_awaiting, context.bot, chat_id, bot_msg_id)

    try:
        first = state.get("first")
        second = state.get("second")
        days = int(state.get("days", 1))
        try:
            all_rates = fetch_sequences_all_pairs(CURRENCIES, days=max(MODELS_SETTINGS["reg"]["max_n"], MODELS_SETTINGS["markov"]["max_n"]))
            pair_key = f"{second}_per_{first}"
            dates = sorted(all_rates.keys())
            prices = [float(all_rates[d][pair_key]) for d in dates]
        except Exception:
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

    except Exception:
        summary_text = "Chart data: unavailable.\n"

    gemini_resp = call_gemini_advice(text, summary_text)

    final_caption = None
    if gemini_resp:
        final_caption = gemini_resp
    else:
        final_caption = "Ассистент не смог предоставить ответ (ошибка подключения к модели)."

    qa_history = state.get("qa_history", [])
    qa_history.append({"q": text, "a": final_caption})
    qa_history = qa_history[-5:]
    state["qa_history"] = qa_history
    state["asked_count"] = state.get("asked_count", 0) + 1
    CHAT_STATE[chat_id] = state

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
    except Exception:
        try:
            context.bot.send_message(chat_id=chat_id, text=final_caption, reply_markup=kb)
        except Exception:
            pass

def telegram_main():
    req = Request(connect_timeout=30, read_timeout=30)
    bot = Bot(token=TELEGRAM_TOKEN, request=req)
    updater = Updater(bot=bot, use_context=True)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler("start", start_handler))
    dp.add_handler(CallbackQueryHandler(cb_query))
    dp.add_handler(MessageHandler(Filters.text & (~Filters.command), question_message_handler))
    updater.start_polling()
    updater.idle()


# ==============================
# MAIN
def train_models_if_needed(markov_min_n, markov_max_n, reg_min_n, reg_max_n):
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
