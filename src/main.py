import os
import random
import csv
import pickle
import time

from collections import defaultdict, Counter
from datetime import date, timedelta

import numpy as np
import requests
from requests.adapters import HTTPAdapter, Retry

from matplotlib import use
use("Agg")
import matplotlib.pyplot as plt

from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup, InputMediaPhoto
from telegram.ext import Updater, CommandHandler, CallbackQueryHandler
from telegram.utils.request import Request

from dotenv import load_dotenv
load_dotenv()


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

TELEGRAM_TOKEN = load_dotenv("TELEGRAM_TOKEN")

DATASETS_PATH = "../datasets/"
MODELS_PATH = "../models/"
TEMP_FOLDER = "../temp/"
LOGO_PATH = "../assets/logo.png"

CURRENCIES = ["EUR", "USD", "RUB"]

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
    
    # counters
    for idx in range(len(sequence) - order):
        current_state = tuple(sequence[idx:idx+order])
        next_state = sequence[idx+order]
        counters[current_state][next_state] += 1
    
    # converting to probabilities
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

    X_mat = np.array(x, dtype=float)                 # (N, n_lags)
    y_vec = np.array(y, dtype=float).reshape(-1, 1)  # (N, 1)

    ones = np.ones((X_mat.shape[0], 1))
    X_with_bias = np.hstack([X_mat, ones])

    Xt = X_with_bias.T
    XtX = Xt @ X_with_bias
    Xty = Xt @ y_vec

    coeffs = np.linalg.inv(XtX) @ Xty

    return coeffs.flatten()


def predict_diff(last_diffs, coeffs):
    n_lags = len(coeffs) - 1  # w/o bias
    
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
    last_known = {c: None for c in currencies}  # RUB per 1 <currency>

    res = {}
    for d in dates:
        js = fetch_for_date(d, BASE_LATEST, BASE_ARCHIVE)
        if js is None:
            # 2nd attemp
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
# TELEGRAM (упрощённый)
# ==============================
CHAT_STATE = {}  # chat_id -> {"msg_id": int, "has_logo": bool}


def _temp_file(user_id):
    os.makedirs(TEMP_FOLDER, exist_ok=True)
    return os.path.join(TEMP_FOLDER, f"{user_id}_{int(time.time()*1000)}.png")


def _kb_first():
    return _make_rows([[(c, f"first:{c}") for c in CURRENCIES]])


def _kb_second(first):
    buttons = [(c, f"second:{first}:{c}") for c in CURRENCIES if c != first]
    buttons.append(("Назад", "back:first"))
    return _make_rows([buttons])


def _kb_days(first, second):
    rows = []
    row = []
    for i in range(1, 10):
        row.append((str(i), f"days:{first}:{second}:{i}"))
        if len(row) == 3:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    rows.append([("Назад", f"back:second:{first}")])
    return _make_rows(rows)


def _make_rows(pairs):
    keyboard = []
    for row in pairs:
        keyboard.append([InlineKeyboardButton(text, callback_data=cb) for text, cb in row])
    return InlineKeyboardMarkup(keyboard)


def _save_chat_state(chat_id, message_id, has_logo):
    CHAT_STATE[chat_id] = {"msg_id": int(message_id), "has_logo": bool(has_logo)}


def _get_chat_state(chat_id):
    return CHAT_STATE.get(chat_id, {"msg_id": None, "has_logo": False})


def start_handler(update, context):
    chat_id = update.effective_chat.id
    kb = InlineKeyboardMarkup([[InlineKeyboardButton("Сделать прогноз", callback_data="do_predict")]])
    if os.path.exists(LOGO_PATH):
        with open(LOGO_PATH, "rb") as f:
            msg = context.bot.send_photo(chat_id=chat_id, photo=f, caption="Нажми чтобы сделать прогноз", reply_markup=kb)
        _save_chat_state(chat_id, msg.message_id, True)
    else:
        msg = context.bot.send_message(chat_id=chat_id, text="Нажми чтобы сделать прогноз", reply_markup=kb)
        _save_chat_state(chat_id, msg.message_id, False)


def _replace_with_logo(bot, chat_id, message_id, caption=None, reply_markup=None):
    state = _get_chat_state(chat_id)
    if state["has_logo"]:
        try:
            bot.edit_message_caption(chat_id=chat_id, message_id=message_id, caption=caption, reply_markup=reply_markup)
        except Exception:
            bot.send_message(chat_id=chat_id, text=caption or "", reply_markup=reply_markup)
        return

    if os.path.exists(LOGO_PATH):
        try:
            with open(LOGO_PATH, "rb") as f:
                media = InputMediaPhoto(f, caption=caption)
                bot.edit_message_media(media=media, chat_id=chat_id, message_id=message_id, reply_markup=reply_markup)
            _save_chat_state(chat_id, message_id, True)
            return
        except Exception:
            try:
                bot.edit_message_caption(chat_id=chat_id, message_id=message_id, caption=caption, reply_markup=reply_markup)
            except Exception:
                bot.send_message(chat_id=chat_id, text=caption or "", reply_markup=reply_markup)
            _save_chat_state(chat_id, message_id, False)
            return

    try:
        bot.edit_message_caption(chat_id=chat_id, message_id=message_id, caption=caption, reply_markup=reply_markup)
    except Exception:
        bot.send_message(chat_id=chat_id, text=caption or "", reply_markup=reply_markup)
    _save_chat_state(chat_id, message_id, False)


def _edit_or_send_media(bot, chat_id, message_id, caption=None, media_path=None, reply_markup=None):
    if media_path and os.path.exists(media_path):
        try:
            with open(media_path, "rb") as f:
                media = InputMediaPhoto(f, caption=caption)
                bot.edit_message_media(media=media, chat_id=chat_id, message_id=message_id, reply_markup=reply_markup)
            _save_chat_state(chat_id, message_id, False)
            return
        except Exception:
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
                new_msg = bot.send_message(chat_id=chat_id, text=caption or "", reply_markup=reply_markup)
                try:
                    bot.delete_message(chat_id=chat_id, message_id=message_id)
                except Exception:
                    pass
                _save_chat_state(chat_id, new_msg.message_id, False)
                return
    else:
        try:
            bot.edit_message_caption(chat_id=chat_id, message_id=message_id, caption=caption, reply_markup=reply_markup)
            state = _get_chat_state(chat_id)
            _save_chat_state(chat_id, message_id, state.get("has_logo", False))
            return
        except Exception:
            new_msg = bot.send_message(chat_id=chat_id, text=caption or "", reply_markup=reply_markup)
            try:
                bot.delete_message(chat_id=chat_id, message_id=message_id)
            except Exception:
                pass
            _save_chat_state(chat_id, new_msg.message_id, False)
            return


def _perform_prediction_and_edit(bot, chat_id, message_id, user_id, first, second, days):
    try:
        needed_days = max(MODELS_SETTINGS["reg"]["max_n"], MODELS_SETTINGS["markov"]["max_n"])
    except Exception:
        needed_days = days if days > 0 else 10

    try:
        all_rates = fetch_sequences_all_pairs(CURRENCIES, days=needed_days)
    except Exception as e:
        try:
            bot.edit_message_caption(chat_id=chat_id, message_id=message_id,
                                     caption=f"Ошибка при получении актуальных данных: {e}")
        except Exception:
            pass
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
        try:
            bot.edit_message_caption(chat_id=chat_id, message_id=message_id,
                                     caption=f"Ошибка при формировании последовательности цен: {e}")
        except Exception:
            pass
        return

    if not prices:
        try:
            bot.edit_message_caption(chat_id=chat_id, message_id=message_id, caption="Ошибка: недостаточно данных от парсера")
        except Exception:
            pass
        return

    diffs = [0.0]
    for i in range(1, len(prices)):
        diffs.append(prices[i] - prices[i-1])
    signs = ["+" if d >= 0 else "-" for d in diffs]

    old_prices = prices[-3:] if len(prices) >= 3 else prices[:]
    if not old_prices:
        try:
            bot.edit_message_caption(chat_id=chat_id, message_id=message_id, caption="Ошибка: недостаточно данных для построения графика")
        except Exception:
            pass
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

    _edit_or_send_media(bot, chat_id, message_id, caption=advice, media_path=out_path, reply_markup=_make_rows([[("Назад", "back:first")]]))

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
        context.bot.edit_message_caption(chat_id=chat_id, message_id=message_id, caption="Выберите первую валюту", reply_markup=_kb_first())
        return

    if cmd == "first" and len(parts) == 2:
        first = parts[1]
        context.bot.edit_message_caption(chat_id=chat_id, message_id=message_id, caption="Выберите вторую валюту", reply_markup=_kb_second(first))
        return

    if cmd == "second" and len(parts) == 3:
        _, first, second = parts
        context.bot.edit_message_caption(chat_id=chat_id, message_id=message_id, caption="Выберите количество дней (1-9)", reply_markup=_kb_days(first, second))
        return

    if cmd == "days" and len(parts) == 4:
        _, first, second, days_str = parts
        try:
            days = int(days_str)
        except ValueError:
            context.bot.edit_message_caption(chat_id=chat_id, message_id=message_id, caption="Неверный выбор дней")
            return
        context.bot.edit_message_caption(chat_id=chat_id, message_id=message_id, caption="Выполняется прогноз, секунду...")
        _perform_prediction_and_edit(context.bot, chat_id, message_id, user_id, first, second, days)
        return

    if cmd == "back":
        if len(parts) >= 2 and parts[1] == "first":
            _replace_with_logo(context.bot, chat_id, message_id, caption="Выберите первую валюту", reply_markup=_kb_first())
            return

        if len(parts) >= 3 and parts[1] == "second":
            first = parts[2]
            _replace_with_logo(context.bot, chat_id, message_id, caption="Выберите вторую валюту", reply_markup=_kb_second(first))
            return

    query.answer(text="Неизвестная команда", show_alert=False)


def telegram_main():
    req = Request(connect_timeout=30, read_timeout=30)
    bot = Bot(token=TELEGRAM_TOKEN, request=req)
    updater = Updater(bot=bot, use_context=True)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler("start", start_handler))
    dp.add_handler(CallbackQueryHandler(cb_query))
    updater.start_polling()
    updater.idle()


# ==============================
# MAIN
# ==============================
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

        # regression models
        for n_lags in range(reg_min_n, reg_max_n+1):
            if len(diffs) > n_lags:
                coeffs = build_regression(diffs, n_lags=n_lags)
                save_model(coeffs, os.path.join(MODELS_PATH, f"regression_{a}{b}_{n_lags}.pkl"))

        # markov models 
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
