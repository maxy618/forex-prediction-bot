import os
import csv

from dotenv import load_dotenv
load_dotenv()

from logging_util import setup_logging
logger = setup_logging(name=__name__)

from model_engine import (
    save_model,
    build_markov_model,
    build_regression,
)

from telegram_bot import telegram_main


CURRENCIES = ["USD", "EUR", "JPY", "GBP", "CHF", "RUB"]

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
        "choose_first": "Выберите первую валюту",
        "choose_second": "Выберите вторую валюту",
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
        "warning_label": "Я понял(-а)",
        "gif": "Показать анимацию",
        "png": "Показать изображение",
    }
}

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

DATASETS_PATH = "../datasets/"
MODELS_PATH = "../models/"
TEMP_FOLDER = "../temp/"
LOGO_PATH = "../assets/logo.png"

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

Читай summary_text, слушай вопрос пользователя и отвечай как нормальный человек.
"""
)


def read_column(column_name, csv_path):
    logger.debug("read_column called column=%s csv_path=%s", column_name, csv_path)
    values = []
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            values.append(row[column_name])
    return values


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
    config = {
        "TELEGRAM_TOKEN": TELEGRAM_TOKEN,
        "TEMP_FOLDER": TEMP_FOLDER,
        "LOGO_PATH": LOGO_PATH,
        "MODELS_PATH": MODELS_PATH,
        "MODELS_SETTINGS": MODELS_SETTINGS,
        "CURRENCIES": CURRENCIES,
        "user_interface": user_interface,
        "CACHE_TTL": CACHE_TTL,
        "GEMINI_API_KEY": GEMINI_API_KEY,
        "GEMINI_MODEL": GEMINI_MODEL,
        "PROMPT_TEMPLATE": PROMPT_TEMPLATE,
    }
    telegram_main(config)
