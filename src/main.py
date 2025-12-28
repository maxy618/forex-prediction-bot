import os
import csv
from dotenv import load_dotenv
load_dotenv()

from logging_util import setup_logging
logger = setup_logging(name=__name__)

from model_engine import cache_data
from telegram_bot import telegram_main


CURRENCIES = ["USD", "EUR", "JPY", "GBP", "CHF", "RUB"]


MODELS_SETTINGS = {
    "REBUILD": True,
    "knn": {
        "k": 30,
        "window_size": 21, 
    }
}


CACHE_TTL = int(os.getenv("CACHE_TTL", 300))


user_interface = {
    "captions": {
        "warning": "Прогноз — это лишь вероятность, а не гарантия. Используй результат как подсказку, а не как финансовый совет",
        "choose_first": "Выберите первую валюту",
        "choose_second": "Выберите вторую валюты",
        "choose_days": "Выберите количество дней (1-9)",
        "predicting": "Выполняется прогноз, секунду...",
        "ask_question": "Отправьте вопрос по графику",
        "awaiting_assistant": "Ожидайте ответ ассистента...",
        "unexpected_error": "❌ Произошла непредвиденная ошибка, попробуйте снова",
        "send_logo_caption": "Нажми чтобы сделать прогноз",
        "confirm_selection": "Валютная пара: {first}/{second}\nКоличество дней: {days}"
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
        "confirm_label": "Все верно"
    }
}


TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
DATASETS_PATH = "../datasets/"
MODELS_PATH = "../models/"
TEMP_FOLDER = "../temp/"
LOGO_PATH = "../assets/logo.png"
ASK_IMG_PATH = "../assets/ask_question.png"
AI_THINKING_PATH = "../assets/ai_thinking.png"
PREDICTING_PATH = "../assets/predicting.png"


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-2.5-flash"


PROMPT_TEMPLATE = (
"""
Ты — casual ассистент для телеграм бота прогнозов валютных пар.
Ниже представлена история диалога (Conversation History) и текущие данные (Current Context).
Используй историю, чтобы поддерживать нить разговора.

Основные правила:
* Отвечай прямо, без лишних оговорок.
* Технические детали раскрывай только по запросу.
* Ответ: 4–6 предложений максимум.
* Тон: дружелюбный, деловой, уверенный.

Conversation History:
{history}

Current Context:
{context}

Вопрос пользователя:
{question}

Ответ:
"""
)


def read_column(column_name, csv_path):
    values = []
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            values.append(row[column_name])
    return values


def train_models_if_needed():
    os.makedirs(MODELS_PATH, exist_ok=True)
    pairs = []
    for a in CURRENCIES:
        for b in CURRENCIES:
            if a != b:
                pairs.append((a, b))

    any_models = any(fname.endswith(".pkl") for fname in os.listdir(MODELS_PATH))
    if any_models and not MODELS_SETTINGS["REBUILD"]:
        return

    for a, b in pairs:
        csv_path = os.path.join(DATASETS_PATH, f"{a}{b}.csv")
        if not os.path.exists(csv_path):
            continue
            
        diffs_s = read_column("Difference", csv_path)
        diffs = [float(x) for x in diffs_s]
        cache_data(diffs, os.path.join(MODELS_PATH, f"knn_data_{a}{b}.pkl"))


if __name__ == "__main__":
    train_models_if_needed()
    config = {
        "TELEGRAM_TOKEN": TELEGRAM_TOKEN,
        "TEMP_FOLDER": TEMP_FOLDER,
        "LOGO_PATH": LOGO_PATH,
        "ASK_IMG_PATH": ASK_IMG_PATH,
        "AI_THINKING_PATH": AI_THINKING_PATH,
        "PREDICTING_PATH": PREDICTING_PATH,
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