# 📈 Forex Prediction Telegram Bot

![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-active-success)
![AI Powered](https://img.shields.io/badge/AI-Gemini%20Flash-blue)

A sophisticated Telegram bot designed to forecast currency exchange rates using a **hybrid machine learning ensemble**. By combining **Linear Regression** for quantitative magnitude prediction and **Markov Chains** for directional trend analysis, this bot provides daily forecasts for major currency pairs (USD, EUR, GBP, etc.) against the Russian Ruble (RUB).

It features real-time data fetching from the Central Bank of Russia (CBR), generates high-quality visualizations (static charts and animated GIFs), and integrates **Google's Gemini AI** to provide natural language financial insights based on the technical data.

---

## 📑 Table of Contents
- [✨ Key Features](#-key-features)
- [🧠 Technical Architecture](#-technical-architecture)
  - [The Hybrid Model](#the-hybrid-model)
  - [Data Flow](#data-flow)
- [📂 Project Structure](#-project-structure)
- [🚀 Installation & Setup](#-installation--setup)
- [⚙️ Configuration](#%EF%B8%8F-configuration)
- [🖥️ Usage](#%EF%B8%8F-usage)
- [🛠️ Modules Breakdown](#%EF%B8%8F-modules-breakdown)
- [⚠️ Disclaimer](#%EF%B8%8F-disclaimer)
- [📄 License](#-license)

---

## ✨ Key Features

*   **Hybrid ML Engine**: unique combination of deterministic (Linear Regression) and probabilistic (Markov Chains) models to improve forecast accuracy.
*   **Smart Trend Analysis**: Uses a configurable **Temperature** parameter to control the stochastic nature of the Markov Chain predictions (balancing between conservative and volatile trends).
*   **Rich Visualizations**:
    *   **Static Plots**: High-resolution PNGs showing historical data vs. predicted trends.
    *   **Animated GIFs**: Dynamic visualizations showing the transition from history to forecast.
*   **AI Financial Analyst**: Integrated **Google Gemini API** allows users to ask questions about the forecast (e.g., *"Why is the trend going down?"*) and receive AI-generated explanations.
*   **Automated Data Pipeline**:
    *   Fetches daily official rates from the CBR API.
    *   Automatically updates historical datasets (`.csv`).
    *   Retrains models on-the-fly if data is updated.
*   **Robust Logging**: Daily rotating logs to monitor bot health and user interactions.

---

## 🧠 Technical Architecture

### The Hybrid Model
The core of this project relies on an ensemble approach to solve the two main problems of time-series forecasting: **Magnitude** (How much will it change?) and **Direction** (Will it go up or down?).

1.  **Linear Regression (The Magnitude):**
    *   Uses the Least Squares method to fit a linear trend to the recent historical data.
    *   Responsible for predicting the *numerical difference* (delta) between days.
    *   *Library:* `numpy` (Matrix operations).

2.  **Markov Chains (The Direction):**
    *   Constructs a transition matrix based on historical "signs" (Did the price go up `+` or down `-`?).
    *   Calculates the probability of the next day's sign based on the previous sequence (Order $k$).
    *   **Temperature Scaling**: A hyperparameter $T$ is applied to the probability distribution.
        *   $T < 1$: Makes the model "conservative" (favors the most likely outcome).
        *   $T = 1$: Standard probability.
        *   $T > 1$: Increases randomness (allows for "black swan" events).

### Data Flow
1.  **User Request**: User selects a currency pair (e.g., USD/RUB) via Telegram.
2.  **Data Fetching**: `parser.py` requests the latest JSON data from CBR.
3.  **Preprocessing**: Data is cleaned, and missing days are interpolated.
4.  **Inference**:
    *   `model_engine.py` loads the trained `.pkl` models.
    *   Generates a prediction for $N$ days ahead.
5.  **Visualization**: `plotter.py` draws the graph and saves it to `temp/`.
6.  **Response**: The bot sends the image/GIF to the user.
7.  **AI Context**: If the user asks a question, the numerical data is sent to Gemini to generate a text explanation.

---

## 📂 Project Structure

```bash
forex-prediction-bot/
├── assets/              # Static assets (logos, loading icons)
├── datasets/            # Historical data (CSV files for training)
│   ├── USD.csv
│   └── EUR.csv
├── logs/                # Rotating log files
├── models/              # Serialized trained models (.pkl)
├── src/                 # Source code
│   ├── main.py          # Entry point for the application
│   ├── parser.py        # CBR API interaction and data cleaning
│   ├── model_engine.py  # Linear Regression & Markov Chain logic
│   ├── plotter.py       # Matplotlib visualization logic
│   └── telegram_bot.py  # Telegram API handlers and UI
├── utils/               # Utility scripts
│   └── logging_util.py  # Logger configuration
├── .env                 # Environment variables (API Keys)
├── .gitignore
├── requirements.txt     # Python dependencies
└── README.md            # Documentation
```

---

## 🚀 Installation & Setup

### Prerequisites
*   Python 3.9 or higher.
*   A Telegram Bot Token (from [@BotFather](https://t.me/BotFather)).
*   A Google Gemini API Key (from [Google AI Studio](https://aistudio.google.com/)).

### Step-by-Step Guide

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/maxy618/forex-prediction-bot.git
    cd forex-prediction-bot
    ```

2.  **Create a Virtual Environment (Recommended)**
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```
    *If `requirements.txt` is missing, install manually:*
    ```bash
    pip install requests matplotlib Pillow python-telegram-bot python-dotenv numpy google-generativeai
    ```

4.  **Set Up Environment Variables**
    Create a `.env` file in the root directory and populate it:
    ```ini
    # .env file
    TELEGRAM_TOKEN=123456789:ABCdefGHIjklMNOpqrsTUVwxyz
    GEMINI_API_KEY=AIzaSyD...
    CACHE_TTL=300
    DEBUG_MODE=False
    ```

5.  **Initialize Directories**
    Ensure the following folders exist (the script usually creates them, but good to be safe):
    ```bash
    mkdir -p datasets models logs temp
    ```

---

## ⚙️ Configuration

You can fine-tune the bot's behavior by modifying `src/main.py` or `src/config.py` (if available).

| Variable | Location | Description | Default |
| :--- | :--- | :--- | :--- |
| `CURRENCIES` | `main.py` | List of supported currency codes (USD, EUR, etc.) | `['USD', 'EUR', 'CNY', ...]` |
| `TEMPERATURE` | `model_engine.py` | Controls prediction randomness. | `0.3` |
| `LOOKBACK_DAYS` | `model_engine.py` | How many past days the Linear Reg considers. | `30` |
| `MARKOV_ORDER` | `model_engine.py` | The depth of the Markov Chain memory. | `5` |

---

## 🖥️ Usage

1.  **Start the Bot**
    ```bash
    python src/main.py
    ```
    *Note: On the first run, the bot may take a few seconds to train the initial models using the CSV data in `datasets/`.*

2.  **Telegram Interaction**
    *   Open your bot in Telegram.
    *   Send `/start`.
    *   **Select Currency**: Choose the base currency (e.g., USD).
    *   **Select Horizon**: Choose how many days to predict (1-7 days).
    *   **Analyze**: The bot will reply with a chart.
    *   **Ask AI**: Click the "Ask AI" button to get a text breakdown of the chart.

---

## 🛠️ Modules Breakdown

*   **`src/parser.py`**:
    *   Connects to `cbr.ru` API.
    *   Parses XML/JSON responses.
    *   Calculates cross-rates if direct pairs aren't available.

*   **`src/model_engine.py`**:
    *   Contains the `Predictor` class.
    *   `train()`: Fits the Linear Regression and builds Markov transition matrices.
    *   `predict()`: Runs the simulation step-by-step for $N$ days.

*   **`src/plotter.py`**:
    *   Uses `matplotlib.pyplot` to draw the "History" (White) and "Forecast" (Cyan/Magenta).
    *   Handles the creation of frames for GIF generation using `PIL`.

*   **`src/telegram_bot.py`**:
    *   Uses `python-telegram-bot` (async/await).
    *   Manages `ConversationHandler` states (CHOOSING_CURRENCY, CHOOSING_DAYS).
    *   Handles callback queries from inline buttons.

---

## ⚠️ Disclaimer

**This tool is for educational and research purposes only.**

The predictions generated by this bot are based on mathematical models (Linear Regression and Markov Chains) and historical data. Financial markets are influenced by unpredictable real-world events (news, geopolitics) that these models cannot foresee.
*   **Do not use this bot as financial advice.**
*   **Do not trade real money based solely on these predictions.**
*   The developers assume no responsibility for financial losses.

---

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

*Created by [maxy618](https://github.com/maxy618)*