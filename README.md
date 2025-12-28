# 📈 Forex Prediction Telegram Bot (KNN & AI-Powered)

![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-active-success)
![AI Powered](https://img.shields.io/badge/AI-Google%20Gemini-magenta)
![Algorithm](https://img.shields.io/badge/Algorithm-k--NN-orange)

A sophisticated, asynchronous Telegram bot designed to forecast currency exchange rates using **Pattern Recognition (k-Nearest Neighbors)**. Unlike simple linear regression models, this bot analyzes historical price sequences to find similar market patterns from the past and projects them into the future.

It features a robust multi-threaded architecture, generates high-quality **animated GIFs** with motion trails, and integrates **Google's Gemini 2.5 Flash** to act as a context-aware financial assistant.

---

## 📑 Table of Contents
- [✨ Key Features](#-key-features)
- [🧠 The Science: How It Works](#-the-science-how-it-works)
  - [The Algorithm (k-NN)](#the-algorithm-k-nn)
  - [Cross-Rate Calculation](#cross-rate-calculation)
- [🤖 AI Integration (Gemini)](#-ai-integration-gemini)
- [🏗️ Technical Architecture](#%EF%B8%8F-technical-architecture)
  - [Concurrency Model](#concurrency-model)
  - [Visual Rendering Engine](#visual-rendering-engine)
- [📂 Project Structure](#-project-structure)
- [🚀 Installation & Setup](#-installation--setup)
- [⚙️ Configuration](#%EF%B8%8F-configuration)
- [🖥️ Usage Guide](#%EF%B8%8F-usage-guide)
- [🐛 Logging & Troubleshooting](#-logging--troubleshooting)
- [⚠️ Disclaimer](#%EF%B8%8F-disclaimer)

---

## ✨ Key Features

*   **Pattern-Based Forecasting**: Uses a **k-Nearest Neighbors (k-NN)** approach to identify historical market situations similar to the current trend.
*   **Multi-Currency Support**: Supports major pairs including **USD, EUR, JPY, GBP, CHF, and RUB**. Automatically calculates cross-rates (e.g., GBP/JPY) even if the source provides only rates against RUB.
*   **Dynamic Visualizations**:
    *   **Animated GIFs**: Smooth transitions showing the path from history to forecast with "motion trails" for visual impact.
    *   **Static PNGs**: High-contrast charts for quick loading.
    *   **Toggle Feature**: Users can switch between GIF and PNG formats on the fly.
*   **AI Financial Analyst**: Integrated **Google Gemini API** allows users to ask natural language questions. The AI receives the exact numerical context (forecast delta, volatility, history) to provide grounded answers.
*   **Resilient Data Pipeline**:
    *   Fetches data from the Central Bank of Russia (CBR) with automatic retries and connection pooling.
    *   Caches historical datasets to minimize API load.
    *   Interpolates missing data points (weekends/holidays).
*   **Asynchronous UI**: Uses a `ThreadPoolExecutor` to handle heavy plotting and network tasks in the background, ensuring the Telegram bot UI never freezes.

---

## 🧠 The Science: How It Works

### The Algorithm (k-NN)
Instead of trying to fit a straight line (Linear Regression) or guess up/down probabilities (Markov Chains), this bot uses **Non-Parametric Pattern Matching**.

1.  **Window Slicing**: The bot takes the most recent sequence of price changes (e.g., the last 21 days). This is the "Query Sequence".
2.  **History Search**: It scans the entire available history of that currency pair.
3.  **Euclidean Distance**: It calculates the mathematical distance between the "Query Sequence" and every possible window in the past.
    $$ d(p, q) = \sqrt{\sum (q_i - p_i)^2} $$
4.  **Neighbor Selection**: It selects the **$k$** (default: 30) closest historical matches (sequences that looked exactly like today's market).
5.  **Projection**: It looks at what happened *immediately after* those historical sequences.
6.  **Averaging**: The forecast is the average of those future outcomes.

*This approach effectively says: "The last time the market moved exactly like this, here is what happened next on average."*

### Cross-Rate Calculation
The data source (`cbr-xml-daily.ru`) provides rates relative to the Russian Ruble (RUB). To get a pair like **GBP/USD**, the bot performs synthetic cross-rate calculation:

$$ Rate_{GBP/USD} = \frac{Rate_{GBP/RUB}}{Rate_{USD/RUB}} $$

This allows prediction for any combination of supported currencies.

---

## 🤖 AI Integration (Gemini)

The bot uses the `google-generativeai` library to connect to the **Gemini 2.5 Flash** model.

**Context Injection:**
When a user asks a question (e.g., *"Should I buy now?"*), the bot doesn't just send the question. It constructs a rich prompt containing:
1.  **Conversation History**: The last 5 Q&A pairs to maintain context.
2.  **Current Data**: The specific currency pair, the current price, the forecasted price, and the calculated delta.
3.  **System Persona**: Instructions to act as a "Casual Financial Assistant."

This ensures the AI doesn't hallucinate random numbers but interprets the *actual* chart data generated by the k-NN model.

---

## 🏗️ Technical Architecture

### Concurrency Model
The bot is built on `python-telegram-bot` (synchronous mode wrapped in threads).
*   **Main Thread**: Handles Telegram updates (button clicks, commands).
*   **Worker Threads**: A `ThreadPoolExecutor` handles CPU-bound tasks (generating GIFs with Pillow) and I/O-bound tasks (fetching JSON from CBR).
*   **Locking**: Per-chat locks (`threading.RLock`) ensure that a user cannot spam buttons and corrupt their session state while a forecast is generating.

### Visual Rendering Engine
Located in `src/plotter.py`:
*   **Matplotlib**: Draws the base vector lines for history (white) and forecast (green/red).
*   **Pillow (PIL)**:
    *   Converts Matplotlib buffers to images.
    *   **Interpolation**: Generates intermediate frames between data points for fluid animation.
    *   **Motion Trails**: Adds a fading "ghost" effect to the leading edge of the animation line.

---

## 📂 Project Structure

```bash
forex-prediction-bot/
├── assets/                  # Static images (logos, loading states)
├── datasets/                # CSV files (if used for pre-training/caching)
├── logs/                    # Daily rotating logs
├── models/                  # Serialized .pkl files (cached k-NN vectors)
├── temp/                    # Temporary storage for generated GIFs/PNGs
├── src/
│   ├── main.py              # Configuration & Entry point
│   ├── telegram_bot.py      # Telegram handlers, UI logic, Threading
│   ├── model_engine.py      # k-NN logic, caching, math operations
│   ├── parser.py            # CBR API client, JSON parsing, retries
│   ├── plotter.py           # Matplotlib & Pillow animation logic
│   ├── logging_util.py      # Logger setup & error tracking
│   └── telegram_utils/      # Helper modules
│       ├── keyboards.py     # Inline keyboard generators
│       ├── state.py         # In-memory session management
│       ├── media_sender.py  # Logic to send/edit/replace photos
│       └── llm_client.py    # Gemini API wrapper
├── .env                     # Secrets (API Keys)
├── requirements.txt         # Dependencies
└── README.md                # This file
```

---

## 🚀 Installation & Setup

### Prerequisites
*   **Python 3.9+**
*   **Telegram Bot Token**: Get one from [@BotFather](https://t.me/BotFather).
*   **Google Gemini API Key**: Get one from [Google AI Studio](https://aistudio.google.com/).

### Step-by-Step

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/your-username/forex-prediction-bot.git
    cd forex-prediction-bot
    ```

2.  **Create Virtual Environment**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/Mac
    # venv\Scripts\activate   # Windows
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment**
    Create a `.env` file in the root directory:
    ```ini
    TELEGRAM_TOKEN=123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11
    GEMINI_API_KEY=AIzaSy...
    CACHE_TTL=300
    HTTP_POOL_SIZE=10
    ```

5.  **Run the Bot**
    ```bash
    cd src
    python main.py
    ```

---

## ⚙️ Configuration

You can tweak the internal logic in `src/main.py` and `src/model_engine.py`.

| Variable | Description | Default |
| :--- | :--- | :--- |
| `MODELS_SETTINGS["knn"]["k"]` | Number of neighbors to find. Higher = smoother, Lower = more volatile. | `30` |
| `MODELS_SETTINGS["knn"]["window_size"]` | How many past days to compare against history. | `21` |
| `HTTP_POOL_SIZE` | Max concurrent connections to the data provider. | `10` |
| `CACHE_TTL` | Time (seconds) to keep data in memory before re-fetching. | `300` |

---

## 🖥️ Usage Guide

1.  **Start**: Send `/start`. The bot initializes and shows a disclaimer.
2.  **Select Currency 1**: Choose the base currency (e.g., USD).
3.  **Select Currency 2**: Choose the quote currency (e.g., RUB).
4.  **Select Horizon**: Choose prediction depth (1 to 9 days).
5.  **Confirm**: The bot processes the data.
6.  **Result**:
    *   View the animated chart.
    *   Read the text summary (Trend direction, percentage change).
    *   **Buttons**:
        *   `Ask Question`: Opens a chat mode with Gemini about this specific graph.
        *   `Show Image/Animation`: Toggles between static PNG and GIF.
        *   `Back`: Return to currency selection.

---

## 🐛 Logging & Troubleshooting

The bot uses a sophisticated logging system (`logging_util.py`).

*   **Logs Location**: `logs/YYYYMMDD.log`
*   **Error Tracking**: Every exception generates a unique **RID (Request ID)** (e.g., `RID=a1b2c3d4`).
*   **User Feedback**: If an error occurs, the user sees "Unexpected error (id: a1b2c3d4)". You can grep this ID in the log file to find the exact stack trace.

**Common Issues:**
*   *Connection Error*: Check `cbr-xml-daily.ru` availability. The bot has built-in retries.
*   *Gemini Error*: Ensure your API key is valid and has quota.
*   *Missing Assets*: Ensure the `assets/` folder contains `logo.png`, `ai_thinking.png`, etc.

---

## ⚠️ Disclaimer

**This tool is for educational and research purposes only.**

The k-Nearest Neighbors algorithm assumes that history repeats itself, which is not always true in financial markets. This bot does not account for fundamental news, geopolitics, or black swan events.
*   **Do not use this bot as financial advice.**
*   **Do not trade real money based solely on these predictions.**

---

*License: MIT*