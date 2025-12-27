

# Currency Exchange Rates Prediction Telegram Bot

## Overview

This project is a Telegram bot that provides automated forecasts for currency exchange rates (USD, EUR, JPY, GBP, CHF, RUB). It utilizes an ensemble of mathematical models: Linear Regression for predicting magnitude and Markov Chains with a **Temperature** parameter for determining trend direction. The bot fetches real-time data from the Central Bank of Russia (CBR), generates visual forecasts (plots and animated GIFs), and includes an AI assistant powered by Google's Gemini model to interpret the data.

## Features

- **Hybrid Prediction**: Uses an ensemble of Linear Regression (quantitative) and Markov Chains (directional).
- **Temperature Parameter**: Controls the randomness of the Markov model (<1 conservative, =1 honest, >1 random).
- **Visualizations**: Generates static PNG charts and animated GIFs showing historical data vs. forecast.
- **AI Assistant**: Chat with the bot to get financial advice or explanation of the charts via the Google Gemini API.
- **Real-time Data**: Fetches daily rates directly from CBR API.
- **State Management**: Handles user sessions seamlessly within Telegram.
- **Automated Training**: Trains models on historical CSV data if models are missing.

## Requirements

- Python 3.9+
- Libraries:
  - `requests`
  - `matplotlib`
  - `Pillow`
  - `python-telegram-bot`
  - `python-dotenv`
  - `numpy`
- External APIs:
  - Telegram Bot Token
  - Google Gemini API Key
- Data Structure:
  - `../datasets/` folder containing historical CSV files (e.g., `EURUSD.csv`) with columns: `Date`, `Price`, `Sign`, `Difference`.
  - `../models/` folder for storing trained model files (`.pkl`).

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/maxy618/forex-prediction-bot
   cd forex-prediction-bot
   ```

2. **Install dependencies**
   Create a `requirements.txt` if needed or run:
   ```bash
   pip install requests matplotlib Pillow python-telegram-bot python-dotenv numpy
   ```

3. **Environment Configuration**
   Create a `.env` file in the root directory:
   ```env
   TELEGRAM_TOKEN=your_telegram_bot_token_here
   GEMINI_API_KEY=your_gemini_api_key_here
   CACHE_TTL=300
   ```

4. **Prepare Assets**
   - Ensure `../datasets/` exists and contains training data.
   - Create `../temp/` for temporary media files.
   - Create `../assets/logo.png` for the bot start image.
   - Create `../assets/ai_thinking.png` and `../assets/predicting.png` for loading states.

## Usage

1. **Run the bot**
   Running `main.py` will automatically check for existing models. If they are missing or `REBUILD` is set to `True`, it will train new ones based on the CSV datasets.
   ```bash
   python main.py
   ```

2. **Interact in Telegram**
   - Send `/start` to initiate the conversation.
   - Follow the inline keyboard prompts to select the base currency, target currency, and number of days (1-9).
   - View the generated chart or GIF.
   - Use the "Ask question" button to consult the AI assistant regarding the forecast.

## Configuration

- **Currency List**: Modify `CURRENCIES` in `main.py`.
- **Model Settings**: Adjust `MODELS_SETTINGS` in `main.py`:
  - `reg`: Min/Max lag days (e.g., 3-10).
  - `markov`: Min/Max order (e.g., 3-10).
- **Temperature Parameter**: Defined in `model_engine.py` (Default: 0.3). Controls the "confidence" of the Markov chain predictions.
- **UI Strings**: All captions and button labels are configurable in the `user_interface` dictionary in `main.py`.

## How It Works

1. **Data Parsing** (`parser.py`): Fetches JSON data from CBR, handles retries, and calculates relative rates for currency pairs.
2. **Mathematical Core** (`model_engine.py`):
   - **Linear Regression**: Uses NumPy for matrix operations (Least Squares method) to predict numeric differences.
   - **Markov Chains**: Builds transition matrices from historical signs and applies a **Temperature** scaling to probabilities (`counts_to_probabilities`).
3. **Visualization** (`plotter.py`): Uses Matplotlib to render historical data (white line) and forecasts (colored line). Supports creating transition frames for GIFs.
4. **Telegram Interface** (`telegram_bot.py`): Manages user states, handles callbacks, threads background tasks for AI/API calls, and serves media files.
5. **Logging** (`logging_util.py`): Implements daily rotating logs to track system health.

## License

MIT License.