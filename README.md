# Currency Exchange Rates Prediction Telegram Bot

## Overview

This project is a Telegram bot that provides forecasts for currency exchange rates (EUR, USD, RUB) using statistical models like linear regression and Markov chains. It fetches real-time and historical data from the Central Bank of Russia (CBR), generates visual forecasts as plots or animated GIFs, and offers AI-powered advice via Google's Gemini model

## Features

- **Currency Pair Selection**: Choose pairs like EUR/USD, USD/RUB, etc
- **Forecasting**: Predict exchange rates for 1–9 days ahead using pre-trained models
- **Visualizations**: Static PNG plots or animated GIFs showing historical and forecasted rates
- **AI Advice**: Ask questions about the forecast, and get responses from Gemini AI
- **Caching**: Efficient data handling with caching for recent forecasts
- **Logging**: Detailed logging with rotation for debugging.
- **Model Training**: Automatically trains models from CSV datasets if needed

## Requirements

- Python 3.12+
- Libraries: `requests`, `matplotlib`, `Pillow`, `python-telegram-bot`, `python-dotenv`, `scikit-learn` (implied for regression), `numpy` (for data handling)
- API Keys: Telegram Bot Token, Google Gemini API Key.
- Datasets: CSV files in `../datasets/` (e.g., `EURUSD.csv`) with columns ```Date,Price,Sign,Difference```
- Models: Stored in `../models/` (auto-generated if missing)

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd <project-directory>
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
   (Note: Create `requirements.txt` based on imports, e.g., requests, matplotlib, Pillow, python-telegram-bot, python-dotenv.)

3. Set up environment variables in `.env`:
   ```
   TELEGRAM_TOKEN=your-telegram-bot-token
   GEMINI_API_KEY=your-gemini-api-key
   CACHE_TTL=300  # Optional, in seconds
   ```

4. Prepare directories:
   - `../datasets/` for CSV data
   - `../models/` for trained models
   - `../temp/` for temporary files (PNGs/GIFs)
   - `../assets/logo.png` for bot logo (optional)

## Usage

1. Train models (if not already done):
   - Run `main.py`—it will check and train models using data from `../datasets/`
   ```
   python main.py
   ```

2. Start the bot:
   - The bot has started polling Telegram for messages

3. Interact with the bot:
   - Send `/start` to begin
   - Follow prompts: Select base currency, target currency, forecast days
   - View forecast plot/GIF
   - Toggle between static/image and animated/GIF
   - Ask questions for AI advice (e.g., "Is it a good time to buy? / when should i sell this one?")

## Configuration

- **Currencies**: Defined in `main.py` as `["EUR", "USD", "RUB"]`. Add more if needed
- **Models**:
  - Regression: Lags from 3 to 10
  - Markov: Orders from 3 to 10, with Laplace smoothing (k=0.2)
  - Set `MODELS_SETTINGS["REBUILD"] = True` to retrain
- **UI Texts**: Customizable in `main.py` under `user_interface` (captions and buttons)
- **Gemini Prompt**: Template in `main.py` for AI responses—edit for tone/style
- **HTTP Settings**: Pool size via `HTTP_POOL_SIZE` env var (default 10)

## How It Works

1. **Data Fetching** (`parser.py`): Pulls JSON from CBR for daily rates, handles retries
2. **Model Training** (`model_engine.py`, not shown in code snippet but implied): Builds regression coefficients and Markov transition matrices from CSV diffs/signs
3. **Forecasting** (`telegram_bot.py`): Uses models to predict diffs/signs, adjusts forecasts
4. **Plotting** (`plotter.py`): Creates matplotlib plots/GIFs with transitions
5. **Bot Logic** (`telegram_bot.py`): Handles user interactions, state management, AI queries
6. **Logging** (`logging_util.py`): Rotates logs daily in `../logs/`

## Contributing

- Fork the repo
- Create a feature branch
- Submit a pull request with clear descriptions

## License

MIT License. See `LICENSE` file for details.