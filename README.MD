

# Currency Exchange Rates Prediction Telegram Bot

This project is a Telegram bot that provides currency exchange rate forecasts. It uses an ensemble of two models: Markov Chains to predict the direction of change (growth/decline) and Linear Regression to predict the magnitude of the change.

**TL;DR**: A Telegram bot that uses an ensemble of autoregressive models (Markov Chains and Linear Regression) to forecast currency exchange rates based on historical data from the Central Bank of Russia.

## Table of Contents

-   [Features](#features)
-   [How It Works](#how-it-works)
-   [Project Structure](#project-structure)
-   [Configuration](#configuration)
-   [Model Training](#model-training)
-   [Deployment Guide](#deployment-guide)
-   [Using the Bot](#using-the-bot)
-   [Contributing](#contributing)
-   [License](#license)

## Features

-   **Currency Pair Forecasting**: Generates forecasts for major currency pairs like EUR/USD, USD/RUB, etc.
-   **Visual Representation**: Displays the forecast as an easy-to-understand graph.
-   **Ensemble Models**: Combines predictions from Markov Chains (for direction) and Linear Regression (for magnitude) to improve accuracy.
-   **Interactive UI**: A user-friendly interface built with Telegram's inline keyboards.
-   **Live Data Fetching**: Automatically fetches the latest currency data from the Central Bank of Russia (CBR) API to ensure predictions is based on current market conditions.

## How It Works

1.  **Data Fetching**: The bot requests the latest and historical currency rates from the CBR's public API.
2.  **Preprocessing**: It processes this data to create sequences of prices, their differences (`diffs`), and the signs of their changes (`signs`).
3.  **Prediction**:
    -   **Markov Chains**: Analyze the sequence of `signs` to predict the next likely direction (up or down).
    -   **Linear Regression**: Analyzes the sequence of `diffs` to predict the numerical value of the next change.
4.  **Ensemble & Visualization**: The predictions from both models are combined to form a final forecast. The sign from the Markov Chain model is used to determine the sign of the predicted difference from the Regression model. The bot then generates a plot showing historical data alongside the projected future trend and sends it to the user.

## Project Structure

```
forex-prediction-bot/
├── src/
│   ├── main.py         # Main bot logic, handlers, and core functions.
│   └── .env            # Environment variables (e.g., your secret Telegram token).
├── datasets/           # Historical data (CSV files) used for training the models.
│   ├── EURRUB.csv
│   ├── EURUSD.csv
│   └── ...             # Other data files.
├── models/             # Pre-trained models saved in pickle (.pkl) format.
│   ├── markov_EURRUB_3.pkl
│   ├── regression_EURUSD_10.pkl
│   └── ...             # Other model files.
├── assets/
│   └── logo.png        # Optional logo for the bot's welcome message.
└── temp/               # A temporary directory where generated graphs are stored before being sent.
```

### Datasets Format

The bot requires historical data in CSV format for training the models. Each file should be named according to the currency pair it represents (e.g., `EURRUB.csv`) and must contain the following columns:

```csv
Date,Price,Sign,Difference
2005-04-04,35.938000,-,0.177000
2005-04-05,35.790000,-,0.148000
2005-04-06,35.845000,+,0.055000
2005-04-07,35.963000,+,0.118000
2005-04-08,35.725000,-,0.238000
```

-   **Date**: The date of the record (YYYY-MM-DD format).
-   **Price**: The exchange rate for that date.
-   **Sign**: The sign of the price change from the previous day (`+` or `-`).
-   **Difference**: The absolute change in price from the previous day (`Price_today - Price_yesterday`).

## Configuration

The bot's behavior can be configured through environment variables and settings within the `main.py` script.

### Environment Variables (.env)

Create a `.env` file in the `src/` directory to store your secret token.

**src/.env:**
```
TELEGRAM_TOKEN="YOUR_SUPER_SECRET_TOKEN_HERE"
```
**Security Note**: Ensure the `src/.env` file is never committed to version control. Add `src/.env` to your `.gitignore` file.

### Model Settings (`main.py`)

You can adjust the model training parameters by modifying the `MODELS_SETTINGS` dictionary in `src/main.py`:

```python
MODELS_SETTINGS = {
    "REBUILD": True,  # Set to True to force retraining of models on startup
    "reg": {
        "min_n": 3,  # Minimum number of lags for regression models
        "max_n": 10, # Maximum number of lags for regression models
    },
    "markov": {
        "min_n": 3,  # Minimum order for Markov chain models
        "max_n": 10, # Maximum order for Markov chain models
        "k" : 0.2,   # Smoothing factor for Markov chain probabilities
    }
}
```

## Model Training

The bot is designed to train its models automatically.

-   **Automatic Training**: On the first run, or if the `models/` directory is empty, the bot will train new models based on the data in the `datasets/` directory.
-   **Forced Retraining**: You can force the bot to retrain all models by setting `"REBUILD": True` in the `MODELS_SETTINGS` dictionary.
-   **Dependencies**: For training to work, the `datasets/` directory must contain the correctly formatted CSV files for all currency pairs you wish to predict.

## Deployment Guide

This guide will walk you through setting up and running the bot on a typical Linux server.

### Prerequisites

-   Python 3.7+
-   Pip (Python's package installer)
-   A Telegram Bot Token (you can get one by talking to [@BotFather](https://t.me/botfather) on Telegram)

### Step 1: Clone the Repository

First, clone your project repository from your Git provider to your server.

```bash
git clone https://github.com/maxy618/forex-prediction-bot
cd forex-prediction-bot
```

### Step 2: Set Up a Virtual Environment

Using a virtual environment is a best practice that isolates your project's dependencies from the system's global Python packages, preventing conflicts.

```bash
# Create a virtual environment named 'venv'
python3 -m venv venv

# Activate it
source venv/bin/activate
```
You will know it's active because your shell prompt will be prefixed with `(venv)`.

### Step 3: Install Dependencies

The project requires several Python libraries. These are listed in a `requirements.txt` file.

1.  Create a `requirements.txt` file in the root of your project (`forex-prediction-bot/requirements.txt`) with the following content:

    **requirements.txt:**
    ```
    python-telegram-bot
    requests
    numpy
    matplotlib
    python-dotenv
    ```

2.  Now, install all the required packages using pip:

    ```bash
    pip install -r requirements.txt
    ```

### Step 4: Configure File Paths

The script uses relative paths (e.g., `../datasets/`). For a stable and reliable deployment, you **must** change these to absolute paths in `src/main.py`. This prevents errors that occur when the script is run from a different directory.

**Example Change in `src/main.py`:**
```python
# FROM (relative paths):
DATASETS_PATH = "../datasets/"
MODELS_PATH = "../models/"
TEMP_FOLDER = "../temp/"
LOGO_PATH = "../assets/logo.png"

# TO (absolute paths):
import os
# Get the absolute path of the directory containing 'src', 'datasets', etc.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASETS_PATH = os.path.join(BASE_DIR, "datasets")
MODELS_PATH = os.path.join(BASE_DIR, "models")
TEMP_FOLDER = os.path.join(BASE_DIR, "temp")
LOGO_PATH = os.path.join(BASE_DIR, "assets", "logo.png")
```

### Step 5: Running the Bot

Once all the above steps are complete, you can run the bot.

1.  Make sure your virtual environment is still active (`source venv/bin/activate`).
2.  Navigate to the `src` directory.
3.  Execute the main script:

    ```bash
    cd src
    python main.py
    ```

If everything is configured correctly, your bot will start, and you should be able to interact with it on Telegram.

**Important Note on Production Use**: Running the bot directly like this is fine for testing, but the process will stop as soon as you close your SSH connection. For a production environment, you should use a process manager like `systemd` or `supervisor` to run the bot as a background service. These tools will automatically restart the bot if it crashes and ensure it runs continuously.

## Using the Bot

Interacting with the bot is straightforward:

1.  Find your bot on Telegram and start a conversation with it.
2.  Send the `/start` command.
3.  Click the "Make a forecast" button.
4.  Select the **base currency**.
5.  Select the **target currency**.
6.  Choose the number of days for the forecast (1-9).
7.  The bot will process your request and send you a graph with the prediction and a short text summary.

## Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement". Don't forget to give the project a star! Thanks again!

## License

Distributed under the MIT License. See `LICENSE` for more information.

---

Happy Pythoning!