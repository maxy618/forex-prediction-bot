# forex-prediction-bot

A compact, experimental Telegram bot and a small toolkit for short-horizon exchange-rate forecasting. The project is intentionally simple — it uses an ensemble of
small models: Markov chains to predict direction (sign) and lagged linear regression to predict magnitude.

This repository includes the data pipeline, training utilities, plotting helpers and a Telegram UI for interactive testing.

📘 Русская версия: `README_RU.md`

---

## Quick overview

- Purpose: provide a compact research / demo project for short-horizon currency rate predictions and a Telegram-based UI for quick interactive forecasts.
- Approach: direction predicted with small-order Markov chains over sign sequences; magnitude predicted with simple lagged linear regressions. The final forecast is an ensemble of those components.

Use-cases: rapid prototyping, research experiments or learning how simple forecasting pipelines can be composed into an interactive bot.

---

## Highlights & Features

- Small, readable codebase focused on clarity and experimentation
- Data parsing utilities for daily rates (CSV-based dataset in `datasets/`) and helper functions to fetch/parsers historical JSON sources
- Training routines that persist Markov and regression models to `models/`
- Plotting + convenient PNG exports for visual inspection (`temp/`)
- A minimal Telegram bot (startup in `src/main.py` → runtime handlers in `src/telegram_bot.py`) for interactive predictions using trained models

---

## Getting started (short)

Requirements: Python 3.8+ and pip.

1) Create & activate a virtual environment

Windows (PowerShell):

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

2) Install dependencies

```powershell
pip install -r requirements.txt
```

3) Add your Telegram token (create `src/.env`)

```env
TELEGRAM_TOKEN=YOUR_BOT_TOKEN_HERE
```

4) Run the bot (development)

```powershell
python src/main.py
```

On the first run the code will train and save model files under `models/` if `REBUILD` is enabled.

---

## Key code paths

The relevant code is split into a few focused modules under `src/`:

- `src/main.py` — small orchestrator: trains models on first run (if enabled) and launches the Telegram runtime via `telegram_bot.telegram_main(config)`.
- `src/telegram_bot.py` — Telegram handlers, UI and per-chat state (runtime logic).
- `src/parser.py` — HTTP session and fetching helpers used to collect historical rates.
- `src/model_engine.py` — modeling code: Markov helpers, regression helpers and save/load functions.
- `src/plotter.py` — plotting helpers for PNG previews used by the bot.
- `datasets/` — CSV files with historical daily prices (source files used for training)
- `models/` — saved model artifacts (Markov & regression pickles / safetensors)
- `temp/` — generated PNG plots used by the bot

---

## Data format

Each CSV in `datasets/` should contain at least a `Date` column and a `Price` column. The repository computes per-day differences and signs if not present.

Minimal example:

```csv
Date,Price
2005-04-04,35.938
2005-04-05,35.790
2005-04-06,35.845
```

The training code expects sequences to be ordered oldest → newest.

---

## Models & how forecasting works (brief)

- Markov models are built over sign sequences (e.g. last N +/− tokens) and estimate probability of next sign.
- Regression models predict the numeric next-day difference using a window of previous differences.
- Ensemble:
  - majority vote across Markov models for direction
  - average of regression outputs for magnitude
  - the direction is used to sign the averaged magnitude when needed

Model artifact naming convention (examples):

- `markov_EURUSD_3.*` — markov model order 3 for EUR→USD
- `regression_EURUSD_5.*` — regression with 5 lag terms

---

## Configuration & environment

Defaults live in `src/main.py` as `MODELS_SETTINGS`. Typical options:

- `REBUILD` — when true, retrains models at startup
- `markov.min_n / max_n` — range of orders to build
- `markov.k` — add-k smoothing value
- `reg.min_n / max_n` — lags range for regressions

Adjust these values while developing; training cost is small for the short ranges used by this demo.

---

## Developer notes

- This project is intentionally compact and educational — models are simple and not production-ready.
Suggested improvements: regularized regression, probabilistic calibration, richer features (e.g., volatility, volumes), improved ensemble logic and test coverage.

If you plan to extend the project:

- Add a dataset: drop a CSV into `datasets/` (see Data format) and rebuild models.
- Modify runtime behavior: update `src/telegram_bot.py` and adjust config in `src/main.py`.

I can also add a small example (adding a dataset + training) or tests if you want — tell me which you'd prefer next.

---

## License

MIT — see the `LICENSE` file.

---

If you want, I can now polish messaging, add quick examples or provide a short developer guide (how to add a new dataset / model). Which would you prefer next?