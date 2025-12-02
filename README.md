📘 [Русская версия](README_RU.md)

# Currency Exchange Rates Prediction Telegram Bot

> **TL;DR:** A small Telegram bot that forecasts currency exchange rates using an ensemble of Markov chains (direction prediction) and lagged linear regression (magnitude prediction). The repository contains the bot, data parsing, model training, and plotting utilities.

---

## Table of Contents

* [Overview](#overview)
* [Concept and algorithms (detailed)](#concept-and-algorithms-detailed)

  * [Preprocessing and features](#preprocessing-and-features)
  * [Markov chains — construction and smoothing](#markov-chains---construction-and-smoothing)
  * [Laplace (add-k) smoothing — formula](#laplace-add-k-smoothing---formula)
  * [Lagged linear regression — maths and numerics](#lagged-linear-regression---maths-and-numerics)
  * [Ensembling rules and tie-breaking](#ensembling-rules-and-tie-breaking)
* [Project structure](#project-structure)
* [Data format (expanded)](#data-format-expanded)
* [Local setup and run](#local-setup-and-run)
* [Configuration and parameters](#configuration-and-parameters)
* [Training and forecasting (implementation notes)](#training-and-forecasting-implementation-notes)
* [Plotting and visualization details](#plotting-and-visualization-details)
* [Parser and data-fetching robustness](#parser-and-data-fetching-robustness)
* [Interpretation and limitations (technical)](#interpretation-and-limitations-technical)
* [License](#license)

---

## Overview

This repository implements a compact pipeline that:

* fetches daily exchange rates (source: Central Bank of Russia JSON API),
* constructs symbol-level price sequences for pairs such as `USD_per_EUR`, `EUR_per_RUB`, etc.,
* computes daily differences and signs (up/down),
* trains Markov chain models on sign sequences to estimate direction probabilities,
* trains lagged linear regressions on differences to estimate the magnitude of the next change,
* combines outputs from multiple models into an ensemble forecast and produces a compact plot for N days ahead,
* exposes a Telegram bot UI for interactive predictions.

---

## Concept and algorithms (detailed)

### Preprocessing and features

Input: a time-ordered series of daily prices `Price(t)`.

Compute:

1. `diff(t) = |Price(t) - Price(t-1)|` — daily change (float).
2. `sign(t) = '+' if diff(t) >= 0 else '-'` — discrete direction label.

Notes and edge cases:

* If there are missing calendar days in the CSV, adjacent rows are treated as consecutive trading points unless data is explicitly filled.
* Zero diffs produce `-` in the current implementation (arbitrary tie-breaker).
* Small floating rounding differences can flip a sign; be aware when interpreting short-term patterns.

### Markov chains — construction and smoothing

A Markov model of order `n` (n-gram over signs) estimates probabilities of the next symbol given the last `n` symbols.

Implementation summary (function: `build_markov_model(sequence, order)`):

* Iterate through the sign sequence and collect counts `counts[state][next]` where `state` is a tuple of `order` previous signs.
* Convert counts to probabilities with an add-`k` smoothing function (implemented in `counts_to_probabilities`).
* The resulting model is `{ "order": n, "table": { state_tuple: { next_sign: probability, ... }, ... } }`.

Predicting: the code picks the last `order` signs, looks them up in the table and samples the next sign according to the stored probabilities. If the state is absent, that model abstains from voting in the ensemble.

Important implementation details:

* If `order >= len(sequence)` the builder raises an error.
* The `table` is a dict keyed by tuples.
* Sampling is pseudo-random; a deterministic forecast can be obtained by selecting the most-probable sign.

### Laplace (add-k) smoothing — formula

Smoothing assigns non-zero mass to unseen transitions.

Given observed counts `m_i` for each symbol `i`, let `n = sum_i m_i`, and `v` be the number of distinct next-symbol values. The smoothed probability for symbol `i` is:

```
P(i) = (m_i + k) / (n + k * v)
```

The code defines `k = MODELS_SETTINGS["markov"]["k"]` (default `0.2`).

### Lagged linear regression — maths and numerics

The regression model predicts the next `diff` using a linear combination of the previous `n_lags` diffs:

```
ŷ_t = w_1 * diff_{t-1} + w_2 * diff_{t-2} + ... + w_n * diff_{t-n} + b
```

Design matrix `X` rows: `[diff_{t-n}, ..., diff_{t-1}]`, target vector `y = diff_t`.

Parameter estimation (function: `build_regression`):

* Build `X_with_bias` by appending a column of ones to `X`.
* Solve for coefficients; implementations may use normal equations or a numerically-stable alternative.

Prediction (function: `predict_diff`) is `weights.dot(inputs) + bias`.

### Ensembling rules and tie-breaking

Ensemble composition in the repository:

* **Sign (direction):** each Markov model votes for `+` or `-`. The ensemble selects the majority vote. Models with missing states abstain. Ties are resolved by random choice between tied leaders.
* **Magnitude:** regression models return numeric diff predictions; the ensemble averages these numbers.
* **Adjustment rule:** if the averaged regression magnitude has sign opposite to the Markov majority, the implementation applies the Markov sign to the magnitude (i.e. `sign * abs(magnitude)`).

This README records the ensemble logic implemented in code (no additional fusion strategies are described here).

## Project structure

```
forex-prediction-bot/
├── src/
│   ├── main.py         # Main script: bot logic, training and utilities
│   └── .env            # (local) TELEGRAM_TOKEN
├── datasets/           # CSV historical data (format below)
├── models/             # Saved .pkl models (markov_..., regression_...)
├── assets/             # static files (logo.png)
├── temp/               # temporary images before sending
├── requirements.txt    # dependencies
└── README.md           # this file (English)
```

## Data format (expanded)

Each CSV file in `datasets/` corresponds to a currency pair and should have a header. Minimal required column: `Price`.

Canonical format (daily rows, oldest → newest):

```csv
Date,Price,Sign,Difference
2005-04-04,35.938000,-,0.177000
2005-04-05,35.790000,-,0.148000
2005-04-06,35.845000,+,0.055000
```

Column descriptions:

* `Date` — ISO date `YYYY-MM-DD`.
* `Price` — numeric price/rate (float).
* `Sign` — optional `+`/`-` token. If absent, `sign` can be computed from `Price`.
* `Difference` — optional `Price_today - Price_yesterday` entry. If absent, recompute from `Price`.

Preprocessing notes: trim leading NaNs and rows with malformed values. When combining sources, ensure consistent quoting direction and document file names (`USDEUR.csv` or `USD_EUR.csv`).

## Local setup and run

**Requirements:** Python 3.8+. `pip`.

**Install dependencies:**

```bash
python -m venv venv
# activate venv
# Linux / macOS
source venv/bin/activate
# Windows
venv\Scripts\activate

pip install -r requirements.txt
```

Minimal `requirements.txt` example:

```
python-telegram-bot
requests
matplotlib
numpy
python-dotenv
```

**.env** (create `src/.env` and do not commit):

```
TELEGRAM_TOKEN="YOUR_BOTFATHER_TOKEN"
```

**Run the bot**:

```bash
python src/main.py
```

On first run the training routine will save regression and markov model pickles in `models/`.

## Configuration and parameters

`MODELS_SETTINGS` in `src/main.py` controls defaults:

```python
MODELS_SETTINGS = {
    "REBUILD": True,    # force retrain on each start
    "reg": {"min_n": 3, "max_n": 10},
    "markov": {"min_n": 3, "max_n": 10, "k": 0.2}
}
```

* `REBUILD` — retrain on start when `True`.
* `reg.min_n, reg.max_n` — range of regression lag counts to train.
* `markov.min_n, markov.max_n` — range of Markov orders to train.
* `markov.k` — add-`k` smoothing constant.

Other constants in `main.py` include `CURRENCIES`, `BASE_LATEST`, `BASE_ARCHIVE`, and HTTP session settings.

## Training and forecasting (implementation notes)

Model file naming:

* `regression_{BASE}{QUOTE}_{n}.pkl` — coefficients for `n` lags (vector length `n + 1`).
* `markov_{BASE}{QUOTE}_{order}.pkl` — dict with `order` and `table`.

Training loop highlights (`train_models_if_needed`):

1. Iterate over non-diagonal pairs of `CURRENCIES` to create pairs like `(EUR, USD)`.
2. Read `Sign` and `Difference` columns from `datasets/{A}{B}.csv`.
3. For each `n_lags` in the regression range, call `build_regression(diffs, n_lags)` and save the result.
4. For each `order` in the Markov range, build and save models.

Forecasting flow:

1. Fetch recent `needed_days` rates via `fetch_sequences_all_pairs`.
2. Build `diffs` and `signs` sequences from prices.
3. Load available regression and markov models from `models/`.
4. Compute `forecasted_diffs` (averaged regression outputs) and `forecasted_signs` (ensemble Markov votes).
5. Apply sign adjustment when the Markov majority sign is present.
6. Convert diffs to cumulative `new_prices` and plot.

## Plotting and visualization details

`plot_sequence` renders old prices and predicted prices. X labels are calendar-based; the predicted-line color is set by comparing final predicted vs last observed price. The plotting utilities produce PNG images saved to `temp/`.

## Parser and data-fetching robustness

`fetch_sequences_all_pairs` builds ISO-date → pair rates for a configured window of days. For each requested day, it queries the configured API endpoints and converts JSON rates into pairwise ratios `target_per_base`.

Notes on behavior:

* The function carries forward the last known value for missing days by design. The code will surface partial-data cases via logs.
* Rate-limiting and API availability should be considered when requesting long historical windows.

## Interpretation and limitations (technical)

This section lists technical caveats:

* **Model expressivity:** Markov chains on sign sequences and linear regression on diffs capture limited short-term dependencies.
* **Numerical stability:** Regression may be ill-conditioned depending on `n_lags` and available data.
* **Combining outputs:** The ensemble is heuristic and reflects the exact fusion logic implemented in code.
* **Sampling vs argmax:** The code uses sampling for Markov predictions by default; deterministic behavior can be obtained by selecting the most-probable symbol in the model table.

## License

This project is distributed under the MIT License — see `LICENSE`.

---