📘 [Русская версия](README_RU.md)

# Currency Exchange Rates Prediction Telegram Bot

> **TL;DR:** A small Telegram bot that forecasts currency exchange rates using a simple ensemble of Markov chains (direction prediction) and lagged linear regression (magnitude prediction). The repository contains the bot, data parsing, model training, and plotting utilities.

---

## Table of Contents

* [Overview](#overview)
* [Concept and algorithms (detailed)](#concept-and-algorithms-detailed)

  * [Preprocessing and features](#preprocessing-and-features)
  * [Markov chains — construction and smoothing](#markov-chains---construction-and-smoothing)
  * [Laplace (add-k) smoothing — deep dive](#laplace-add-k-smoothing---deep-dive)
  * [Lagged linear regression — maths and numerics](#lagged-linear-regression---maths-and-numerics)
  * [Ensembling rules and tie-breaking](#ensembling-rules-and-tie-breaking)
* [Project structure](#project-structure)
* [Data format (expanded)](#data-format-expanded)
* [Local setup and run](#local-setup-and-run)
* [Configuration and parameters (explained)](#configuration-and-parameters-explained)
* [Training and forecasting (implementation notes)](#training-and-forecasting-implementation-notes)
* [Plotting and visualization details](#plotting-and-visualization-details)
* [Parser and data-fetching robustness](#parser-and-data-fetching-robustness)
* [Interpretation and limitations (technical)](#interpretation-and-limitations-technical)
* [Ideas for improvements / TODO (practical roadmap)](#ideas-for-improvements--todo-practical-roadmap)
* [License](#license)

---

## Overview

This repository implements a compact pipeline that:

* fetches daily exchange rates (source: Central Bank of Russia JSON API),
* constructs symbol-level price sequences for pairs such as `USD_per_EUR`, `EUR_per_RUB`, etc.,
* computes daily differences and signs (up/down),
* trains small Markov chain models on sign sequences to estimate the *direction* probability,
* trains simple lagged linear regressions on differences to estimate the *magnitude* of the next change,
* combines outputs from multiple models into an ensemble forecast and produces a compact plot for N days ahead,
* exposes a Telegram bot UI for interactive predictions.

This document expands the original README by adding deeper explanations of algorithmic choices (in particular smoothing), numerical caveats, and concrete implementation recommendations.

---

## Concept and algorithms (detailed)

### Preprocessing and features

Input: a time-ordered series of daily prices `Price(t)`.

Compute:

1. `diff(t) = Price(t) - Price(t-1)` — raw daily change (signed float). If you prefer relative changes, use `pct_change(t) = Price(t)/Price(t-1) - 1`.
2. `sign(t) = '+' if diff(t) >= 0 else '-'` — discrete direction label.

Notes and edge cases:

* If there are missing calendar days in the CSV, the code currently assumes adjacent rows represent consecutive trading points. If you want to treat calendar days explicitly, fill missing dates with NaNs or carry-forward the last known price depending on intent.
* Zero diffs produce `+` in the current implementation; that choice is an arbitrary tie breaker — you can map zeros to a neutral token (e.g. `0`) if you later want to train a 3-state Markov model.
* Very small floating rounding differences may flip a sign. Consider rounding diffs to a small epsilon when stability is desired.

### Markov chains — construction and smoothing

A Markov model of order `n` (n-gram over signs) estimates probabilities of next symbol given the last `n` symbols.

Implementation summary (function: `build_markov_model(sequence, order)`):

* Iterate through the sign sequence and collect counts `counts[state][next]` where `state` is a tuple of `order` previous signs.
* Convert counts to probabilities with an add-`k` smoothing function (implemented in `counts_to_probabilities`).
* The resulting model is `{ "order": n, "table": { state_tuple: { next_sign: probability, ... }, ... } }`.

When predicting, the code picks the last `order` signs, looks them up in the table and samples the next sign using the probability distribution. If the state is absent, it returns `None` and that model abstains from voting in the ensemble.

Important implementation details:

* If `order >= len(sequence)` the builder raises an error — training a higher-order model than the sequence length is meaningless.
* The `table` is a dict keyed by tuples; using tuples instead of string-joined tokens preserves clarity and avoids collisions.
* The sampling in `predict_state` is pseudo-random; to produce deterministic forecasts you can replace sampling with `argmax` on probabilities.

### Laplace (add-k) smoothing — deep dive

Smoothing is crucial for robust probability estimation when some `state->next` transitions were never observed.

**The add-k smoothing formula used**:

Given a counter with observed counts `m_i` for each possible symbol `i` (here `i` in `{'+', '-'}`), let `n = sum_i m_i`, and `v` be the number of distinct next-symbol values observed in the counter (in our case `v <= 2`). The smoothed probability for symbol `i` is:

```
P(i) = (m_i + k) / (n + k * v)
```

This is a generalization of Laplace (add-1) smoothing. The code defines `k = MODELS_SETTINGS["markov"]["k"]` and uses a small `k` (default `0.2`).

**Why smoothing matters**:

* Without smoothing, an unobserved transition has estimated probability `0`. This leads to brittle sampling and zero-probability traps when chaining predictions.
* Smoothing assigns non-zero mass to unseen transitions, enabling exploration and avoiding overconfident zero-probability events.

**Choice of k**:

* `k = 1.0` — classic Laplace smoothing. Aggressive for small `n` (pulls distribution towards uniform).
* `0 < k < 1` — milder smoothing (the default `0.2` in this repo), allowing observed counts to dominate while still protecting unseen events.

**Practical guidance and examples**:

Assume state `('+' , '+')` has counts `{ '+': 4, '-': 1 }`.

* With `k=0` → probabilities `{ '+': 4/5 = 0.8, '-': 0.2 }`.
* With `k=1` → `{ '+': (4+1)/(5+1*2) = 5/7 ≈ 0.714, '-': (1+1)/7 = 2/7 ≈ 0.286 }`.
* With `k=0.2` → denominator `5 + 0.2*2 = 5.4`; `P('+') = 4.2/5.4 ≈ 0.7778; P('-') = 1.2/5.4 ≈ 0.2222`.

So `k=0.2` gently shifts probabilities towards uniformity without overruling the data.

**Edge cases**:

* If a state has only one observed next value (e.g. `{'+': 3}`) and `k` is small, the smoothed probabilities remain close to `{'+': 1.0, '-': small}`.
* If `v` is taken as `len(counter)` (observed distinct next tokens) then transitions to tokens that were never seen are still impossible unless you explicitly enumerate the full token set (e.g. `['+', '-', '0']`). For binary tokens `+/-` this is not an issue, but for models with more possible tokens you may want to force `v` to be the size of the global alphabet.

**Alternative smoothing approaches**:

* **Backoff models**: if the exact `order` state is missing, back off to a lower-order state count. That generally improves coverage (useful for higher order models).
* **Kneser-Ney / Good-Turing**: more advanced and powerful for language modeling — overkill for binary signs but conceptually interesting.

**Implementation hints**:

* Consider passing an explicit `alphabet` to `counts_to_probabilities` so `v` can be constant across states (recommended when you expand token set).
* Normalize numerically stable: compute `p = (m + k) / (n + k * v)` but guard `n + k * v != 0`.
* When sampling across probabilities, ensure the sum is ~1.0; re-normalize floating point results to avoid cumulative rounding errors.

### Lagged linear regression — maths and numerics

The regression model predicts the next `diff` using a linear combination of the previous `n_lags` diffs. Formally:

```
ŷ_t = w_1 * diff_{t-1} + w_2 * diff_{t-2} + ... + w_n * diff_{t-n} + b
```

The code builds the design matrix `X` with rows `[diff_{t-n}, ..., diff_{t-1}]` and target vector `y = diff_t`.

**Parameter estimation** (function: `build_regression`):

* Compute `X_with_bias` by appending a column of ones to `X`.
* Solve `coeffs = (X^T X)^{-1} X^T y` (normal equations) and return the flattened coefficients `[w_1, ..., w_n, b]`.

**Numerical caveats**:

* The matrix `X^T X` can be singular or ill-conditioned (especially when features are collinear or when `N` is small relative to `n_lags`). Inverting it directly may raise `LinAlgError` or amplify noise.
* Recommended robust alternatives:

  * Use `np.linalg.pinv(XtX)` (Moore–Penrose pseudo-inverse), e.g. `coeffs = np.linalg.pinv(X_with_bias) @ y_vec`.
  * Add ridge regularization: solve `(X^T X + λ I) θ = X^T y`. This shrinks coefficients toward zero and stabilizes inversion. Small λ (1e-6 .. 1e-2) often helps.

**Implementation variation using pseudo-inverse**:

```
coeffs = np.linalg.pinv(X_with_bias) @ y_vec
```

This computes coefficients more stably and avoids explicit inversion of `X^T X`.

**Prediction (function: `predict_diff`)**:

* The prediction is simply `weights.dot(inputs) + bias`. In code it's implemented as an explicit loop; a dot product is more concise and efficient.

**Practical considerations**:

* If `n_lags` is large relative to available training rows, you will overfit. Limit `n_lags` or increase data.
* Consider scaling diffs (standardization) if magnitudes vary widely — this can improve conditioning.
* Evaluate models with rolling window validation to measure real generalization performance (MAE/RMSE).

### Ensembling rules and tie-breaking

The repo uses a simple ensemble:

* **Sign (direction)**: Each Markov model votes for `+` or `-`. The ensemble selects the majority vote. If models abstain (missing state) they don't cast a vote. Ties are resolved by random choice between tied leaders.
* **Magnitude**: Each regression model returns a numeric diff prediction; the ensemble averages these numbers.
* **Adjustment rule**: If the averaged regression magnitude has sign opposite to the Markov majority, the implementation forces the Markov sign onto the absolute magnitude (i.e. `sign * abs(magnitude)`). This ensures the ensemble respects the categorical direction vote.

Why this design?

* Separating direction and magnitude allows different model classes to focus on what they do best: discrete pattern recurrence (Markov) vs. linear trend extrapolation (regression).
* The adjustment rule is a pragmatic decision to avoid contradictory forecasts (a negative direction predicted by Markov while mean regression predicts a positive diff). Other reconciliation strategies are possible (weighted averaging, confidence-based fusion).

Improvements you can try:

* Weight Markov votes by model order confidence (e.g., higher-order models get higher weight only when their state is present), or weight regressions by in-sample RMSE.
* Use probabilistic magnitude combination: sample magnitudes from a residual distribution rather than deterministic average.

## Project structure

```
forex-prediction-bot/
├── src/
│   ├── main.py         # Main script: bot logic, training and utilities (provided)
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

Recommended canonical format (daily rows, oldest → newest):

```csv
Date,Price,Sign,Difference
2005-04-04,35.938000,-,0.177000
2005-04-05,35.790000,-,0.148000
2005-04-06,35.845000,+,0.055000
```

Column descriptions and additional recommendations:

* `Date` — ISO date `YYYY-MM-DD`. Keep it for human-readability and plotting. If you omit it, the code still works assuming the sequence order is preserved.
* `Price` — numeric price/rate (float). If you have quoted rates with commas, normalize to dot decimals.
* `Sign` — optional `+`/`-` token. If absent, `sign` can be computed from `Price`.
* `Difference` — optional `Price_today - Price_yesterday` entry. If absent, recompute from `Price`.

Preprocessing suggestions:

* Trim leading NaNs and rows with malformed values.
* Optionally, add a `Volume` column if you plan to extend models.
* When combining data sources, ensure consistent quoting direction (which currency is numerator/denominator) and document file names accordingly (`USDEUR.csv` or `USD_EUR.csv`).

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

`requirements.txt` minimal set:

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

**Optional file paths**: the script uses relative paths by default. For reliability in production, set them to absolute paths inside `main.py` as shown in the original README.

**Run the bot**:

```bash
# from repository root
python src/main.py
```

On first run the training routine `train_models_if_needed(...)` will save regression and markov model picks in `models/`.

## Configuration and parameters (explained)

`MODELS_SETTINGS` in `src/main.py` controls defaults:

```python
MODELS_SETTINGS = {
    "REBUILD": True,    # force retrain on each start
    "reg": {"min_n": 3, "max_n": 10},
    "markov": {"min_n": 3, "max_n": 10, "k": 0.2}
}
```

* `REBUILD = True` — retrain every start (development-friendly). Set to `False` to preserve existing pickles.
* `reg.min_n, reg.max_n` — range of regression lag counts to train (inclusive).
* `markov.min_n, markov.max_n` — range of Markov orders to train.
* `markov.k` — the add-`k` smoothing constant; defaults to `0.2`.

**Other constants in `main.py`:**

* `CURRENCIES` — list of 3-letter codes used to build pairwise sequences.
* `BASE_LATEST`, `BASE_ARCHIVE` — endpoints for CBR API.
* `SESSION` & `retry_strategy` — HTTP settings for robust data fetching.

## Training and forecasting (implementation notes)

**Model files naming**:

* `regression_{BASE}{QUOTE}_{n}.pkl` — coefficients for `n` lags (vector length `n + 1` including bias).
* `markov_{BASE}{QUOTE}_{order}.pkl` — dict with `order` and `table`.

**Training loop highlights (`train_models_if_needed`)**:

1. Iterate over non-diagonal pairs of `CURRENCIES` to create pairs like `(EUR, USD)`.
2. Read `Sign` and `Difference` columns from `datasets/{A}{B}.csv`.
3. For each `n_lags` in the regression range, call `build_regression(diffs, n_lags)`, save the result.
4. For each `order` in the Markov range, build and save models.

**Forecasting flow (`_perform_prediction_and_edit`)**:

1. Fetch recent `needed_days` rates (max of configured model orders/lags) via `fetch_sequences_all_pairs`.
2. Build `diffs` and `signs` sequences from prices.
3. Load available regression and markov models from `models/`.
4. Compute `forecasted_diffs` with `forecast_diffs` (averaged regression outputs) and `forecasted_signs` with `forecast_signs` (ensemble Markov votes).
5. Apply sign adjustment: if Markov sign is present and contradicts regression sign, flip magnitude sign.
6. Convert diffs to cumulative `new_prices` and plot.

## Plotting and visualization details

The plotting function `plot_sequence` renders old prices (white line) and predicted prices (green/red depending on last change). Important details:

* Axes, spines and ticks are styled for a dark background.
* X labels are calendar-based (`%d.%m`) starting at `today - (m - 1)` where `m = len(old_prices)`.
* The color for predicted line is chosen by comparing `new_prices[-1]` to `old_prices[-1]`.

Improvements and alternatives:

* Add confidence bands: compute ensemble stdev across regression outputs and draw shaded area.
* Plot the underlying diffs as bar chart in a secondary axis to visualize volatility.
* Add markers for observed vs predicted points to help visual diagnosis.

## Parser and data-fetching robustness

`fetch_sequences_all_pairs` builds a dictionary of ISO-date → pair rates for `days` days back from today. The process:

* For each day in the window, call `fetch_for_date` which attempts `base_latest` or `base_archive/YYYY/MM/DD/daily_json.js`.
* On failures, it retries once (with a short sleep) and finally tries `base_latest` as a fallback for previous days.
* Convert the JSON rates into a matrix of pairwise ratios `target_per_base`.

Potential issues and fixes:

* Rate limits: `Retry` strategy is configured. If you request long historical windows, respect API limits.
* Partial data: the function keeps `last_known` per currency and carries forward the last observed value for missing days — this implicitly fills holes but may hide gaps. If you prefer to fail on gaps, modify the logic to raise when data is missing.
* Timezone and calendar differences: the API provides end-of-day rates for Moscow time; be careful when aligning with other data sources.

## Interpretation and limitations (technical)

This section lists technical caveats without prescriptive usage guidance.

* **Model expressivity:** Markov chains on sign sequences and linear regression on diffs are extremely limited modeling choices — they capture simple short-term dependencies and linear autocorrelations only.
* **Numerical stability:** Regression via direct inversion of `X^T X` can be unstable. Use `pinv` or ridge regularization.
* **Combining outputs:** The ensemble is heuristic; different reconciliation strategies will materially change forecasts.
* **Sampling vs argmax:** Sampling from Markov distributions produces non-deterministic forecasts; replacing sampling with `argmax` yields deterministic outputs (but may overfit the most-likely choice).

## Ideas for improvements / TODO (practical roadmap)

1. **Modeling:**

   * Add backoff or interpolated Markov models (fall back to lower-order state when higher order is unseen).
   * Add probabilistic magnitude models (heteroscedastic regression or Bayesian linear regression) to derive confidence intervals.
   * Add features: moving averages, volatility measures, lagged signs as binary features, day-of-week seasonality.

2. **Training & evaluation:**

   * Implement rolling cross-validation and log MAE/RMSE per model. Store per-model metadata (rmse, n_train, last_refit).
   * Automate retraining schedule with model versioning.

3. **Numerical stability & regularization:**

   * Replace direct inverse with `np.linalg.pinv` or solve via `np.linalg.lstsq`.
   * Add Ridge (`λ`) as a configurable option.

4. **User experience:**

   * Add EMAs and a small textual summary of recent volatility.
   * Offer a deterministic forecast mode for reproducibility.

5. **Engineering:**

   * Add unit tests for `build_markov_model`, `build_regression`, `counts_to_probabilities` and `plot_sequence`.
   * Add a Dockerfile and simple systemd unit as deployment options.

## License

This project is distributed under the MIT License — see `LICENSE`.

---