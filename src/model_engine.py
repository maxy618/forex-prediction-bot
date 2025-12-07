import os
import pickle
import random
from collections import defaultdict, Counter
import numpy as np


DEFAULT_MARKOV_K = 0.2


def save_model(model, path_to_model):
    os.makedirs(os.path.dirname(path_to_model) or '.', exist_ok=True)
    with open(path_to_model, 'wb') as file:
        pickle.dump(model, file)


def load_model(path_to_model):
    if not os.path.exists(path_to_model):
        raise FileNotFoundError(f"{path_to_model} does not exist")
    with open(path_to_model, 'rb') as file:
        model = pickle.load(file)
        return model


def counts_to_probabilities(counter, k=DEFAULT_MARKOV_K):
    """Convert a Counter of next-state counts into a probability mapping using add-k smoothing."""
    laplace_smoothing = lambda m, n, k, v: (m + k) / (n + k * v)
    n = sum(counter.values())
    v = len(counter)

    probs = {}
    for key, count in counter.items():
        p = laplace_smoothing(count, n, k, v)
        probs[key] = p

    return probs


def build_markov_model(sequence, order=1, k=DEFAULT_MARKOV_K):
    if order >= len(sequence):
        raise ValueError("sequence length should be > order")

    counters = defaultdict(Counter)

    for idx in range(len(sequence) - order):
        current_state = tuple(sequence[idx:idx+order])
        next_state = sequence[idx+order]
        counters[current_state][next_state] += 1

    model = {}
    for state, counter in counters.items():
        model[state] = counts_to_probabilities(counter, k=k)

    return {
        "order": order,
        "table": model
    }


def predict_state(sequence, model):
    order = model["order"]
    table = model["table"]

    if len(sequence) < order:
        raise ValueError("sequence length should be >= order")

    current_state = tuple(sequence[-order:])
    next_states = table.get(current_state)
    if next_states is None:
        return None

    rand_num = random.random()
    cumul_sum = 0
    for sign, prob in next_states.items():
        cumul_sum += prob
        if rand_num <= cumul_sum:
            return sign

    # fallback to argmax
    return max(next_states, key=next_states.get)


def predict_ensemble_sign(sequence, models):
    votes = {"+": 0, "-": 0}
    for model in models:
        next_state = predict_state(sequence, model)
        if next_state is None:
            continue
        votes[next_state] += 1

    max_votes = max(votes.values())
    leaders = [k for k, v in votes.items() if v == max_votes]
    return random.choice(leaders)


def forecast_signs(sequence, models, n=1):
    result = []
    curr_seq = sequence.copy()
    for _ in range(n):
        next_state = predict_ensemble_sign(curr_seq, models)
        result.append(next_state)
        curr_seq.append(next_state)
    return result


def build_regression(diffs_list, n_lags=1):
    if len(diffs_list) <= n_lags:
        raise ValueError(f"diffs_list length should be > n_lags")

    x = []
    y = []
    for i in range(n_lags, len(diffs_list)):
        prev_diffs = diffs_list[i - n_lags:i]
        curr_diff  = diffs_list[i]
        x.append(prev_diffs)
        y.append(curr_diff)

    X_mat = np.array(x, dtype=float)
    y_vec = np.array(y, dtype=float).reshape(-1, 1)

    ones = np.ones((X_mat.shape[0], 1))
    X_with_bias = np.hstack([X_mat, ones])

    Xt = X_with_bias.T
    XtX = Xt @ X_with_bias
    Xty = Xt @ y_vec

    coeffs = np.linalg.inv(XtX) @ Xty

    return coeffs.flatten()


def predict_diff(last_diffs, coeffs):
    n_lags = len(coeffs) - 1
    if len(last_diffs) < n_lags:
        raise ValueError("last_diffs length is smaller than required n_lags")

    weights = coeffs[:-1]
    bias = coeffs[-1]
    inputs = last_diffs[-n_lags:]

    prediction = float(np.dot(weights, inputs) + bias)
    return prediction


def predict_ensemble_diff(last_diffs, models):
    results = [predict_diff(last_diffs, coeffs) for coeffs in models]
    return sum(results) / len(results)


def forecast_diffs(last_diffs, models, n=1):
    result = []
    diffs = last_diffs.copy()
    for _ in range(n):
        next_diff = predict_ensemble_diff(diffs, models)
        result.append(next_diff)
        diffs.append(next_diff)
    return result
