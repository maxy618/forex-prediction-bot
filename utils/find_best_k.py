import os
import csv
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from model_engine import find_knn_forecast  # type: ignore

DATASET_PATH = "../datasets/USDRUB.csv"
WINDOW_SIZE = 21
FORECAST_HORIZON = 1
VAL_SIZE = 3000
K_START = 1
K_END = 100
STEP = 1

def load_diffs(path):
    if not os.path.exists(path):
        print(f"Error: File {path} not found.")
        sys.exit(1)
    diffs = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                val = float(row["Difference"])
                diffs.append(val)
            except ValueError:
                continue
    return diffs

def calculate_metrics(actual, predicted):
    actual = np.array(actual)
    predicted = np.array(predicted)
    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    return mae, rmse

def run_optimization():
    print(f"Loading data from {DATASET_PATH}...")
    data = load_diffs(DATASET_PATH)
    total_len = len(data)
    if total_len < VAL_SIZE + WINDOW_SIZE + FORECAST_HORIZON:
        print("Error: Not enough data for the requested validation size.")
        return

    val_start_index = total_len - VAL_SIZE
    print(f"Total days: {total_len}")
    print(f"Validation starts at index: {val_start_index}")
    print(f"Testing k from {K_START} to {K_END}...")

    k_list = list(range(K_START, K_END + 1, STEP))
    mae_results = []
    rmse_results = []
    avg_results = []

    for k in k_list:
        errors_mae = []
        errors_rmse = []
        stride = 1
        for t in range(val_start_index, total_len - FORECAST_HORIZON, stride):
            history = data[:t]
            query_seq = data[t - WINDOW_SIZE : t]
            actual_future = data[t : t + FORECAST_HORIZON]
            predicted_future = find_knn_forecast(
                history_diffs=history,
                query_sequence=query_seq,
                k=k,
                horizon=FORECAST_HORIZON
            )
            mae, rmse = calculate_metrics(actual_future, predicted_future)
            errors_mae.append(mae)
            errors_rmse.append(rmse)

        avg_mae = np.mean(errors_mae)
        avg_rmse = np.mean(errors_rmse)
        avg_both = (avg_mae + avg_rmse) / 2.0

        mae_results.append(avg_mae)
        rmse_results.append(avg_rmse)
        avg_results.append(avg_both)

        print(f"k={k:02d} | MAE={avg_mae:.4f} | RMSE={avg_rmse:.4f} | AVG={(avg_both):.4f}")

    top5_mae_idx = np.argsort(mae_results)[:5]
    print("\nTop 5 k by MAE:")
    for i in top5_mae_idx:
        print(f"k={k_list[i]} | MAE={mae_results[i]:.4f}")

    top5_rmse_idx = np.argsort(rmse_results)[:5]
    print("\nTop 5 k by RMSE:")
    for i in top5_rmse_idx:
        print(f"k={k_list[i]} | RMSE={rmse_results[i]:.4f}")

    print("\nAVG (MAE+RMSE)/2 for all k:")
    for k_val, avg_val in zip(k_list, avg_results):
        print(f"k={k_val} | AVG={avg_val:.4f}")

if __name__ == "__main__":
    run_optimization()
