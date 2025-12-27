import os
import pickle
import numpy as np


def cache_data(model, path_to_model):
    os.makedirs(os.path.dirname(path_to_model) or '.', exist_ok=True)
    with open(path_to_model, 'wb') as file:
        pickle.dump(model, file)


def load_cached_data(path_to_model):
    if not os.path.exists(path_to_model):
        raise FileNotFoundError(f"{path_to_model} does not exist")
    with open(path_to_model, 'rb') as file:
        model = pickle.load(file)
        return model


def find_knn_forecast(history_diffs, query_sequence, k=5, horizon=1):
    history = np.array(history_diffs, dtype=float)
    query = np.array(query_sequence, dtype=float)
    
    seq_len = len(query)
    data_len = len(history)
    
    if data_len < seq_len + horizon:
        return [0.0] * horizon

    distances = []
    
    limit = data_len - seq_len - horizon + 1
    
    for i in range(limit):
        window = history[i : i + seq_len]
        dist = np.linalg.norm(window - query)
        distances.append((dist, i))
    
    distances.sort(key=lambda x: x[0])
    best_neighbors = distances[:k]
    
    predictions = []
    for _, idx in best_neighbors:
        future_start = idx + seq_len
        future_end = future_start + horizon
        pred_seq = history[future_start : future_end]
        predictions.append(pred_seq)
    
    if not predictions:
        return [0.0] * horizon
        
    predictions_mat = np.array(predictions)
    avg_prediction = np.mean(predictions_mat, axis=0)
    
    return avg_prediction.tolist()