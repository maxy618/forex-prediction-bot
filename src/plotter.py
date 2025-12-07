import os
from datetime import date, timedelta
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def make_axes_limits(prices):
    min_price = min(prices)
    max_price = max(prices)
    diff = max_price - min_price

    if diff == 0:
        diff = min_price * 0.01 if min_price != 0 else 1

    low = min_price - diff * 0.5
    high = max_price + diff * 0.5
    return low, high


def plot_sequence(old_prices, new_prices, filename):
    all_prices = old_prices + new_prices
    if not all_prices:
        raise ValueError("no prices to plot")

    y_min, y_max = make_axes_limits(all_prices)

    color_new = "green" if new_prices and new_prices[-1] > old_prices[-1] else "red"

    m = len(old_prices)
    n = len(new_prices)
    total = m + n

    start_date = date.today() - timedelta(days=max(0, m - 1))
    dates = [start_date + timedelta(days=i) for i in range(total)]
    labels = [d.strftime("%d.%m") for d in dates]

    plt.figure(figsize=(5, 3), facecolor="black")
    ax = plt.gca()
    ax.set_facecolor("black")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    ax.tick_params(colors="white", labelsize=8)
    plt.ylim(y_min, y_max)

    old_x = list(range(m))
    plt.plot(old_x, old_prices, color="white", linewidth=2)

    if n > 0:
        new_x = list(range(m - 1, total))
        new_y = [old_prices[-1]] + new_prices
        plt.plot(new_x, new_y, color=color_new, linewidth=2)

    plt.xticks(ticks=list(range(total)), labels=labels, rotation=45, fontsize=7)
    plt.tight_layout()

    out_dir = os.path.dirname(filename) or "."
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(filename, bbox_inches="tight", facecolor="black")
    plt.close()

    return filename
