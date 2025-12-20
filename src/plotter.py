import os
from datetime import date, timedelta
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import io


DPI = 100
FIGSIZE = (4, 4)
RESIZE_TO = (640, 640)
TRANSITION_STEPS = 10
FRAME_MS = 400
FINAL_HOLD_MS = 3000
TRAIL_SAMPLES = 6
TRAIL_SHIFT_PX = 8
TRAIL_ALPHA = 0.30


def make_axes_limits(prices):
    min_price = min(prices)
    max_price = max(prices)
    diff = max_price - min_price
    if diff == 0:
        diff = min_price * 0.01 if min_price != 0 else 1
    low = min_price - diff * 0.5
    high = max_price + diff * 0.5
    return low, high


def render_plot_image(old_prices, new_prices):
    buf = io.BytesIO()
    all_prices = old_prices + new_prices
    if not all_prices:
        raise ValueError("no prices to plot")

    y_min, y_max = make_axes_limits(all_prices)
    color_new = "green" if new_prices and new_prices[-1] > old_prices[-1] else "red"

    m, n = len(old_prices), len(new_prices)
    total = m + n
    start_date = date.today() - timedelta(days=max(0, m - 1))
    dates = [start_date + timedelta(days=i) for i in range(total)]
    labels = [d.strftime("%d.%m") for d in dates]

    plt.figure(figsize=FIGSIZE, facecolor="black")
    ax = plt.gca()
    ax.set_facecolor("black")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors="white", labelsize=8)
    plt.ylim(y_min, y_max)

    plt.plot(list(range(m)), old_prices, color="white", linewidth=2)
    if n > 0:
        plt.plot(list(range(m-1, total)), [old_prices[-1]] + new_prices, color=color_new, linewidth=2)

    plt.xticks(ticks=list(range(total)), labels=labels, rotation=45, fontsize=7)
    plt.tight_layout()
    plt.savefig(buf, bbox_inches="tight", facecolor="black", dpi=DPI)
    plt.close()

    buf.seek(0)
    return Image.open(buf).convert("RGBA").resize(RESIZE_TO, Image.Resampling.LANCZOS) if RESIZE_TO else Image.open(buf).convert("RGBA")


def plot_sequence(old_prices, new_prices, filename):
    img = render_plot_image(old_prices, new_prices)

    out_dir = os.path.dirname(filename) or "."
    os.makedirs(out_dir, exist_ok=True)

    img.convert("RGB").save(filename, format="PNG", dpi=(DPI, DPI))

    return filename


def add_motion_trail(base_img, forward_img, progress):
    w, h = base_img.size
    trail = Image.new("RGBA", (w, h), (0,0,0,0))
    for i in range(TRAIL_SAMPLES):
        t = i / (TRAIL_SAMPLES - 1) if TRAIL_SAMPLES > 1 else 0
        dx = int(round(t * TRAIL_SHIFT_PX * progress))
        alpha = int(round(255 * TRAIL_ALPHA * (1-t) * progress))
        if alpha <= 0:
            continue
        tmp = forward_img.copy()
        a = tmp.split()[3].point(lambda p, scale=alpha/255: int(p*scale))
        tmp.putalpha(a)
        trail.paste(tmp, (dx,0), tmp)
    return Image.alpha_composite(base_img, trail)


def build_transition_frames(img_a, img_b):
    frames = []
    for i in range(1, TRANSITION_STEPS+1):
        alpha = i / (TRANSITION_STEPS + 1)
        blended = Image.blend(img_a, img_b, alpha).convert("RGBA")
        frames.append(add_motion_trail(blended, img_b, alpha))
    return frames


def make_forecast_gif(old_prices, new_prices, gif_path):
    frames_main = [render_plot_image(old_prices, new_prices[:k]) for k in range(len(new_prices)+1)]
    seq_frames = []
    for idx in range(len(frames_main)):
        seq_frames.append(frames_main[idx])
        if idx < len(frames_main) - 1:
            seq_frames.extend(build_transition_frames(frames_main[idx], frames_main[idx+1]))

    durations = [max(10, int(FRAME_MS/(TRANSITION_STEPS+1)))]*(len(seq_frames)-1) + [FINAL_HOLD_MS]

    out_dir = os.path.dirname(gif_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    seq_frames[0].save(
        gif_path,
        save_all=True,
        append_images=seq_frames[1:],
        duration=durations,
        loop=0,
        format="GIF"
    )
    return gif_path

