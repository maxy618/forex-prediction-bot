import os
import time
from datetime import date, timedelta

import requests
from requests.adapters import HTTPAdapter, Retry


HTTP_POOL_SIZE = int(os.getenv("HTTP_POOL_SIZE", 10))

SESSION = requests.Session()
retry_strategy = Retry(
    total=3,
    backoff_factor=0.3,
    status_forcelist=(429, 500, 502, 503, 504),
    allowed_methods=("GET",)
)
adapter = HTTPAdapter(pool_connections=HTTP_POOL_SIZE, pool_maxsize=HTTP_POOL_SIZE, max_retries=retry_strategy)
SESSION.mount("https://", adapter)
SESSION.mount("http://", adapter)
SESSION.headers.update({"User-Agent": "Mozilla/5.0"})

BASE_LATEST = "https://www.cbr-xml-daily.ru/daily_json.js"
BASE_ARCHIVE = "https://www.cbr-xml-daily.ru/archive"


def get_dates_list(num_days):
    today = date.today()
    start = today - timedelta(days=num_days-1)

    day = start
    res = []
    while day <= today:
        res.append(day)
        day += timedelta(days=1)
    return res


def fetch_for_date(d, base_latest=BASE_LATEST, base_archive=BASE_ARCHIVE):
    if d == date.today():
        url = base_latest
    else:
        url = f"{base_archive}/{d.year}/{d.month:02d}/{d.day:02d}/daily_json.js"
    try:
        r = SESSION.get(url, timeout=6)
        if r.status_code == 200:
            try:
                return r.json()
            except ValueError:
                pass
    except Exception:
        pass

    # retry once
    try:
        time.sleep(0.2)
        r = SESSION.get(url, timeout=6)
        if r.status_code == 200:
            try:
                return r.json()
            except ValueError:
                pass
    except Exception:
        pass

    # fallback
    if url != base_latest:
        try:
            r = SESSION.get(base_latest, timeout=6)
            if r.status_code == 200:
                try:
                    return r.json()
                except ValueError:
                    pass
        except Exception:
            pass
    return None


def rub_per_one_from_json(js, code):
    if code == "RUB":
        return 1.0
    if not js or "Valute" not in js:
        return None
    info = js["Valute"].get(code)
    if not info:
        return None
    nominal = int(info.get("Nominal", 1))
    value = float(info.get("Value"))
    return value / nominal


def fetch_sequences_all_pairs(currencies, days=10):
    dates = get_dates_list(days)
    last_known = {c: None for c in currencies}

    res = {}
    for d in dates:
        js = fetch_for_date(d)
        if js is None:
            time.sleep(0.2)
            js = fetch_for_date(d)

        if js is not None:
            for c in currencies:
                val = rub_per_one_from_json(js, c)
                if val is not None:
                    last_known[c] = val
        else:
            raise RuntimeError("fetched empty json")

        for c in currencies:
            if last_known[c] is None:
                raise RuntimeError(f"no data for {c} on / before {d.isoformat()}")

        day_rates = {}
        for base in currencies:
            for target in currencies:
                pair = f"{target}_per_{base}"
                if base == target:
                    rate = 1.0
                else:
                    B = last_known[base]
                    T = last_known[target]
                    rate = B / T
                day_rates[pair] = float(rate)
        res[d.isoformat()] = day_rates

    return res
