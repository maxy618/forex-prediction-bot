from decimal import Decimal, getcontext, InvalidOperation
import csv
from datetime import datetime
from itertools import permutations


INPUT = "../datasets/history.csv"


base_currencies = ["USD", "EUR", "JPY", "GBP", "CHF", "RUB"]
pairs = [a + b for a, b in permutations(base_currencies, 2)]


getcontext().prec = 40


PRICE_PREC = 6
DIFF_PREC  = 6


quant_price = Decimal('1').scaleb(-PRICE_PREC)
quant_diff  = Decimal('1').scaleb(-DIFF_PREC)


with open(INPUT, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    header = reader.fieldnames
    rows = [row for row in reader]


valid_entries = []
for r in rows:
    date_str = (r.get("Date") or "").strip()
    if not date_str:
        continue
    try:
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    except Exception:
        date_obj = None
    valid_entries.append((date_obj, date_str, r))


valid_entries.sort(key=lambda x: (x[0] is None, x[0] or datetime.min))


for pair in pairs:
    A = pair[:3]
    B = pair[3:]
    out_rows = []

    for date_obj, date_str, r in valid_entries:
        sA = "1" if A == "EUR" else (r.get(A) or "").strip()
        sB = "1" if B == "EUR" else (r.get(B) or "").strip()

        if not sA or not sB:
            continue
        if sA.upper() == "N/A" or sB.upper() == "N/A":
            continue

        try:
            decA = Decimal(sA.replace(",", "").strip())
            decB = Decimal(sB.replace(",", "").strip())
        except (InvalidOperation, ValueError):
            continue

        if decA == 0:
            continue

        price = decB / decA
        out_rows.append((date_str, price))

    if len(out_rows) < 2:
        print(f"Pair {pair}: not enough data to generate file (found {len(out_rows)} rows).")
        continue

    out_filename = f"../datasets/{pair}.csv"
    with open(out_filename, "w", newline='', encoding='utf-8') as outf:
        writer = csv.writer(outf)
        writer.writerow(["Date", "Price", "Sign", "Difference"])
        for i in range(1, len(out_rows)):
            date_cur, price_cur = out_rows[i]
            _, price_prev = out_rows[i - 1]
            sign = "+" if price_cur > price_prev else "-"
            diff = abs(price_cur - price_prev)

            price_fmt = format(price_cur.quantize(quant_price), 'f')
            diff_fmt  = format(diff.quantize(quant_diff), 'f')

            writer.writerow([date_cur, price_fmt, sign, diff_fmt])

    print(f"Created {out_filename} ({len(out_rows)-1} rows).")
