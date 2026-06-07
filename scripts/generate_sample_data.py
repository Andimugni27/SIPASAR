"""Generate dummy CSV mimic struktur Online Retail UCI untuk testing pipeline."""
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N_CUSTOMERS = 80
N_TRANSACTIONS = 600
START_DATE = datetime(2024, 1, 1)
END_DATE = datetime(2024, 12, 31)

PRODUCTS = [
    ("85123A", "WHITE HANGING HEART T-LIGHT HOLDER", 2.95),
    ("71053",  "WHITE METAL LANTERN", 3.39),
    ("84406B", "CREAM CUPID HEARTS COAT HANGER", 2.75),
    ("84029G", "KNITTED UNION FLAG HOT WATER BOTTLE", 3.39),
    ("84029E", "RED WOOLLY HOTTIE WHITE HEART", 3.39),
    ("22752",  "SET 7 BABUSHKA NESTING BOXES", 7.65),
    ("21730",  "GLASS STAR FROSTED T-LIGHT HOLDER", 4.25),
    ("22633",  "HAND WARMER UNION JACK", 1.85),
    ("22632",  "HAND WARMER RED POLKA DOT", 1.85),
    ("85099B", "JUMBO BAG RED RETROSPOT", 1.95),
    ("22423",  "REGENCY CAKESTAND 3 TIER", 12.75),
    ("23298",  "SPOTTY BUNTING", 4.95),
    ("22720",  "SET OF 3 CAKE TINS PANTRY DESIGN", 4.95),
    ("23203",  "JUMBO BAG DOILEY PATTERNS", 1.95),
    ("47566",  "PARTY BUNTING", 4.65),
]

COUNTRIES = ["United Kingdom"] * 8 + ["France", "Germany", "EIRE", "Spain"]


def generate():
    rows = []
    invoice_counter = 536365

    for _ in range(N_TRANSACTIONS):
        cust_id = int(RNG.integers(12000, 12000 + N_CUSTOMERS))
        invoice_no = str(invoice_counter)
        invoice_counter += 1

        days_offset = int(RNG.integers(0, (END_DATE - START_DATE).days))
        invoice_date = START_DATE + timedelta(days=days_offset,
                                              hours=int(RNG.integers(8, 20)),
                                              minutes=int(RNG.integers(0, 60)))
        country = str(RNG.choice(COUNTRIES))
        n_items = int(RNG.integers(1, 6))

        for _ in range(n_items):
            stock, desc, price = PRODUCTS[int(RNG.integers(0, len(PRODUCTS)))]
            qty = int(RNG.integers(1, 20))
            rows.append({
                "InvoiceNo": invoice_no,
                "StockCode": stock,
                "Description": desc,
                "Quantity": qty,
                "InvoiceDate": invoice_date.strftime("%Y-%m-%d %H:%M:%S"),
                "UnitPrice": price,
                "CustomerID": cust_id,
                "Country": country,
            })

    n_bad = 20
    for _ in range(n_bad):
        rows.append({
            "InvoiceNo": "C" + str(invoice_counter),
            "StockCode": PRODUCTS[0][0],
            "Description": PRODUCTS[0][1],
            "Quantity": -int(RNG.integers(1, 5)),
            "InvoiceDate": "2024-06-15 10:00:00",
            "UnitPrice": PRODUCTS[0][2],
            "CustomerID": int(RNG.integers(12000, 12000 + N_CUSTOMERS)),
            "Country": "United Kingdom",
        })
        invoice_counter += 1

    df = pd.DataFrame(rows)
    nan_idx = RNG.choice(df.index, size=10, replace=False)
    df.loc[nan_idx, "CustomerID"] = np.nan

    out = Path(__file__).parent.parent / "data" / "sample_data.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Generated {len(df)} rows -> {out}")


if __name__ == "__main__":
    generate()
