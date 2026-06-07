import pandas as pd

REQUIRED_COLUMNS = [
    "InvoiceNo",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "CustomerID",
    "Country",
]


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom hilang: {missing}")

    df = df.copy()
    df = df.dropna(subset=["CustomerID"])
    df = df.drop_duplicates()
    df = df[df["Quantity"] > 0]
    df = df[df["UnitPrice"] > 0]
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df = df.dropna(subset=["InvoiceDate"])

    df["CustomerID"] = df["CustomerID"].astype(int).astype(str)
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

    return df.reset_index(drop=True)
