import pandas as pd


def top_products_per_segment(
    df: pd.DataFrame,
    rfm_labeled: pd.DataFrame,
    top_n: int = 5,
    segment_col: str = "Segment",
) -> pd.DataFrame:
    merged = df.merge(
        rfm_labeled[["CustomerID", segment_col]], on="CustomerID", how="inner"
    )

    grouped = (
        merged.groupby([segment_col, "StockCode", "Description"])
        .agg(
            TotalQuantity=("Quantity", "sum"),
            TotalRevenue=("TotalPrice", "sum"),
            OrderCount=("InvoiceNo", "nunique"),
        )
        .reset_index()
    )

    grouped["Rank"] = grouped.groupby(segment_col)["TotalRevenue"].rank(
        method="dense", ascending=False
    )
    top = grouped[grouped["Rank"] <= top_n].sort_values(
        [segment_col, "Rank"]
    ).reset_index(drop=True)

    return top
