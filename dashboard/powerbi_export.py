"""
Exports clean, Power BI-ready data for stakeholder dashboards.
Demonstrates ability to communicate findings to non-technical audiences.
"""
import pandas as pd
from pathlib import Path

def export_segment_summary(segmented_path: str = "data/processed/segmented_customers.csv",
                            out_dir: str = "reports"):
    Path(out_dir).mkdir(exist_ok=True)
    df = pd.read_csv(segmented_path)

    # Summary table per segment
    summary = df.groupby(["segment","segment_label"]).agg(
        n_customers = ("customer_id",  "count"),
        avg_recency = ("recency",      "mean"),
        avg_frequency = ("frequency",  "mean"),
        avg_monetary  = ("monetary",   "mean"),
        avg_basket    = ("avg_basket_size", "mean"),
    ).reset_index().round(2)

    summary.to_csv(f"{out_dir}/segment_summary_powerbi.csv", index=False)
    print(f"Exported segment summary: {out_dir}/segment_summary_powerbi.csv")

    # RFM distribution
    rfm_dist = df[["customer_id","recency","frequency","monetary","RFM_total","segment_label"]]
    rfm_dist.to_csv(f"{out_dir}/rfm_distribution_powerbi.csv", index=False)
    print(f"Exported RFM distribution: {out_dir}/rfm_distribution_powerbi.csv")

if __name__ == "__main__":
    export_segment_summary()
