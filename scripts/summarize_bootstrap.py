"""Report paired HGB-minus-RF differences from the archived MMSI resamples."""
from pathlib import Path

import pandas as pd


def main():
    path = (Path(__file__).resolve().parents[1] / "results" / "models_v1"
            / "mmsi_cluster_bootstrap_distribution.csv")
    draws = pd.read_csv(path)
    for name, group in draws.groupby("Evaluation_Set", sort=False):
        if len(group) != 1000 or group.Bootstrap_Index.nunique() != 1000:
            raise ValueError(f"Incomplete or duplicate bootstrap draws in {name}")
        print(name)
        for metric in ("Delta_F1_HGB_minus_RF", "Delta_AUPRC_HGB_minus_RF"):
            values = group[metric]
            lower, upper = values.quantile([.025, .975])
            print(f"  {metric}: {values.mean():.4f} [{lower:.4f}, {upper:.4f}]")


if __name__ == "__main__":
    main()

