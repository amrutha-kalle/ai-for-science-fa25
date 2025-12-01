#!/usr/bin/env python
import sys
import pandas as pd


METRICS = [
    "exact_match_up_to_scale",
    "element_set_match",
    "l1_distance_fraction",
]


def load_csv(path: str, model_label: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Ensure numeric
    for col in METRICS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["model"] = model_label
    return df


def summarize_group(df: pd.DataFrame, group_cols):
    """
    Compute summary statistics (counts, match rates, mean/median L1)
    grouped by group_cols (e.g., ['model', 'mode'] or just ['model']).
    """
    if not isinstance(group_cols, list):
        group_cols = [group_cols]

    grouped = df.groupby(group_cols)

    rows = []
    for keys, g in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)

        row = {}
        # Add group identifiers
        for col, key in zip(group_cols, keys):
            row[col] = key

        row["n_samples"] = len(g)

        # Exact match rate (0/1 indicator)
        if "exact_match_up_to_scale" in g.columns:
            row["exact_match_rate"] = g["exact_match_up_to_scale"].mean()

        # Element set match rate
        if "element_set_match" in g.columns:
            row["element_set_match_rate"] = g["element_set_match"].mean()

        # L1 distance stats
        if "l1_distance_fraction" in g.columns:
            row["l1_mean"] = g["l1_distance_fraction"].mean()
            row["l1_median"] = g["l1_distance_fraction"].median()

        rows.append(row)

    return pd.DataFrame(rows)


def print_comparison(table: pd.DataFrame, title: str, by_mode: bool = True):
    print("=" * 80)
    print(title)
    print("=" * 80)

    if by_mode and "mode" in table.columns:
        modes = sorted(table["mode"].unique())
        for mode in modes:
            sub = table[table["mode"] == mode].copy()
            print(f"\n--- Mode: {mode} ---")
            # Pivot so rows are metrics, columns are models
            pivot = sub.set_index("model")[[
                "n_samples",
                "exact_match_rate",
                "element_set_match_rate",
                "l1_mean",
                "l1_median",
            ]].T
            print(pivot.to_string(float_format=lambda x: f"{x:.4f}"))
    else:
        # Just overall per-model
        pivot = table.set_index("model")[[
            "n_samples",
            "exact_match_rate",
            "element_set_match_rate",
            "l1_mean",
            "l1_median",
        ]].T
        print(pivot.to_string(float_format=lambda x: f"{x:.4f}"))

    print("\n")


def main():
    if len(sys.argv) != 3:
        print("Usage: python compare_gep_vs_no_gep.py <gep_csv> <no_gep_csv>")
        sys.exit(1)

    gep_path = sys.argv[1]
    no_gep_path = sys.argv[2]

    # Load both runs
    df_gep = load_csv(gep_path, "GEP")
    df_no_gep = load_csv(no_gep_path, "No-GEP")

    df_all = pd.concat([df_gep, df_no_gep], ignore_index=True)

    # 1) Per-model, per-mode stats
    per_mode = summarize_group(df_all, ["model", "mode"])
    print_comparison(per_mode, "Per-mode Stoichiometry Metrics (GEP vs No-GEP)", by_mode=True)

    # 2) Overall per-model stats (all modes pooled)
    overall = summarize_group(df_all, ["model"])
    print_comparison(overall, "Overall Stoichiometry Metrics (All Modes Combined)", by_mode=False)

    # 3) Quick deltas (GEP - No-GEP) per mode for exact_match_rate and l1_mean
    if {"model", "mode", "exact_match_rate", "l1_mean"}.issubset(per_mode.columns):
        print("=" * 80)
        print("Deltas (GEP - No-GEP) by mode")
        print("=" * 80)

        # Reshape to model columns
        pivot_exact = per_mode.pivot(index="mode", columns="model", values="exact_match_rate")
        pivot_l1 = per_mode.pivot(index="mode", columns="model", values="l1_mean")

        if "GEP" in pivot_exact.columns and "No-GEP" in pivot_exact.columns:
            delta_exact = pivot_exact["GEP"] - pivot_exact["No-GEP"]
            delta_l1 = pivot_l1["GEP"] - pivot_l1["No-GEP"]

            print("\nExact match rate (GEP - No-GEP):")
            print(delta_exact.to_frame("delta_exact_match_rate").to_string(float_format=lambda x: f"{x:+.4f}"))

            print("\nMean L1 distance (GEP - No-GEP):")
            print(delta_l1.to_frame("delta_l1_mean").to_string(float_format=lambda x: f"{x:+.4f}"))

    print("\nDone.")


if __name__ == "__main__":
    main()
