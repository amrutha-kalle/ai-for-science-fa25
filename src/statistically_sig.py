#!/usr/bin/env python3
import argparse
import csv
import math
import statistics
from typing import List, Tuple


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# ---------- Proportions: hit-rate test ----------
def two_proportion_z_test(h1: int, n1: int, h2: int, n2: int) -> Tuple[float, float, float, float, float]:
    if n1 == 0 or n2 == 0:
        raise ValueError("n1 and n2 must be > 0")
    p1 = h1 / n1
    p2 = h2 / n2
    p_pooled = (h1 + h2) / (n1 + n2)
    se = math.sqrt(p_pooled * (1 - p_pooled) * (1 / n1 + 1 / n2))
    if se == 0:
        raise ValueError("Standard error is zero; cannot compute z-statistic.")
    z = (p1 - p2) / se
    p_value = 2 * (1 - normal_cdf(abs(z)))
    return p1, p2, z, p_value, p_pooled


# ---------- Mean (all samples): normal approx ----------
def mean_diff_z_test(x1: List[float], x2: List[float]) -> Tuple[float, float, float]:
    """
    Difference in means using normal approximation (Welch-style SE).
    Returns mean1, mean2, p_value
    """
    n1 = len(x1)
    n2 = len(x2)
    if n1 < 2 or n2 < 2:
        raise ValueError("Need at least 2 samples in each group to compare means.")
    m1 = statistics.mean(x1)
    m2 = statistics.mean(x2)
    # use sample std (unbiased) for SE
    s1 = statistics.stdev(x1)
    s2 = statistics.stdev(x2)
    se = math.sqrt((s1 ** 2) / n1 + (s2 ** 2) / n2)
    if se == 0:
        raise ValueError("Standard error is zero; cannot compute mean difference test.")
    z = (m1 - m2) / se
    p_value = 2 * (1 - normal_cdf(abs(z)))
    return m1, m2, p_value


# ---------- Median (all samples): Mann–Whitney U ----------
def mann_whitney_u_test(x1: List[float], x2: List[float]) -> Tuple[float, float, float, float]:
    """
    Mann–Whitney U with normal approximation on ranks.
    Returns median1, median2, U, p_value
    """
    n1, n2 = len(x1), len(x2)
    if n1 == 0 or n2 == 0:
        raise ValueError("Need at least one sample in each group for Mann–Whitney.")
    median1 = statistics.median(x1)
    median2 = statistics.median(x2)

    combined = [(v, 0) for v in x1] + [(v, 1) for v in x2]
    combined.sort(key=lambda t: t[0])

    ranks = [0.0] * (n1 + n2)
    i = 0
    while i < len(combined):
        j = i + 1
        while j < len(combined) and combined[j][0] == combined[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0  # 1-based ranks
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j

    R1 = 0.0
    for idx, (_, label) in enumerate(combined):
        if label == 0:
            R1 += ranks[idx]

    U1 = R1 - n1 * (n1 + 1) / 2.0
    U2 = n1 * n2 - U1
    U = min(U1, U2)

    mean_U = n1 * n2 / 2.0
    var_U = n1 * n2 * (n1 + n2 + 1) / 12.0
    if var_U == 0:
        raise ValueError("Variance of U is zero; cannot compute p-value.")
    z = (U - mean_U) / math.sqrt(var_U)
    p_value = 2 * (1 - normal_cdf(abs(z)))
    return median1, median2, U, p_value


# ---------- CSV loading ----------
def load_values_and_hits(csv_path: str, energy_column: str, threshold: float):
    """
    Load ALL energies + count hits under threshold.
    Returns:
        values_all (List[float]), n_total, n_hits
    """
    values_all: List[float] = []
    n_total = 0
    n_hits = 0
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        if energy_column not in reader.fieldnames:
            raise SystemExit(
                f"[error] energy column '{energy_column}' not found in {csv_path}. "
                f"Available columns: {reader.fieldnames}"
            )
        for row in reader:
            try:
                e = float(row[energy_column])
            except Exception:
                continue
            values_all.append(e)
            n_total += 1
            if e < threshold:
                n_hits += 1
    return values_all, n_total, n_hits


# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(
        description=(
            "Compare two CSVs (e.g., text vs cif-family): "
            "hit rate (proportion), overall mean, and overall median."
        )
    )
    ap.add_argument("--text_csv", required=True, help="CSV for text-only model.")
    ap.add_argument("--cif_csv", required=True, help="CSV for cif-family model (cif + text+cif).")
    ap.add_argument(
        "--energy-column",
        default="predicted_formation_energy_eV_per_atom",
        help="Column name for energy (default: predicted_formation_energy_eV_per_atom).",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Hit threshold for the hit-rate test (default: 0.0, e.g. E < 0).",
    )
    ap.add_argument("--label-text", default="text", help="Label for text CSV (default: 'text').")
    ap.add_argument("--label-cif", default="cif_family", help="Label for cif CSV (default: 'cif_family').")
    args = ap.parse_args()

    vals_text, n_text, h_text = load_values_and_hits(args.text_csv, args.energy_column, args.threshold)
    vals_cif, n_cif, h_cif = load_values_and_hits(args.cif_csv, args.energy_column, args.threshold)

    if n_text == 0 or n_cif == 0:
        raise SystemExit(f"[error] One of the CSVs has zero usable rows. text: n={n_text}, cif_family: n={n_cif}")

    # --- Hit-rate comparison (unchanged) ---
    p_cif, p_text, z_hits, p_hits, p_pooled = two_proportion_z_test(h_cif, n_cif, h_text, n_text)

    print("=== Hit-rate comparison (E < threshold) ===")
    print(f"Energy column: {args.energy_column}")
    print(f"Hit definition: {args.energy_column} < {args.threshold}\n")

    print(f"{args.label_cif} (cif CSV: {args.cif_csv}):")
    print(f"  n_total = {n_cif}")
    print(f"  n_hits  = {h_cif}")
    print(f"  hit rate = {p_cif*100:.2f}%\n")

    print(f"{args.label_text} (text CSV: {args.text_csv}):")
    print(f"  n_total = {n_text}")
    print(f"  n_hits  = {h_text}")
    print(f"  hit rate = {p_text*100:.2f}%\n")

    print(f"Pooled hit rate = {p_pooled*100:.2f}%")
    print(f"z-statistic (hit rate) = {z_hits:.3f}")
    print(f"two-sided p-value (hit rate) = {p_hits:.5f}")
    print("=> " + ("Significant" if p_hits < 0.05 else "Not significant") + " at α = 0.05.\n")

    # --- Overall mean comparison (ALL rows, not just hits) ---
    print("=== Overall mean comparison (all samples) ===")
    try:
        mean_cif, mean_text, p_mean = mean_diff_z_test(vals_cif, vals_text)
        print(f"{args.label_cif} mean = {mean_cif:.4f}")
        print(f"{args.label_text} mean = {mean_text:.4f}")
        print(f"two-sided p-value (mean difference) = {p_mean:.5f}")
        print("=> " + ("Significant" if p_mean < 0.05 else "Not significant") + " at α = 0.05.\n")
    except ValueError as e:
        print(f"[warn] Could not compare means: {e}\n")

    # --- Overall median comparison (ALL rows, not just hits) ---
    print("=== Overall median comparison (all samples, Mann–Whitney) ===")
    try:
        median_cif, median_text, U, p_med = mann_whitney_u_test(vals_cif, vals_text)
        print(f"{args.label_cif} median = {median_cif:.4f}")
        print(f"{args.label_text} median = {median_text:.4f}")
        print(f"U statistic = {U:.3f}")
        print(f"two-sided p-value (Mann–Whitney) = {p_med:.5f}")
        print("=> " + ("Significant" if p_med < 0.05 else "Not significant") + " at α = 0.05.")
    except ValueError as e:
        print(f"[warn] Could not compare medians: {e}")


if __name__ == "__main__":
    main()
