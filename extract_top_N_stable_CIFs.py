#!/usr/bin/env python
import argparse
import csv
import os

from pymatgen.core import Structure


def get_formula_from_cif(cif_dir: str, cif_id: str) -> str:
    """
    Given a CIF ID from CGCNN (e.g. 'OQMD_1020518__1'),
    load the corresponding CIF file and return its reduced formula.

    Adjust the filename pattern here if your CIF names differ.
    """
    # Try <id>.cif
    cif_path = os.path.join(cif_dir, f"{cif_id}.cif")
    if not os.path.exists(cif_path):
        # Fallback: strip trailing '__1' etc. if needed
        base_id = cif_id.split("__")[0]
        alt_path = os.path.join(cif_dir, f"{base_id}.cif")
        if os.path.exists(alt_path):
            cif_path = alt_path
        else:
            raise FileNotFoundError(
                f"Could not find CIF for ID '{cif_id}' in {cif_dir} "
                f"(tried '{cif_path}' and '{alt_path}')"
            )

    struct = Structure.from_file(cif_path)
    return struct.composition.reduced_formula


def main():
    parser = argparse.ArgumentParser(
        description="Extract top-N most stable formulas from CGCNN predictions."
    )
    parser.add_argument(
        "--results",
        required=True,
        help="Path to CGCNN test_results.csv (ID,target,predicted_energy).",
    )
    parser.add_argument(
        "--cif-dir",
        required=True,
        help="Directory containing CIF files (one per ID).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of most stable formulas to keep (default: 20).",
    )
    parser.add_argument(
        "--out",
        default="topN_stable_formulas.csv",
        help="Output CSV path (default: topN_stable_formulas.csv).",
    )

    args = parser.parse_args()

    # Map: formula -> (best_energy, example_id)
    best_by_formula = {}

    with open(args.results, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 3:
                continue
            cif_id, _target, energy_str = row[0], row[1], row[2]
            try:
                energy = float(energy_str)
            except ValueError:
                # Skip header or malformed lines
                continue

            try:
                formula = get_formula_from_cif(args.cif_dir, cif_id)
            except FileNotFoundError as e:
                print(f"Warning: {e}")
                continue

            # Keep the most stable (lowest) energy per formula
            if formula not in best_by_formula or energy < best_by_formula[formula][0]:
                best_by_formula[formula] = (energy, cif_id)

    # Sort formulas by best (most negative) formation energy
    sorted_entries = sorted(best_by_formula.items(), key=lambda kv: kv[1][0])

    top_n = min(args.top_n, len(sorted_entries))
    print(f"Found {len(sorted_entries)} unique formulas, writing top {top_n}.")

    with open(args.out, "w", newline="") as f_out:
        writer = csv.writer(f_out)
        # First column: formula, second column: predicted formation energy
        writer.writerow(["formula", "predicted_formation_energy"])
        for formula, (energy, _cid) in sorted_entries[:top_n]:
            writer.writerow([formula, energy])

    print(f"Wrote {top_n} most stable formulas to {args.out}")


if __name__ == "__main__":
    main()
