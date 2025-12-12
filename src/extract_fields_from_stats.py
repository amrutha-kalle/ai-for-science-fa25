#!/usr/bin/env python
import csv
import argparse

def main():
    parser = argparse.ArgumentParser(
        description="Extract basic props for top stable formulas from CrystaLLM stats."
    )
    parser.add_argument("--stats", required=True,
                        help="Input stats CSV (evaluate_cifs output).")
    parser.add_argument("--top", required=True,
                        help="CSV with top stable formulas (must have column 'formula_structural').")
    parser.add_argument("--out", default="top20_props.csv",
                        help="Output CSV file (default: top20_props.csv).")
    args = parser.parse_args()

    # 1) Load set of top formulas from top20_stable.csv
    top_formulas = set()
    with open(args.top, newline="") as f_top:
        reader = csv.DictReader(f_top)
        for row in reader:
            # normalize a bit: strip spaces
            formula = row["formula_structural"].replace(" ", "")
            top_formulas.add(formula)

    # 2) Read stats and keep only rows whose formula_structural is in top_formulas
    with open(args.stats, newline="") as f_in, open(args.out, "w", newline="") as f_out:
        reader = csv.DictReader(f_in)
        writer = csv.writer(f_out)

        # header
        writer.writerow(["formula_structural", "space_group", "cell_volume", "density"])

        for row in reader:
            formula_structural = row["formula_structural"]
            formula_norm = formula_structural.replace(" ", "")

            if formula_norm not in top_formulas:
                continue

            space_group = row["space_group"]
            cell_volume = row["cell_volume"]
            density = row["density"]

            writer.writerow([formula_structural, space_group, cell_volume, density])

    print(f"Wrote filtered props for top formulas to {args.out}")

if __name__ == "__main__":
    main()
