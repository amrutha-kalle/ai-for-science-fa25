#!/usr/bin/env python
import json
import csv
import sys
import re
from collections import Counter
from typing import Dict, Any, List


# ---------- composition utilities ----------

def parse_composition(formula: str) -> Counter:
    """
    Parse a composition like 'Cr1 Ho1 O6 Te1' or 'Ho2O8Re1Se1Te2'
    into a Counter({element: count}).
    """
    if not formula:
        return Counter()

    s = formula.replace(" ", "")
    pattern = r"([A-Z][a-z]?)(\d*\.?\d*)"
    comp = Counter()
    for elem, count_str in re.findall(pattern, s):
        if count_str in ("", "."):
            count = 1.0
        else:
            try:
                count = float(count_str)
            except ValueError:
                count = 1.0
        comp[elem] += count
    return comp


def normalize_comp(c: Counter) -> Counter:
    total = sum(c.values())
    if total == 0:
        return Counter(c)
    return Counter({el: v / total for el, v in c.items()})


def l1_fractional_distance(target: Counter, sample: Counter) -> float:
    """
    L1 distance between *fractional* compositions.
    0.0 = perfect match, larger = more different.
    """
    t = normalize_comp(target)
    s = normalize_comp(sample)
    elems = set(t) | set(s)
    return sum(abs(t.get(e, 0.0) - s.get(e, 0.0)) for e in elems)


def exact_match_up_to_scale(target: Counter, sample: Counter, tol: float = 1e-6) -> bool:
    """
    True if sample is a scaled version of target:
      sample[element] ≈ k * target[element] for all elements,
    and the set of elements matches exactly.
    """
    if not target or not sample:
        return False

    if set(target.keys()) != set(sample.keys()):
        return False

    scale = None
    for e, v in target.items():
        if abs(v) > tol:
            sample_v = sample.get(e, 0.0)
            scale = sample_v / v
            break

    if scale is None:
        return False

    for e, v in target.items():
        if abs(sample.get(e, 0.0) - scale * v) > tol:
            return False

    return True


# ---------- main processing ----------

def process_comparisons(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    data: top-level JSON dict that contains a 'comparisons' list.
    Returns a list of rows for CSV.
    """
    if "comparisons" not in data or not isinstance(data["comparisons"], list):
        print("JSON does not contain a 'comparisons' list – nothing to do.", file=sys.stderr)
        return []

    rows: List[Dict[str, Any]] = []

    for entry in data["comparisons"]:
        if not isinstance(entry, dict):
            continue

        cif_name = entry.get("cif_name", "")
        target_comp_str = entry.get("composition", "").strip()
        prompts = entry.get("prompts", {})
        info_samples = entry.get("info_samples", {})

        target_comp = parse_composition(target_comp_str)
        num_target_elems = len(target_comp)

        for mode in ("text", "cif", "text+cif"):
            mode_samples = info_samples.get(mode, [])
            if not isinstance(mode_samples, list):
                continue

            prompt_for_mode = ""
            if isinstance(prompts, dict):
                prompt_for_mode = prompts.get(mode, "")

            for idx, sample_info in enumerate(mode_samples):
                if not isinstance(sample_info, dict):
                    continue

                sample_comp_str = sample_info.get("composition", "").strip()
                sample_comp = parse_composition(sample_comp_str)
                num_sample_elems = len(sample_comp)

                n_atoms_sample = sample_info.get("n_atoms", None)

                element_set_match = int(set(target_comp.keys()) == set(sample_comp.keys()))
                exact_match = int(exact_match_up_to_scale(target_comp, sample_comp))
                l1_dist = l1_fractional_distance(target_comp, sample_comp)

                row = {
                    "cif_name": cif_name,
                    "mode": mode,
                    "sample_idx": idx,

                    # original query info
                    "prompt_composition": target_comp_str,
                    "prompt_text_for_mode": prompt_for_mode,

                    # generated info
                    "sample_composition": sample_comp_str,
                    "element_set_match": element_set_match,
                    "exact_match_up_to_scale": exact_match,
                    "l1_distance_fraction": l1_dist,
                    "num_prompt_elements": num_target_elems,
                    "num_sample_elements": num_sample_elems,
                    "n_atoms_sample": n_atoms_sample,
                }
                rows.append(row)

    return rows


def main():
    if len(sys.argv) != 3:
        print("Usage: python evaluate_stoichiometry.py <input_json> <output_csv>", file=sys.stderr)
        sys.exit(1)

    json_path = sys.argv[1]
    csv_path = sys.argv[2]

    with open(json_path, "r") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        print("Top-level JSON is not a dict – expected the structure with 'comparisons'.", file=sys.stderr)
        sys.exit(1)

    rows = process_comparisons(data)
    print(f"Extracted {len(rows)} sample rows")

    if not rows:
        print("No rows to write. Exiting.")
        sys.exit(0)

    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote CSV to {csv_path}")


if __name__ == "__main__":
    main()
