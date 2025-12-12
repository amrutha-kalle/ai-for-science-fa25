#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
from pathlib import Path
from typing import Dict, List

from pymatgen.io.cif import CifParser
from pymatgen.core.composition import Composition


def alphabetical_formula(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    try:
        return Composition(s).alphabetical_formula.replace(" ", "")
    except Exception:
        return s.replace(" ", "")


def formula_from_cif(cif_path: Path) -> str:
    try:
        struct = CifParser(str(cif_path)).get_structures(primitive=False)[0]
        return struct.composition.alphabetical_formula.replace(" ", "")
    except Exception:
        return ""


def read_rows(results_csv: Path) -> List[Dict]:
    """
    Expect columns:
      id, base, prompt_tag, sample, predicted_formation_energy_eV_per_atom
    (formula optional).
    """
    with results_csv.open(newline="") as f:
        r = csv.DictReader(f)
        rows = [dict(row) for row in r]

    need = {"id", "base", "prompt_tag", "sample", "predicted_formation_energy_eV_per_atom"}
    missing = need - set(rows[0].keys() if rows else set())
    if missing:
        raise SystemExit(f"[error] CSV missing columns: {missing}")

    cleaned = []
    for row in rows:
        try:
            energy = float(row["predicted_formation_energy_eV_per_atom"])
        except Exception:
            continue
        cleaned.append({
            "id": row["id"].strip(),
            "base": row["base"].strip(),
            "prompt_tag": row["prompt_tag"].strip(),   # 'cif' | 'text' | 'text+cif'
            "sample": row["sample"].strip(),
            "energy": energy,
            "formula": alphabetical_formula(row.get("formula", "")),
        })
    return cleaned


def attach_formulas_if_missing(rows: List[Dict], cifdir: Path) -> None:
    hits = misses = 0
    for r in rows:
        if r["formula"]:
            continue
        cif_path = cifdir / f"{r['id']}.cif"
        if cif_path.exists():
            f = formula_from_cif(cif_path)
            if f:
                r["formula"] = f
                hits += 1
                continue
        misses += 1
    print(f"[formula] filled={hits}, still_missing={misses}")


def dedup_by_base(rows: List[Dict]) -> List[Dict]:
    """Keep one row per base — the most stable (min energy)."""
    best: Dict[str, Dict] = {}
    for r in rows:
        b = r["base"]
        if b not in best or r["energy"] < best[b]["energy"]:
            best[b] = r
    return list(best.values())


def unique_sorted(rows: List[Dict]) -> List[Dict]:
    """Unique by base (most stable) and sorted by energy ascending."""
    return sorted(dedup_by_base(rows), key=lambda r: r["energy"])


def write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["id", "base", "prompt_tag", "sample", "formula",
                  "predicted_formation_energy_eV_per_atom"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({
                "id": r["id"],
                "base": r["base"],
                "prompt_tag": r["prompt_tag"],
                "sample": r["sample"],
                "formula": r.get("formula", ""),
                "predicted_formation_energy_eV_per_atom": r["energy"],
            })


def stats(rows: List[Dict]) -> Dict[str, float]:
    n = len(rows)
    if n == 0:
        return {"count": 0, "hit_count": 0, "hit_rate": 0.0, "avg_energy": float("nan")}
    hit_count = sum(1 for r in rows if r["energy"] < 0.0)
    avg_energy = sum(r["energy"] for r in rows) / n
    hit_rate = hit_count / n
    return {
        "count": n,
        "hit_count": hit_count,
        "hit_rate": hit_rate,
        "avg_energy": avg_energy,
    }


def print_stats(label: str, rows: List[Dict]) -> None:
    s = stats(rows)
    print(
        f"[stats] {label}: "
        f"count={s['count']}, "
        f"hit_count={s['hit_count']}, "
        f"hit_rate={s['hit_rate']:.3f}, "
        f"avg_energy={s['avg_energy']:.4f}"
    )


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Write full CSVs (unique by base) for: overall, per prompt, "
            "and CIF-family (cif ∪ text+cif), plus hit-rate and avg energy."
        )
    )
    ap.add_argument("--results", required=True,
                    help="Path to results_full.csv (id,base,prompt_tag,sample,...)")
    ap.add_argument("--cifdir", required=True,
                    help="Dir containing CIFs named <id>.cif (for formula fill)")
    ap.add_argument("--outdir", required=True, help="Output directory")
    args = ap.parse_args()

    results_csv = Path(args.results)
    cifdir = Path(args.cifdir)
    outdir = Path(args.outdir)

    rows = read_rows(results_csv)
    if not rows:
        raise SystemExit("[error] no usable rows")

    attach_formulas_if_missing(rows, cifdir)

    # Overall (unique by base)
    overall = unique_sorted(rows)
    write_csv(outdir / "overall_unique.csv", overall)
    print_stats("overall_unique", overall)

    # Per-prompt (unique by base inside each prompt_tag)
    for tag in ("cif", "text", "text+cif"):
        subset = [r for r in rows if r["prompt_tag"] == tag]
        unique_subset = unique_sorted(subset)
        write_csv(outdir / f"prompt_{tag}_unique.csv", unique_subset)
        print_stats(f"prompt_{tag}_unique", unique_subset)

    # CIF-family (cif ∪ text+cif), unique by base:
    # for bases that appear in both, this picks the most stable of the two.
    cif_family = [r for r in rows if r["prompt_tag"] in {"cif", "text+cif"}]
    cif_family_unique = unique_sorted(cif_family)
    write_csv(outdir / "cif_family_unique.csv", cif_family_unique)
    print_stats("cif_family_unique", cif_family_unique)

    print("✅ Wrote full CSVs (unique-by-base, sorted by stability):")
    print(f"  - {outdir/'overall_unique.csv'}")
    print(f"  - {outdir/'prompt_cif_unique.csv'}")
    print(f"  - {outdir/'prompt_text_unique.csv'}")
    print(f"  - {outdir/'prompt_text+cif_unique.csv'}")
    print(f"  - {outdir/'cif_family_unique.csv'}")


if __name__ == "__main__":
    main()
