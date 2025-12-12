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


def write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["id","base","prompt_tag","sample","formula","predicted_formation_energy_eV_per_atom"]
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


def topN_unique(rows: List[Dict], N: int) -> List[Dict]:
    return sorted(dedup_by_base(rows), key=lambda r: r["energy"])[:N]


def main():
    ap = argparse.ArgumentParser(
        description="Write top-N CSVs (unique by base): overall, per prompt, and CIF-family (cif ∪ text+cif)."
    )
    ap.add_argument("--results", required=True, help="Path to results_full.csv (id,base,prompt_tag,sample,...)")
    ap.add_argument("--cifdir", required=True, help="Dir containing CIFs named <id>.cif (for formula fill)")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--topN", type=int, default=20, help="How many most-stable to keep")
    args = ap.parse_args()

    results_csv = Path(args.results)
    cifdir = Path(args.cifdir)
    outdir = Path(args.outdir)

    rows = read_rows(results_csv)
    if not rows:
        raise SystemExit("[error] no usable rows")

    attach_formulas_if_missing(rows, cifdir)

    N = args.topN

    # Overall top-N (unique by base)
    write_csv(outdir / "topN_overall.csv", topN_unique(rows, N))

    # Per-prompt top-N (unique by base inside each prompt_tag)
    for tag in ("cif", "text", "text+cif"):
        subset = [r for r in rows if r["prompt_tag"] == tag]
        write_csv(outdir / f"topN_prompt_{tag}.csv", topN_unique(subset, N))

    # CIF-family (cif ∪ text+cif)
    cif_family = [r for r in rows if r["prompt_tag"] in {"cif", "text+cif"}]
    write_csv(outdir / "topN_cif_family.csv", topN_unique(cif_family, N))

    print("✅ Wrote:")
    print(f"  - {outdir/'topN_overall.csv'}")
    print(f"  - {outdir/'topN_prompt_cif.csv'}")
    print(f"  - {outdir/'topN_prompt_text.csv'}")
    print(f"  - {outdir/'topN_prompt_text+cif.csv'}")
    print(f"  - {outdir/'topN_cif_family.csv'}")


if __name__ == "__main__":
    main()
