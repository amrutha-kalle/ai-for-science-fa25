#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path
from typing import Optional, Dict, List

from pymatgen.io.cif import CifParser
from pymatgen.core.composition import Composition  # NEW

# ---------- helpers ----------

ID_TAIL_RE = re.compile(r"__p(\d+)s(\d+)$")

def to_alphabetical(formula_str: str) -> str:
    """
    Return alphabetical canonical formula (reduced), no spaces,
    e.g. 'U2 Br2 O3' -> 'Br2O3U2'.
    """
    if not formula_str:
        return ""
    try:
        return Composition(formula_str).alphabetical_formula.replace(" ", "")
    except Exception:
        # If parsing ever fails, just strip spaces as a last resort
        return formula_str.replace(" ", "")

def parse_id(id_str: str):
    """Return (base, prompt:int, sample:int) from an id like 'BASE__p2s1' (with or without .cif)."""
    stem = id_str[:-4] if id_str.endswith(".cif") else id_str
    m = ID_TAIL_RE.search(stem)
    if not m:
        return None
    prompt = int(m.group(1))
    sample = int(m.group(2))
    base = stem[: m.start()]
    return base, prompt, sample

def cif_path_for_row(cifdir: Path, row: Dict) -> Optional[Path]:
    """Resolve CIF path from id using flat layout first, then fallbacks."""
    id_str = row.get("id", "")
    parsed = parse_id(id_str)
    if not parsed:
        return None
    base, prompt, sample = parsed

    # 1) Flat layout (your current case)
    flat = cifdir / f"{base}__p{prompt}s{sample}.cif"
    if flat.exists():
        return flat

    # 2) Some bases include a trailing '__1' that isn't in filenames — try removing it
    if base.endswith("__1"):
        alt_flat = cifdir / f"{base[:-3]}__p{prompt}s{sample}.cif"
        if alt_flat.exists():
            return alt_flat

    # 3) Old nested layout (keep for compatibility)
    nested = cifdir / "generated_cifs" / base / f"prompt{prompt}_sample{sample}.cif"
    if nested.exists():
        return nested

    # 4) Recursive fallback
    hits = list(cifdir.rglob(f"{base}__p{prompt}s{sample}.cif"))
    if hits:
        return hits[0]
    hits = list(cifdir.rglob(f"generated_cifs/{base}/prompt{prompt}_sample{sample}.cif"))
    if hits:
        return hits[0]
    return None

def formula_from_cif(cif_path: Optional[Path]) -> str:
    if not cif_path or not cif_path.exists():
        return ""
    try:
        struct = CifParser(str(cif_path)).get_structures(primitive=False)[0]
        # Use pymatgen's alphabetical reduced formula (no spaces)
        return struct.composition.alphabetical_formula.replace(" ", "")
    except Exception:
        return ""

def attach_formula(rows: List[Dict], cifdir: Path) -> List[Dict]:
    hits = misses = 0
    for r in rows:
        f = (r.get("formula") or "").strip()
        if f:
            f = to_alphabetical(f)  # normalize existing
        else:
            cifp = cif_path_for_row(cifdir, r)
            f = formula_from_cif(cifp)
        if f:
            r["formula"] = f
            hits += 1
        else:
            r["formula"] = ""
            misses += 1
    print(f"[extract_topN] attach formula: hits={hits}, misses={misses}")
    return rows

def read_rows(csv_path: Path) -> List[Dict]:
    with open(csv_path, newline="") as f:
        r = csv.DictReader(f)
        rows = [dict(row) for row in r]
    # basic sanity
    need = {"id", "predicted_formation_energy_eV_per_atom"}
    missing = need - set(rows[0].keys() if rows else set())
    if missing:
        raise SystemExit(f"[error] CSV missing columns: {missing}")
    # normalize prompt and energy
    cleaned = []
    for row in rows:
        id_str = row.get("id", "").strip()
        if not id_str:
            continue
        # prompt: use column if present, else parse from id
        p_str = (row.get("prompt") or "").strip()
        if p_str.isdigit():
            prompt = int(p_str)
        else:
            parsed = parse_id(id_str)
            if not parsed:
                continue
            _, prompt, _ = parsed
        # energy
        try:
            energy = float(row["predicted_formation_energy_eV_per_atom"])
        except Exception:
            continue
        cleaned.append({
            "id": id_str,
            "prompt": prompt,
            "predicted_formation_energy_eV_per_atom": energy,
            "formula": to_alphabetical((row.get("formula") or "").strip()),  # normalize if present
        })
    return cleaned

def write_csv(path: Path, rows: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["id", "prompt", "formula", "predicted_formation_energy_eV_per_atom"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({
                "id": r["id"],
                "prompt": r["prompt"],
                "formula": r.get("formula", ""),
                "predicted_formation_energy_eV_per_atom": r["predicted_formation_energy_eV_per_atom"],
            })

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(
        description="Extract top-N most stable Chemeleon candidates from CGCNN results."
    )
    ap.add_argument("--results", required=True, help="Path to results_full.csv")
    ap.add_argument("--cifdir", required=True, help="Directory containing generated CIFs")
    ap.add_argument("--outdir", default="cgcnn_topN_chemeleon", help="Output directory")
    ap.add_argument("--topN", type=int, default=20, help="Top N to keep (most negative energy)")
    args = ap.parse_args()

    results_csv = Path(args.results)
    cifdir = Path(args.cifdir)
    outdir = Path(args.outdir)
    topN = args.topN

    rows = read_rows(results_csv)
    if not rows:
        raise SystemExit("[error] No usable rows parsed from results CSV.")

    # fill in (and normalize) formulas from CIFs
    rows = attach_formula(rows, cifdir)

    # de-duplicate per (base, prompt), keep the most stable (min energy)
    best_by_key: Dict[tuple, Dict] = {}
    for r in rows:
        parsed = parse_id(r["id"])
        if not parsed:
            continue
        base, prompt, _ = parsed
        key = (base, prompt)
        if key not in best_by_key or r["predicted_formation_energy_eV_per_atom"] < best_by_key[key]["predicted_formation_energy_eV_per_atom"]:
            best_by_key[key] = r

    dedup_rows = list(best_by_key.values())

    # overall topN (from both prompts)
    overall = sorted(dedup_rows, key=lambda x: x["predicted_formation_energy_eV_per_atom"])[:topN]
    # per-prompt topN
    p1 = sorted([r for r in dedup_rows if r["prompt"] == 1], key=lambda x: x["predicted_formation_energy_eV_per_atom"])[:topN]
    p2 = sorted([r for r in dedup_rows if r["prompt"] == 2], key=lambda x: x["predicted_formation_energy_eV_per_atom"])[:topN]

    write_csv(outdir / "topN_overall.csv", overall)
    write_csv(outdir / "topN_prompt1.csv", p1)
    write_csv(outdir / "topN_prompt2.csv", p2)

    print("✅ Wrote:")
    print(f"  - {outdir/'topN_overall.csv'}")
    print(f"  - {outdir/'topN_prompt1.csv'}")
    print(f"  - {outdir/'topN_prompt2.csv'}")


if __name__ == "__main__":
    main()
