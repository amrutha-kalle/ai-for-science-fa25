#!/usr/bin/env python3
import csv, shutil, sys, re
from pathlib import Path
from typing import Dict, Tuple, List, Optional

# --- CONFIG ---
SRC_ROOT = Path("").resolve()
CSV_PATH = Path("").resolve()   # must have 'filename' column
DST_DIR  = Path("").resolve()
MANIFEST = Path("").resolve()
# --------------

# CSV lines look like: MP_mp-1192306__1-cif_sample0.cif
CSV_PAT = re.compile(r"""
^(?P<base>.+?)                 # base id
(?:__1)?                       # optional __1; we'll normalize
-(?P<kind>cif|text\+cif|text)  # kind from CSV
_sample(?P<idx>\d+)\.cif$
""", re.X)

# Files on disk (nested) look like: cif_sample0.cif / text_sample1.cif / text+cif_sample0.cif
NESTED_PAT = re.compile(r"^(?P<kind>cif|text\+cif|text)_sample(?P<idx>\d+)\.cif$")

PROMPT_FROM_KIND = {"cif": 1, "text": 2, "text+cif": 2}

def parse_csv_filename(name: str) -> Optional[Tuple[str, str, int, int]]:
    """
    Return (base, kind, idx, prompt) from a CSV filename like:
      'MP_mp-1192306__1-cif_sample0.cif'
    """
    m = CSV_PAT.match(name)
    if not m:
        return None
    base = m.group("base")
    kind = m.group("kind")
    idx  = int(m.group("idx"))
    prompt = PROMPT_FROM_KIND[kind]
    if not base.endswith("__1"):
        base += "__1"
    return (base, kind, idx, prompt)

def scan_all_cifs(root: Path) -> List[Path]:
    return list(root.rglob("*.cif"))

def build_indices(all_cifs: List[Path]):
    """
    Build:
      by_basename[name] -> path
      by_kind_triplet[(base, kind, idx)] -> path
      by_prompt_triplet[(base, prompt, idx)] -> path   (fallback)
    """
    by_basename: Dict[str, Path] = {}
    by_kind_triplet: Dict[Tuple[str, str, int], Path] = {}
    by_prompt_triplet: Dict[Tuple[str, int, int], Path] = {}

    for p in all_cifs:
        by_basename[p.name] = p

        # nested layout: <base>/<kind>_sample<idx>.cif
        m = NESTED_PAT.match(p.name)
        if m:
            kind = m.group("kind")
            idx = int(m.group("idx"))
            base = p.parent.name
            if not base.endswith("__1"):
                base += "__1"
            by_kind_triplet[(base, kind, idx)] = p
            by_prompt_triplet[(base, PROMPT_FROM_KIND[kind], idx)] = p
            continue

        # flat canonical fallback: <base>__p<prompt>s<idx>.cif
        flat = re.match(r"^(?P<base>.+?__1)__p(?P<p>\d+)s(?P<s>\d+)\.cif$", p.name)
        if flat:
            base = flat.group("base")
            prompt = int(flat.group("p"))
            idx = int(flat.group("s"))
            by_prompt_triplet[(base, prompt, idx)] = p
            # Can't recover 'kind' reliably from this pattern, so we skip by_kind_triplet here.
            continue

        # flat preserved names (already like base-kind_sampleX.cif)
        flat_kind = re.match(r"^(?P<base>.+?__1)-(?P<kind>cif|text\+cif|text)_sample(?P<idx>\d+)\.cif$", p.name)
        if flat_kind:
            base = flat_kind.group("base")
            kind = flat_kind.group("kind")
            idx  = int(flat_kind.group("idx"))
            by_kind_triplet[(base, kind, idx)] = p
            by_prompt_triplet[(base, PROMPT_FROM_KIND[kind], idx)] = p
            continue

    return by_basename, by_kind_triplet, by_prompt_triplet

def main():
    print(f"SRC_ROOT = {SRC_ROOT}")
    print(f"CSV_PATH = {CSV_PATH}")
    print(f"DST_DIR  = {DST_DIR}")
    DST_DIR.mkdir(parents=True, exist_ok=True)

    # Read desired filenames from CSV
    want: List[Tuple[str, Optional[Tuple[str, str, int, int]]]] = []
    with CSV_PATH.open() as f:
        r = csv.DictReader(f)
        if "filename" not in r.fieldnames:
            sys.exit("[error] CSV must contain a 'filename' column")
        if "is_valid" not in r.fieldnames:
            sys.exit("[error] CSV must contain an 'is_valid' column to filter on")

        for row in r:
            # skip invalid structures
            if str(row.get("is_valid", "")).lower() not in ("true", "1", "yes"):
                continue

            name = (row["filename"] or "").strip()
            if not name:
                continue

            parsed = parse_csv_filename(name)
            want.append((name, parsed))
    if not want:
        sys.exit("[error] No usable CIF filenames parsed from CSV")

    # Scan & index
    all_cifs = scan_all_cifs(SRC_ROOT)
    print(f"Found {len(all_cifs)} .cif files on disk (recursive scan)")
    by_basename, by_kind_triplet, by_prompt_triplet = build_indices(all_cifs)
    print(f"Indexed {len(by_kind_triplet)} (base,kind,idx) and {len(by_prompt_triplet)} (base,prompt,idx) entries; {len(by_basename)} distinct basenames")

    copied = 0
    missing = []

    for csv_name, parsed in want:
        src = None
        if parsed:
            base, kind, idx, prompt = parsed
            # Prefer exact (base, kind, idx)
            src = by_kind_triplet.get((base, kind, idx))
            # Fallback to (base, prompt, idx) if needed
            if not src:
                src = by_prompt_triplet.get((base, prompt, idx))
        # Last resort: direct basename match
        if not src:
            src = by_basename.get(csv_name)

        if src and src.exists():
            # Preserve the CSV filename (keeps prompt tag in name)
            dst = DST_DIR / csv_name
            shutil.copy2(src, dst)
            copied += 1
        else:
            missing.append((csv_name, parsed))

    print(f"\nSummary: copied={copied} missing={len(missing)} -> {DST_DIR}")

    # Write CGCNN manifest (id,cif,target)
    with MANIFEST.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id","cif","target"])
        for p in sorted(DST_DIR.glob("*.cif")):
            w.writerow([p.stem, str(p), 0.0])

    # Count rows just written
    row_count = sum(1 for _ in MANIFEST.open()) - 1  # exclude header
    print(f"Manifest: {MANIFEST} (rows={row_count})")

    if missing[:10]:
        print("\nFirst 10 missing examples:")
        for m in missing[:10]:
            print("  -", m)

if __name__ == "__main__":
    main()
