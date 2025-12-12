#!/usr/bin/env python3
import argparse, csv, os, re, shutil, subprocess, sys
from pathlib import Path

# Matches: MP_mp-1192306__1-cif_sample0.cif
#          OQMD_1716571__1-text+cif_sample1.cif
#          NOMAD_XXXX__1-text_sample0.cif
FILE_RE = re.compile(
    r"^(?P<base>.+?)-(?:"
    r"(?P<tag>cif|text\+cif|text)_sample(?P<sample>\d+)"
    r")\.cif$"
)

def main():
    p = argparse.ArgumentParser(description="Run CGCNN on flat CIF set while preserving prompt tags in filenames.")
    p.add_argument("--cgcnn-root", required=True, help="Path to cgcnn repo (where predict.py lives)")
    p.add_argument("--generated-dir", required=True, help="Flat folder containing *-cif_sample*.cif, *-text*.cif, etc.")
    p.add_argument("--model", default="pre-trained/formation-energy-per-atom.pth.tar",
                   help="Model path (relative to --cgcnn-root or absolute)")
    p.add_argument("--dataset-dir", required=True, help="Folder to build the eval dataset (will be recreated)")
    p.add_argument("--copy", action="store_true", help="Copy CIFs instead of symlink")
    args = p.parse_args()

    cgcnn_root = Path(args.cgcnn_root).resolve()
    gen_dir     = Path(args.generated_dir).resolve()
    dataset_dir = Path(args.dataset_dir).resolve()
    model_path  = (cgcnn_root / args.model) if not os.path.isabs(args.model) else Path(args.model)

    # 0) sanity
    predict_py = cgcnn_root / "predict.py"
    atom_init_src = cgcnn_root / "data" / "sample-regression" / "atom_init.json"
    if not predict_py.exists():
        sys.exit(f"[error] {predict_py} not found.")
    if not atom_init_src.exists():
        sys.exit(f"[error] {atom_init_src} not found (bad --cgcnn-root?)")
    if not gen_dir.exists():
        sys.exit(f"[error] --generated-dir not found: {gen_dir}")

    # 1) prepare dataset dir
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    dataset_dir.mkdir(parents=True)
    shutil.copy(atom_init_src, dataset_dir / "atom_init.json")

    # 2) collect flat CIFs (preserve filenames/IDs)
    rows = []   # (id, dummy_target)
    meta = []   # (id, base, tag, sample, src_path)
    n = 0
    for f in sorted(gen_dir.glob("*.cif")):
        m = FILE_RE.match(f.name)
        if not m:
            # skip anything not following the -{tag}_sampleK.cif scheme
            continue
        base   = m.group("base")
        tag    = m.group("tag")        # 'cif' | 'text' | 'text+cif'
        sample = m.group("sample")     # string number
        cif_id = f.stem                # use full name minus .cif as the ID

        dst = dataset_dir / f.name
        if args.copy:
            shutil.copy2(f, dst)
        else:
            try:
                os.symlink(f, dst)
            except FileExistsError:
                pass
            except OSError:
                shutil.copy2(f, dst)

        rows.append((cif_id, "0.0"))
        meta.append((cif_id, base, tag, sample, str(f)))
        n += 1

    if n == 0:
        sys.exit(f"[error] No matching CIFs found in {gen_dir} (expected *-cif_sample*.cif, *-text*.cif, *-text+cif*.cif)")

    # 3) write id_prop.csv (two columns)
    id_prop = dataset_dir / "id_prop.csv"
    with id_prop.open("w", newline="") as fh:
        w = csv.writer(fh)
        for r in rows:
            w.writerow(r)

    # 4) run CGCNN
    if not model_path.exists():
        sys.exit(f"[error] model not found at {model_path}")
    print(f"Running CGCNN prediction on {n} CIFs…")
    cmd = [
        sys.executable, str(predict_py),
        str(model_path),
        str(dataset_dir),
        "--workers", "0",
        "--print-freq", "1000",
    ]
    subprocess.check_call(cmd, cwd=str(cgcnn_root))

    # 5) read cgcnn outputs
    tr_path = cgcnn_root / "test_results.csv"
    if not tr_path.exists():
        sys.exit("[error] CGCNN didn't produce test_results.csv in cgcnn root.")

    # map id -> (base, tag, sample)
    meta_map = {cid: (base, tag, sample) for (cid, base, tag, sample, _src) in meta}

    full_rows = []
    with tr_path.open() as fh:
        r = csv.reader(fh)
        for row in r:
            if len(row) < 3:
                continue
            cid, _target, pred = row[0], row[1], row[2]
            base, tag, sample = meta_map.get(cid, ("?", "?", "?"))
            full_rows.append([cid, base, tag, sample, pred])

    # 6) write results_full.csv and per-tag splits
    out_full = dataset_dir / "results_full.csv"
    with out_full.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "base", "prompt_tag", "sample", "predicted_formation_energy_eV_per_atom"])
        w.writerows(full_rows)

    def write_tag(tag_name, fname):
        subset = [r for r in full_rows if r[2] == tag_name]
        with (dataset_dir / fname).open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["id", "base", "sample", "predicted_formation_energy_eV_per_atom"])
            for cid, base, _t, sample, pred in subset:
                w.writerow([cid, base, sample, pred])

    write_tag("cif",      "results_cif.csv")
    write_tag("text",     "results_text.csv")
    write_tag("text+cif", "results_text_plus_cif.csv")

    print("Done.")
    print(f"- Full results:            {out_full}")
    print(f"- CIF only:                {dataset_dir/'results_cif.csv'}")
    print(f"- Text only:               {dataset_dir/'results_text.csv'}")
    print(f"- Text+CIF:                {dataset_dir/'results_text_plus_cif.csv'}")

if __name__ == "__main__":
    main()
