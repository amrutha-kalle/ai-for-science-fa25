#!/usr/bin/env python3
import argparse, csv, os, re, shutil, subprocess, sys
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cgcnn-root", required=True, help="Path to cgcnn repo (where predict.py lives)")
    p.add_argument("--generated-dir", default="generated_cifs", help="Folder with per-CIF subdirs")
    p.add_argument("--model", default="pre-trained/formation-energy-per-atom.pth.tar",
                   help="Model path relative to --cgcnn-root or absolute")
    p.add_argument("--dataset-dir", default="cgcnn_eval", help="Where to build the eval dataset")
    p.add_argument("--copy", action="store_true", help="Copy CIFs instead of symlink (use on Windows)")
    args = p.parse_args()

    cgcnn_root = Path(args.cgcnn_root).resolve()
    generated_dir = Path(args.generated_dir).resolve()
    dataset_dir = Path(args.dataset_dir).resolve()
    model_path = (cgcnn_root / args.model) if not os.path.isabs(args.model) else Path(args.model)

    # 1) Prepare dataset dir
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    dataset_dir.mkdir(parents=True)

    # atom_init.json — reuse cgcnn's provided file
    atom_init_src = cgcnn_root / "data" / "sample-regression" / "atom_init.json"
    if not atom_init_src.exists():
        print(f"ERROR: {atom_init_src} not found. Point --cgcnn-root to the cgcnn repo.", file=sys.stderr)
        sys.exit(1)
    shutil.copy(atom_init_src, dataset_dir / "atom_init.json")

    # 2) Collect CIFs and write id_prop.csv
    id_prop_path = dataset_dir / "id_prop.csv"
    rows = []
    pattern = re.compile(r"^(prompt(?P<prompt>[12])_sample(?P<sample>\d+))\.cif$")
    n_found = 0

    for cif_dir in sorted(generated_dir.iterdir()):
        if not cif_dir.is_dir(): 
            continue
        for f in sorted(cif_dir.glob("*.cif")):
            m = pattern.match(f.name)
            if not m:
                # skip any non-conforming CIF names
                continue
            prompt = m.group("prompt")
            sample = m.group("sample")
            # build a unique ID that encodes base, prompt, sample
            base = cif_dir.name
            cif_id = f"{base}__p{prompt}s{sample}"
            # link or copy into dataset_dir
            dst = dataset_dir / f"{cif_id}.cif"
            if args.copy:
                shutil.copy(f, dst)
            else:
                try:
                    os.symlink(f, dst)
                except FileExistsError:
                    pass
                except OSError:
                    # fallback to copy if symlink not allowed
                    shutil.copy(f, dst)
            # id_prop second column can be dummy for predict.py
            rows.append((cif_id, "0.0", f"{base}", prompt, sample))
            n_found += 1

    if n_found == 0:
        print(f"No CIFs found under {generated_dir} matching prompt*/sample*.cif", file=sys.stderr)
        sys.exit(1)

    # write id_prop.csv (only first two columns are used by CGCNN; we’ll keep extras for our parsing later)
    with open(id_prop_path, "w", newline="") as fh:
        w = csv.writer(fh)
        for r in rows:
            w.writerow(r[:2])  # CGCNN expects: id, dummy_target

    # 3) Run predict.py
    predict_py = cgcnn_root / "predict.py"
    if not predict_py.exists():
        print(f"ERROR: {predict_py} not found", file=sys.stderr)
        sys.exit(1)
    if not model_path.exists():
        print(f"ERROR: model not found at {model_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Running CGCNN prediction on {n_found} CIFs…")
    cmd = [
        sys.executable, str(predict_py),
        str(model_path),
        str(dataset_dir),
        "--workers", "0",
        "--print-freq", "1000",  # quiet
    ]
    # IMPORTANT: New PyTorch defaults may require weights_only=False in predict.py (you already patched that).
    subprocess.check_call(cmd, cwd=str(cgcnn_root))

    # 4) Postprocess test_results.csv → split by prompt/sample
    tr_path = cgcnn_root / "test_results.csv"
    if not tr_path.exists():
        print("ERROR: CGCNN did not write test_results.csv in cgcnn root.", file=sys.stderr)
        sys.exit(1)

    # Map back ID → (base,prompt,sample)
    meta = {}
    for base, _, base_name, prompt, sample in rows:
        meta[base] = (base_name, prompt, sample)

    full_rows = []
    with open(tr_path, "r") as fh:
        r = csv.reader(fh)
        for cif_id, target, pred in r:
            # cif_id is like "<base>__p1s0"
            base_name, prompt, sample = meta.get(cif_id, ("?", "?", "?"))
            full_rows.append([cif_id, base_name, prompt, sample, target, pred])

    out_full = dataset_dir / "results_full.csv"
    with open(out_full, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "original_base", "prompt", "sample", "target_dummy", "predicted_formation_energy_eV_per_atom"])
        w.writerows(full_rows)

    # prompt splits
    def write_subset(prompt_val, fname):
        subset = [row for row in full_rows if row[2] == prompt_val]
        with open(dataset_dir / fname, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["id", "original_base", "sample", "predicted_formation_energy_eV_per_atom"])
            for _, base, _, sample, _, pred in subset:
                w.writerow([f"{base}__p{prompt_val}s{sample}", base, sample, pred])

    write_subset("1", "results_prompt1.csv")
    write_subset("2", "results_prompt2.csv")

    # simple pivot-like summary avg by (prompt,sample)
    import statistics as stats
    summary = {}
    for _, base, prompt, sample, _, pred in full_rows:
        key = (prompt, sample)
        summary.setdefault(key, []).append(float(pred))
    out_pivot = dataset_dir / "results_by_prompt_and_sample.csv"
    with open(out_pivot, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["prompt", "sample", "n", "mean_pred_eV_per_atom", "median_pred", "min", "max"])
        for (prompt, sample), vals in sorted(summary.items()):
            w.writerow([prompt, sample, len(vals),
                        f"{stats.mean(vals):.6f}",
                        f"{stats.median(vals):.6f}",
                        f"{min(vals):.6f}",
                        f"{max(vals):.6f}"])

    print("Done.")
    print(f"- Full results: {out_full}")
    print(f"- Prompt 1 only: {dataset_dir/'results_prompt1.csv'}")
    print(f"- Prompt 2 only: {dataset_dir/'results_prompt2.csv'}")
    print(f"- Summary by (prompt,sample): {out_pivot}")

if __name__ == "__main__":
    main()
